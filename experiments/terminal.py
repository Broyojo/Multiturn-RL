import base64
import re
import selectors
import subprocess
import time
import traceback
import uuid
from pathlib import Path
from subprocess import Popen
from uuid import uuid4

import aiohttp
import docker
from swebench.harness.constants import DOCKER_USER, DOCKER_WORKDIR

CLIENT = docker.from_env(timeout=300, max_pool_size=1024)

FORBIDDEN_URL_RE = re.compile(
    r"https?://[^ \t\r\n]*\b(?:github(?:usercontent)?\.com|gitlab\.com)\b"
    r"|(?:github(?:usercontent)?\.com|gitlab\.com)",
    re.I,
)

GIT_FORBIDDEN_RE = re.compile(
    r"""
    \bgit\s+(
        clone|pull|fetch|push|remote|
        log|show|blame|cat-file|rev-[^\s]+|
        checkout\s+[^\s]*(?:\^|~|origin/|[a-f0-9]{7,40})|
        diff\s+[^\s]*(?:\^|~|[a-f0-9]{7,40}|--cached)
    )\b
    """,
    re.I | re.X,
)


def _is_reward_hacking(command: str) -> bool:
    cmd = command.strip().lower()
    return FORBIDDEN_URL_RE.search(cmd) or GIT_FORBIDDEN_RE.search(cmd)


class Terminal:
    def __init__(self, image: str, commit: str | None = None):
        self.container = CLIENT.containers.run(
            image,
            name=f"terminal-{uuid4()}",
            command="/bin/bash",
            detach=True,
            tty=True,
            stdin_open=True,
            platform="linux/x86_64",
            user=DOCKER_USER,
            working_dir=DOCKER_WORKDIR,
            environment={"TERM": "xterm", "LC_ALL": "C.UTF-8"},
            restart_policy={"Name": "unless-stopped"},
        )
        if commit is not None:
            git_setup_cmd = f"""git fetch && 
            git checkout {commit} && 
            git branch -D main master || true && 
            git remote remove origin || true &&
            git checkout -b main"""
            self.exec_run(f"bash -c '{git_setup_cmd}'")

        self.socket = self.container.attach_socket(params={"stdin": 1, "stdout": 1, "stderr": 1, "stream": 1})
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def exec_run(self, command):
        try:
            return self.container.exec_run(
                command,
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
            )
        except Exception:
            print(traceback.format_exc())
            print(self.container.name)

    def _strip_ansi(self, text):
        return self.ansi_escape.sub("", text)

    def _clean_control(self, stream: str) -> str:
        buf = []
        for ch in stream:
            if ch == "\b":
                if buf:
                    buf.pop()
            else:
                buf.append(ch)
        return "".join(buf)

    def encode_input(self, text: str) -> bytes:
        if text.startswith("^"):
            result = bytearray()
            i = 0
            while i < len(text):
                if text[i] == "^":
                    i += 1
                    if i >= len(text):
                        break
                    if 0 <= (char_ord := ord(text[i])) <= 255 and char_ord >= 64:
                        result.append(char_ord - 64)
                else:
                    if 0 <= (char_ord := ord(text[i])) <= 255:
                        result.append(char_ord)
                i += 1
            return bytes(result)
        return text.encode()

    def read_container_output(self, timeout=3):
        output_buffer = b""
        start_time = time.time()

        sel = selectors.DefaultSelector()
        sel.register(self.socket._sock, selectors.EVENT_READ)

        try:
            while True:
                # waits up to 0.1s for the socket to be readable
                events = sel.select(timeout=0.1)
                for key, _ in events:
                    chunk = key.fileobj.recv(4096)
                    if not chunk:
                        # EOF
                        sel.unregister(key.fileobj)
                        break
                    output_buffer += chunk
                if time.time() - start_time >= timeout:
                    break
        finally:
            sel.close()

        return self._clean_control(output_buffer.decode("utf-8", errors="replace"))

    def __call__(self, input: str, timeout=1):
        if _is_reward_hacking(input):
            return "Error: This operation is not permitted in this environment. Please use the given repo and code."

        try:
            self.socket._sock.sendall(self.encode_input(input))

            output = self.read_container_output(timeout=timeout)
            output = output.replace("\r", "")  # fix carriage-return
            if output.startswith(input):
                output = output[len(input) :]
            output = self._strip_ansi(output)
            return output
        except Exception:
            print(traceback.format_exc())
            print(self.container.name)
            return ""

    def stop(self):
        try:
            if hasattr(self, "socket") and self.socket:
                try:
                    self.socket._sock.close()
                except Exception as e:
                    print(f"Error closing socket: {e}")

            Popen(
                f"docker stop {self.container.id} -t 15 && docker rm {self.container.id}",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"Error stopping container: {e}")
            print(traceback.format_exc())

    def get_patch(self, base_commit: str):
        try:
            result = self.exec_run(
                # we cannot do `git add -A` here since that may add unadded files
                # instead, have the agent add new files to git itself
                f"bash -c 'git diff {base_commit}'",
            )
            if result is None:
                return ""
            return result.output.decode("utf-8", errors="replace")
        except Exception:
            print(traceback.format_exc())
            print(self.container.name)
            return ""


class RemoteTerminal:
    REQUEST_TYPE_RUN_COMMAND = 0
    REQUEST_TYPE_GET_OUTPUT = 1
    REQUEST_TYPE_START_SANDBOX = 2
    REQUEST_TYPE_SHUTDOWN_SANDBOX = 3

    COMMAND_EXECUTION_ERROR = 400
    COMMAND_EXECUTION_FINISH = 200
    COMMAND_EXECUTION_TIMEOUT = 408
    INSTANCE_START_ERROR = 500
    INTERNAL_ERROR = 500

    def __init__(self, image: str, commit: str, url: str, http_timeout=300, network_disabled=False):
        self.image = image
        self.commit = commit
        self.url = url
        self.network_disabled = network_disabled
        self.trajectory_id = str(uuid.uuid4())
        self.http_timeout = http_timeout
        self.session: aiohttp.ClientSession | None = None
        self.sandbox_started = False

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.http_timeout))
        return self.session

    async def _make_request(self, payload: dict) -> dict:
        session = await self._get_session()

        try:
            async with session.post(self.url, json=payload, headers={"Content-Type": "application/json"}) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
        except aiohttp.ClientError as e:
            print(f"Request failed: {str(e)}")
            raise

    async def _start_sandbox(self) -> dict:
        payload = {
            "id": str(uuid.uuid4()),
            "trajectory": self.trajectory_id,
            "request_type": self.REQUEST_TYPE_START_SANDBOX,
            "start_sandbox_input": {
                "image_id": self.image,
                "user": DOCKER_USER,
                "working_dir": DOCKER_WORKDIR,
                "network_disabled": self.network_disabled,
                "shell_path": "/bin/bash",
            },
        }

        response = await self._make_request(payload)

        if response.get("return_reason") == self.INSTANCE_START_ERROR:
            raise Exception(f"Failed to start sandbox: {response.get('error', 'Unknown error')}")

        if self.commit:
            git_setup_cmd = f"""git fetch && 
            git checkout {self.commit} && 
            git branch -D main master || true && 
            git remote remove origin || true &&
            git checkout -b main"""
            await self._run_command(f"bash -c '{git_setup_cmd}'")

        self.sandbox_started = True
        return response

    async def _run_command(self, command: str, working_dir: str, timeout: int = 30, is_interactive=False) -> dict:
        # TODO: should this be able to specify user to run command as?
        payload = {
            "id": str(uuid.uuid4()),
            "trajectory": self.trajectory_id,
            "request_type": self.REQUEST_TYPE_RUN_COMMAND,
            "run_command_input": {
                "command": command,
                "timeout_in_seconds": timeout,
                "working_dir": working_dir,
                "network_disabled": False,
                "shell_path": "/bin/bash",
                "is_interactive": is_interactive,
                "env": ["TERM=xterm", "LC_ALL=C.UTF-8"],
            },
        }
        return await self._make_request(payload)

    async def exec_run(self, command: str, working_dir: str, timeout=30):
        try:
            if not self.sandbox_started:
                await self._start_sandbox()

            response = await self._run_command(command, working_dir=working_dir, timeout=timeout)

            class ExecResult:
                def __init__(self, output: str, exit_code: int):
                    self.output = output.encode("utf-8", errors="replace")
                    self.exit_code = exit_code

            return ExecResult(response.get("output", ""), response.get("exit_code", 0))
        except Exception:
            print(traceback.format_exc())
            print(self.trajectory_id)
            return None

    async def __call__(self, input: str, timeout: int = 1) -> str:
        if _is_reward_hacking(input):
            return "Error: This operation is not permitted in this environment. Please use the given repo and code."

        try:
            if not self.sandbox_started:
                await self._start_sandbox()

            response = await self._run_command(input, timeout, interactive=True)

            return_reason = response.get("return_reason")

            if return_reason == self.COMMAND_EXECUTION_FINISH:
                output = response.get("output", "")
                return output

            return ""

        except Exception:
            import traceback

            print(traceback.format_exc())
            print(self.trajectory_id)
            return ""

    async def stop(self):
        if self.sandbox_started:
            try:
                payload = {
                    "id": str(uuid.uuid4()),
                    "trajectory": self.trajectory_id,
                    "request_type": self.REQUEST_TYPE_SHUTDOWN_SANDBOX,
                }
                await self._make_request(payload)
            except Exception as e:
                print(f"Error stopping container: {e}")
            finally:
                self.sandbox_started = False

        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    async def copy_to_container(self, src: Path, dst: Path):
        if str(dst.parent) == ".":
            raise ValueError(f"Destination path parent directory cannot be empty!, dst: {dst}")

        if not src.exists():
            raise FileNotFoundError(f"Source file does not exist: {src}")

        try:
            with open(src, "rb") as f:
                file_data = f.read()

            encoded_data = base64.b64encode(file_data).decode("utf-8")

            await self.exec_run(f"mkdir -p {dst.parent}")

            chunk_size = 1024 * 1024
            if len(encoded_data) <= chunk_size:
                command = f'echo "{encoded_data}" | base64 -d > "{dst}"'
                result = await self.exec_run(command)
                if result and result.exit_code != 0:
                    raise Exception(f"Failed to copy file: {result.output.decode()}")
            else:
                await self.exec_run(f'echo -n "" > "{dst}"')

                for i in range(0, len(encoded_data), chunk_size):
                    chunk = encoded_data[i : i + chunk_size]
                    command = f'echo "{chunk}" | base64 -d >> "{dst}"'
                    result = await self.exec_run(command)
                    if result and result.exit_code != 0:
                        raise Exception(f"Failed to copy file chunk: {result.output.decode()}")

            result = await self.exec_run(f'stat -c "%s" "{dst}" 2>/dev/null || echo "0"')
            if result:
                remote_size = int(result.output.decode().strip())
                local_size = len(file_data)
                if remote_size != local_size:
                    raise Exception(f"File size mismatch: local={local_size}, remote={remote_size}")

        except Exception as e:
            import traceback

            print(traceback.format_exc())
            print(f"Error copying {src} to {dst}: {e}")
            raise


if __name__ == "__main__":
    terminal = Terminal(
        image="swesmith.x86_64.john-kurkowski__tldextract.3d1bf184",
        commit="a2e2dab2e2f3ab56ed60f6af0abe78dafbc81cb3",
    )
    try:
        while True:
            user_input = input("input: ")
            if user_input.startswith("get_patch("):
                print(terminal.get_patch(user_input.split("(")[1][:-1]))
            else:
                print(terminal(user_input + "\n"))
    finally:
        terminal.stop()
