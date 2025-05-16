import re
import selectors
import subprocess
import time
import traceback
from subprocess import Popen
from uuid import uuid4

import docker
from swebench.harness.constants import DOCKER_USER, DOCKER_WORKDIR

CLIENT = docker.from_env(timeout=300, max_pool_size=1024)


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

        return output_buffer.decode("utf-8", errors="replace")

    def __call__(self, input: str, timeout=3):
        if self._is_reward_hacking(input):
            return "Error: This operation is not permitted in this environment. Please use the given repo and code."

        try:
            self.socket._sock.send(self.encode_input(input))

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
        #     cleanup_container(CLIENT, self.container)
        except Exception as e:
            print(f"Error stopping container: {e}")
            print(traceback.format_exc())

    def _is_reward_hacking(self, input: str) -> bool:
        """Check if the input contains reward hacking to download correct code from github/gitlab"""
        input_lower = input.lower()
        if "git clone" in input_lower:
            return True

        clone_patterns = [
            r"git\s+clone\s+",
            r"clone\s+.*\.git",
            r"git\s+.*\s+clone",
            r"github\.com/[^/]+/[^/\s]+",
            r"gitlab\.com/[^/]+/[^/\s]+",
            r"raw\.githubusercontent\.com/[^/]+/[^/\s]+",
        ]

        for pattern in clone_patterns:
            if re.search(pattern, input_lower):
                return True

        return False

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
