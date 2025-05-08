import re
import selectors
import time
import traceback

import docker
from swebench.harness.constants import DOCKER_USER, DOCKER_WORKDIR, UTF8

CLIENT = docker.from_env(max_pool_size=1024, timeout=100000)


class Terminal:
    def __init__(self, image: str = "ubuntu:latest", commit: str | None = None):
        self.container = CLIENT.containers.run(
            image,
            command="/bin/bash",
            detach=True,
            tty=True,
            stdin_open=True,
            platform="linux/x86_64",
            user=DOCKER_USER,
            mem_limit="10g",
            working_dir=DOCKER_WORKDIR,
        )
        if commit is not None:
            self.exec_run("git fetch")
            val = self.exec_run(f"git checkout {commit}")
            if val is None or val.exit_code != 0:
                print(f"CHECKOUT FAILED: {val.output.decode(UTF8)}")

            # allow git commits
            self.exec_run('git config --global user.email "you@example.com"')
            self.exec_run('git config --global user.name "Your Name"')
            # Make sure the main/master branch is deleted
            self.exec_run("git branch -D main master")
            # Remove the remote origin to prevent pulling main again
            self.exec_run("git remote remove origin")
            # Clear the reflog and perform garbage collection
            self.exec_run("git reflog expire --expire=now --all")
            self.exec_run("git gc --prune=now --aggressive")

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
                    result.append(ord(text[i]) - 64)
                else:
                    result.append(ord(text[i]))
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

    def __call__(self, input: str, timeout=1):
        # print("=" * 50)
        # print(f"typing {repr(input)}")

        self.socket._sock.send(self.encode_input(input))

        output = self.read_container_output(timeout=timeout)
        output = output.replace("\r", "")  # fix carriage-return
        if output.startswith(input):
            output = output[len(input) :]
        output = self._strip_ansi(output)
        return output

    def stop(self, timeout=0):
        self.container.stop(timeout=timeout)
        self.container.remove(force=True)

    def get_patch(self, base_commit: str):
        try:
            return self.exec_run(
                # we cannot do `git add -A` here since that may add unadded files
                # instead, have the agent add new files to git itself
                f"bash -c 'git diff {base_commit}'",
            ).output.decode("utf-8", errors="replace")
        except docker.errors.APIError as e:
            if "is not running" in str(e):
                return ""
            raise


def interact():
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


if __name__ == "__main__":
    # asyncio.run(main())
    interact()
