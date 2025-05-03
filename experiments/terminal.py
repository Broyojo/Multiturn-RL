import re
import select
import time

import docker
from swebench.harness.constants import DOCKER_USER, DOCKER_WORKDIR, UTF8

CLIENT = docker.from_env(max_pool_size=1024)


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
            self.container.exec_run(
                "git fetch", workdir=DOCKER_WORKDIR, user=DOCKER_USER
            )
            val = self.container.exec_run(
                f"git checkout {commit}", workdir=DOCKER_WORKDIR, user=DOCKER_USER
            )
            if val.exit_code != 0:
                print(f"CHECKOUT FAILED: {val.output.decode(UTF8)}")

            # allow git commits
            self.container.exec_run(
                'git config --global user.email "you@example.com"',
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
            )
            self.container.exec_run(
                'git config --global user.name "Your Name"',
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
            )
        self.socket = self.container.attach_socket(
            params={"stdin": 1, "stdout": 1, "stderr": 1, "stream": 1}
        )
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

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

        while True:
            readable, _, _ = select.select([self.socket._sock], [], [], 0.1)
            if readable:
                chunk = self.socket._sock.recv(4096)
                if not chunk:
                    break
                output_buffer += chunk
            if time.time() - start_time >= timeout:
                break

        return output_buffer.decode("utf-8", errors="replace")

    def __call__(self, input: str, timeout=1):
        print("=" * 50)
        print(f"typing {repr(input)}")

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
        return self.container.exec_run(
            # we cannot do `git add -A` here since that may add unadded files
            # instead, have the agent add new files to git itself
            f"bash -c 'cd {DOCKER_WORKDIR} && git diff {base_commit}'",
            user=DOCKER_USER,
        ).output.decode("utf-8")


def interact():
    with Terminal(
        image="swesmith.x86_64.john-kurkowski__tldextract.3d1bf184",
        commit="a2e2dab2e2f3ab56ed60f6af0abe78dafbc81cb3",
    ) as terminal:
        while True:
            user_input = input("input: ")
            if user_input.startswith("get_patch("):
                print(terminal.get_patch(user_input.split("(")[1][:-1]))
            else:
                print(terminal(user_input + "\n"))


if __name__ == "__main__":
    # asyncio.run(main())
    interact()
