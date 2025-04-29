import re
import select
import time

import docker

CLIENT = docker.from_env(max_pool_size=1024)

"""
TODO:

maybe make the docker container serving into an API server? so we can use kubernetes and more high speed.
we just connect to the TTY's through websockets?

"""


class Terminal:
    def __init__(self, image: str = "python:3.12"):
        self.container = CLIENT.containers.run(
            image,
            command="/bin/bash",
            detach=True,
            tty=True,
            stdin_open=True,
            remove=True,
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

    def __enter__(self):
        return self

    def stop(self, timeout=0):
        self.container.stop(timeout=timeout)

    def __exit__(self, exc_type, exc_val, traceback):
        self.stop()


# client = docker.from_env()

# class Terminal:
#     def __init__(self, image: str = "python:3.12"):
#         self.container = client.containers.create(
#             image,
#             command="/bin/bash",
#             detach=True,
#             tty=True,
#             stdin_open=True,
#         )
#         self.socket = self.container.attach_socket(
#             params={"stdin": 1, "stdout": 1, "stderr": 1, "stream": 1}
#         )
#         self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

#     def start(self):
#         self.container.start()

#     def stop(self, timeout=0):
#         self.container.stop(timeout=timeout)
#         self.container.remove(force=True)

#     def _strip_ansi(self, text):
#         return self.ansi_escape.sub("", text)

#     @staticmethod
#     def _encode_ctrl(text: str) -> bytes:
#         if text.startswith("^"):
#             result = bytearray()
#             i = 0
#             while i < len(text):
#                 if text[i] == "^":
#                     i += 1
#                     if i >= len(text):
#                         break
#                     result.append(ord(text[i]) - 64)
#                 else:
#                     result.append(ord(text[i]))
#                 i += 1
#             return bytes(result)
#         return text.encode()

#     def read_container_output(self, timeout=3):
#         output_buffer = b""
#         start_time = time.time()

#         while True:
#             readable, _, _ = select.select([self.socket._sock], [], [], 0.1)
#             if readable:
#                 chunk = self.socket._sock.recv(4096)
#                 if not chunk:
#                     break
#                 output_buffer += chunk
#             if time.time() - start_time >= timeout:
#                 break

#         return output_buffer.decode("utf-8", errors="replace")

#     def __call__(self, input: str, timeout=1):
#         print("=" * 50)
#         print(f"typing {repr(input)}")

#         self.socket._sock.send(self._encode_ctrl(input))

#         output = self.read_container_output(timeout=timeout)
#         output = output.replace("\r", "")  # fix carriage-return
#         if output.startswith(input):
#             output = output[len(input) :]
#         output = self._strip_ansi(output)
#         return output

#     def __enter__(self):
#         self.start()
#         return self

#     def __exit__(self, exc_type, exc_val, traceback):
#         self.stop()

# GLOBAL_POOL = ThreadPoolExecutor(max_workers=32, thread_name_prefix="docker")

# class AsyncTerminal:
#     def __init__(self, *args, **kwargs):
#         self._inner = Terminal(*args, **kwargs)

#     async def _call_in_thread(self, func, *args, **kwargs):
#         loop = asyncio.get_running_loop()
#         return await loop.run_in_executor(GLOBAL_POOL, lambda: func(*args, **kwargs))

#     async def start(self):
#         await self._call_in_thread(self._inner.start)

#     async def stop(self, timeout=0):
#         await self._call_in_thread(self._inner.stop, timeout=timeout)

#     async def __aenter__(self):
#         await self._call_in_thread(self._inner.__enter__)
#         return self

#     async def __aexit__(self, exc_type, exc_val, tb):
#         await self._call_in_thread(self._inner.__exit__, exc_type, exc_val, tb)

#     async def __call__(self, cmd: str, timeout: float = 1.0) -> str:
#         return await self._call_in_thread(self._inner, cmd, timeout)

# async def run_in_shell(idx: int) -> str:
#     async with AsyncTerminal("python:3.12") as term:
#         delay = randint(1, 4)
#         cmd = f"python - <<'PY'\nimport time, os, platform;" \
#               f"time.sleep({delay}); print('shell', {idx}, 'done on', platform.node())\nPY\n"
#         output = await term(cmd, timeout=delay + 1)
#         return f"[shell {idx}] >>> {output.strip()}"

# async def main():
#     MAX_NUM = 50
#     bar = tqdm(total=MAX_NUM * (MAX_NUM + 1) / 2)
#     wall_times = []
#     for n in range(1, MAX_NUM):
#         start = time.perf_counter()
#         tasks = []
#         for i in range(n):
#             tasks.append(run_in_shell(i))
#             bar.update(1)
#         for line in await asyncio.gather(*tasks):
#             print(line)
#         wall_time = time.perf_counter() - start
#         print(f"total wall-time: {wall_time:.2f}s")
#         wall_times.append(wall_time)
#         plt.plot(list(range(1, len(wall_times)+1)), wall_times)
#         plt.xlabel("N")
#         plt.ylabel("Wall-time (s)")
#         plt.title("Wall-time vs number of containers")
#         plt.savefig("walltime-shared-2.png")


def interact():
    with Terminal() as terminal:
        while True:
            user_input = input("input: ")
            print(terminal(user_input + "\n"))


if __name__ == "__main__":
    # asyncio.run(main())
    interact()
