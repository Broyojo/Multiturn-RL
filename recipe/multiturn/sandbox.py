import asyncio
from uuid import uuid4

import ray
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from tqdm import tqdm
from util import run_async

import docker


class Sandbox:
    def __init__(
        self,
        image: str,
        command: str,
        user: str,
        working_dir: str,
        network: str,
        ports: dict | None = None,
        mcp_port: int = 3000,
    ):
        self.image = image
        self.command = command
        self.user = user
        self.working_dir = working_dir
        self.network = network
        self.ports = ports
        self.mcp_port = mcp_port
        self.container_id = None
        self.container_node_ip = None
        self.client = None
        self.container = None
        self.session = None
        self._mcp_client_manager = None

    async def __aenter__(self):
        try:
            future = launch_container.remote(
                image=self.image,
                command=self.command,
                name=f"sandbox-{uuid4().hex}",
                user=self.user,
                working_dir=self.working_dir,
                network=self.network,
                ports=self.ports,
            )

            self.container_id, self.container_node_ip = await run_async(lambda: ray.get(future))

            self.client = await run_async(lambda: docker.DockerClient(base_url=f"tcp://{self.container_node_ip}:2375"))
            self.container = await run_async(lambda: self.client.containers.get(self.container_id))
            await run_async(lambda: self.container.reload())

            mapped_port = self.container.ports.get(f"{self.mcp_port}/tcp")[0]["HostPort"]
            mcp_url = f"http://{self.container_node_ip}:{mapped_port}/mcp"

            self._mcp_client_manager = streamablehttp_client(mcp_url)
            read_stream, write_stream, _ = await self._mcp_client_manager.__aenter__()

            self.session = ClientSession(read_stream, write_stream)
            await self.session.__aenter__()

            # TODO: add proper wait with timeout for container and MCP server to be ready
            await asyncio.sleep(2)

            await self.session.initialize()

            return self
        except Exception:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            try:
                await self.session.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                print(f"Error closing MCP session: {e}")

        if self._mcp_client_manager:
            try:
                await self._mcp_client_manager.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                print(f"Error closing MCP client: {e}")

        if self.container:
            try:
                await run_async(lambda: self.container.stop())
                await run_async(lambda: self.container.remove(force=True))
            except Exception as e:
                print(f"Error cleaning up container: {e}")

        if self.client:
            try:
                await run_async(lambda: self.client.close())
            except Exception as e:
                print(f"Error closing Docker client: {e}")


@ray.remote
def launch_container(
    image: str, command: str, name: str, user: str, working_dir: str, network: str, ports: dict | None = None
):
    client = docker.from_env()
    container = client.containers.create(
        image,
        name=name,
        command=command,
        detach=True,
        platform="linux/x86_64",
        user=user,
        working_dir=working_dir,
        restart_policy={"Name": "unless-stopped"},
        ports=ports,
    )
    network = client.networks.get(network)
    network.connect(container)
    container.start()

    node_ip = ray.util.get_node_ip_address()
    container.reload()

    return container.id, node_ip


async def main():
    NUM_SANDBOXES = 2
    bar = tqdm(total=NUM_SANDBOXES, desc="Launching Sandboxes")

    async def start_sandbox():
        async with Sandbox(
            image="timemagic/rl-mcp:general",
            command="/mcp/daemon-mcp.py",
            user="root",
            working_dir="/",
            network="mcp-network",
            ports={"3000/tcp": None},
            mcp_port=3000,
        ) as sandbox:
            bar.write(
                f"Sandbox launched with ID: {sandbox.container_id}, Node IP: {sandbox.container_node_ip}, Ports: {sandbox.container.ports}"  # noqa: E501
            )
            bar.write(f"Container status: {sandbox.container.status}")
            print(await sandbox.session.list_tools())
            bar.update(1)
            await asyncio.sleep(20)

    await asyncio.gather(*[start_sandbox() for _ in range(NUM_SANDBOXES)])


if __name__ == "__main__":
    asyncio.run(main())
