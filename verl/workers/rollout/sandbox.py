import ray

import docker


@ray.remote
def _start_container(base_image: str, mcp_config: dict):
    client = docker.from_env()
    client.containers.create(
        image=base_image,
        command=
    )
    ...
    
    return node_ip, port, contianer_id


async def launch_container(base_image: str, mcp_config: dict):
    ip, port, id = ray.get(_start_container.remote(base_image, mcp_config))


class Sandbox:
    def __init__(self):
        pass
