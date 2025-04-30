import docker
from swebench.harness.docker_build import build_instance_image
from swebench.harness.docker_utils import exec_run_with_timeout

client = docker.client.from_env()

# Build a Docker image for a specific instance
instance_id = "httpie-cli/httpie#1088"
image_tag = build_instance_image(client=client, instance_id)

# Run commands in the container
result = exec_run_with_timeout(
    container_name=f"container_{instance_id}",
    image_tag=image_tag,
    cmd="cd /workspace && pytest",
    timeout=300,
)
