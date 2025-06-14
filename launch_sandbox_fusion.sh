set -ex

command -v docker >/dev/null 2>&1 || { echo >&2 "Docker is not installed. Please install Docker first."; exit 1; }

# Start sandbox fusion server
docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609
echo "Sandbox fusion server started on port 8080."