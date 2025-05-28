# Multiturn-RL
Multiturn RL experiments

## Setup
```bash
# install uv if necessary
$ curl -LsSf https://astral.sh/uv/install.sh | sh
# install dependencies
$ uv venv
$ uv pip install "torch==2.6.0"
$ uv sync --no-build-isolation
$ uv pip install "vllm==0.8.5"
```

## Run
```bash
$ cd experiments
$ python prepare_swebench_data.py
$ ./train.sh
```