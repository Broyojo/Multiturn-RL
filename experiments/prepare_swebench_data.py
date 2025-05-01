import os

import docker
from datasets import Dataset, load_dataset
from swebench import build_instance_images, run_threadpool
from swebench.harness.constants import LATEST
from swebench.harness.utils import load_swebench_dataset
from swesmith.build_repo.download_images import (
    SWEFT_ORG,
    TAG,
    get_docker_hub_login,
    get_docker_repositories,
    get_dockerhub_token,
)

client = docker.from_env(max_pool_size=1024)

SYSTEM_PROMPT = """
You are a software engineering AI agent whose task is to resolve issues in Github repositories. You will be given an issue as well as access to a TTY into a Docker container with the repository installed. You can access the terminal to this docker container like so:

<terminal>echo hello world
</terminal>
<output>
hello world
~ $ </output>

Your job is to solve the issue given by the user by modifying the code in the repository.

Make sure to alternate between thinking and doing terminal actions. Your final response to the user should be in <answer></answer> tags. Here is an example template:s

<think>[your thinking here]</think>
<terminal>[your terminal input here]</terminal>
<think>[your thinking here]</think>
<terminal>[your terminal input here]</terminal>
...
<think>[your thinking here]</think>
<answer>[your final answer here]</answer>
""".strip()


def download_swesmith_images():
    username, password = get_docker_hub_login()
    token = get_dockerhub_token(username, password)

    # Get list of swesmith repositories
    repos = get_docker_repositories(SWEFT_ORG, token)
    repos = [r for r in repos if r["name"].startswith("swesmith")]

    for idx, r in enumerate(repos):
        print("-", r["name"])
        if idx == 4:
            print(f"(+ {len(repos) - 5} more...)")
            break

    def download(r):
        client.images.pull(f"{SWEFT_ORG}/{r['name']}:{TAG}")
        # Rename images via tagging
        new_name = f"{r['name'].replace('_1776_', '__')}:{TAG}"
        client.images.get(f"{SWEFT_ORG}/{r['name']}:{TAG}").tag(new_name)

    success, failure = run_threadpool(
        download, [(r,) for r in repos], max_workers=os.cpu_count()
    )

    return success, failure


def build_swesmith_train(ds):
    success, _ = download_swesmith_images()
    verl_ds = []
    for i, d in enumerate(ds):
        verl_ds.append(
            {
                "data_source": "swesmith",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": d["problem_statement"]},
                ],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": "train",
                    "index": i,
                    "docker_image": d["image_name"],
                    "base_commit": d["base_commit"],
                },
            }
        )
    return Dataset.from_list(verl_ds)


def build_swebench_test(dataset, split):
    successful, _ = build_instance_images(
        client=client, dataset=dataset, max_workers=8, tag=LATEST
    )

    image_map = {t[0].instance_id: t[0] for t in successful}

    verl_dataset = []
    for i, d in enumerate(dataset):
        if d["instance_id"] not in image_map:
            print(f"instance id {d['instance_id']} not found!!")
            continue

        verl_dataset.append(
            {
                "data_source": "swebench",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": d["problem_statement"]},
                ],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": split,
                    "index": i,
                    "base_commit": d["base_commit"],
                    "docker_image": image_map[d["instance_id"]].instance_image_key,
                },
            }
        )

    return Dataset.from_list(verl_dataset)


def main():
    OUT = "./data/swebench/"
    os.makedirs(OUT, exist_ok=True)

    train_dataset = load_dataset("SWE-bench/SWE-smith", split="train")
    train_dataset = train_dataset.filter(
        lambda e: len(e["problem_statement"]) > 0, num_proc=16
    )
    train_dataset = build_swesmith_train(train_dataset)

    test_dataset = build_swebench_test(
        load_swebench_dataset("princeton-nlp/SWE-bench", "test"), split="test"
    )

    train_dataset.to_parquet(os.path.join(OUT, "train.parquet"))
    test_dataset.to_parquet(os.path.join(OUT, "test.parquet"))


"""
some notes:

swesmith/swebench expects the following format:

jsonl file, with each line like this:
{
    KEY_INSTANCE_ID: <instance id>
    KEY_MODEL: <model>
    KEY_PREDICTION: <patch>
}

this should be saved in a jsonl file after the trajectory if finished

then, the reward function either runs the swebench evaluation script or the swesmith evaluation script depending on if it is training or testing
"""

if __name__ == "__main__":
    main()
