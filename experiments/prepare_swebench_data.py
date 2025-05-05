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

# TODO: improve this system prompt and make it simpler and prime the model less on what to type or do
SYSTEM_PROMPT = """
You are an AI software engineering assistant. Your task is to resolve the given Github issue and you will be given access to the repository through the command line. You will need to modify the repo, remembering to track any new files with git, in order to fix the issue.

You put your internal thoughts in between <think></think>, your terminal input in between <terminal></terminal>, and your final answer in between <answer></answer>.  Format your response by alternating between thinking and terminal action, with the last message ending in an answer section after thinking.

Tip: the terminal input is a generic stdin input, so to run a command, you need to emit a newline character at the end. Additionally, you are able to type arbitrary control sequences, such as ^C, ^D, ^[[A, etc.
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

    success, failure = run_threadpool(download, [(r,) for r in repos], max_workers=os.cpu_count())

    return success, failure


def build_swesmith_train(ds):
    success, _ = download_swesmith_images()  # TODO: filter for successful images downloaded
    verl_ds = []
    swesmith_ds = []
    for i, d in enumerate(ds):
        swesmith_ds.append(d)
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
                    "base_commit": d["base_commit"],
                    "instance_id": d["instance_id"],
                    "docker_image": d["image_name"],
                },
            }
        )
    Dataset.from_list(swesmith_ds).to_json("./data/swebench/swesmith.jsonl", batch_size=len(swesmith_ds))
    return Dataset.from_list(verl_ds)


def build_swebench_test(dataset, split):
    successful, _ = build_instance_images(client=client, dataset=dataset, max_workers=os.cpu_count(), tag=LATEST)

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
                    "instance_id": d["instance_id"],
                    "docker_image": image_map[d["instance_id"]].instance_image_key,
                },
            }
        )
    Dataset.from_list(dataset).to_json("./data/swebench/swebench.json", batch_size=len(dataset), lines=False)

    return Dataset.from_list(verl_dataset)


def main():
    OUT = "./data/swebench/"
    os.makedirs(OUT, exist_ok=True)

    train_dataset = load_dataset("SWE-bench/SWE-smith", split="train")
    train_dataset = train_dataset.filter(lambda e: len(e["problem_statement"]) > 0, num_proc=16)
    train_dataset = build_swesmith_train(train_dataset)

    test_dataset = build_swebench_test(load_swebench_dataset("princeton-nlp/SWE-bench_Verified", "test"), split="test")

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
