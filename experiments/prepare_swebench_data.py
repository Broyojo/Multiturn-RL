import os

import docker
import ray
from datasets import Dataset, load_dataset
from reward import swesmith_eval
from swebench import build_instance_images, run_threadpool
from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION, LATEST
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
You are an AI software engineering assistant. Your task is to resolve the given Github issue and you will be given access to the repository through the command line. The repository is already cloned at the current working directory and available through your terminal, there is no need to clone it manually. You will need to modify the repo and make and commit your changes to git to fix the issue.

The terminal input is a generic stdin input, so to run a command, you need to emit a newline character at the end. Additionally, you are able to type arbitrary control sequences, such as ^C, ^D, ^[[A, etc. Example: <terminal>ls -la\n</terminal> or <terminal>^C</terminal>.
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
                    "base_commit": d["base_commit"],
                    "instance_id": d["instance_id"],
                    "docker_image": d["image_name"],
                    "data_row": d,
                    "source": "swesmith",
                },
            }
        )
    verl_ds = filter_valid_swesmith(verl_ds)
    swesmith_ds = []
    for row in verl_ds:
        swesmith_ds.append(row["extra_info"]["data_row"])
    Dataset.from_list(swesmith_ds).to_json("./data/swebench/swesmith.jsonl", batch_size=len(swesmith_ds))
    return Dataset.from_list(verl_ds)


def filter_valid_swesmith(ds):
    """check that the swesmith task gives score of 0 when no patch is given"""
    futures = [
        swesmith_eval.remote(
            patch={
                KEY_INSTANCE_ID: row["extra_info"]["instance_id"],
                KEY_MODEL: "testing",
                KEY_PREDICTION: "",
            },
            data_row=row["extra_info"]["data_row"],
            step=0,
        )
        for row in ds
    ]
    results = ray.get(futures)
    valid_ds = []
    for result, data in zip(results, ds, strict=True):
        if result == 0:
            valid_ds.append(data)
    return valid_ds


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
                    "data_row": d,
                    "source": "swebench",
                },
            }
        )
    Dataset.from_list(dataset).to_json("./data/swebench/swebench.json", batch_size=len(dataset), lines=False)

    return Dataset.from_list(verl_dataset)


def main():
    OUT = "./data/swebench/"
    os.makedirs(OUT, exist_ok=True)

    scratch = False
    if scratch:
        train_dataset = load_dataset("SWE-bench/SWE-smith", split="train")
        train_dataset = train_dataset.filter(lambda e: len(e["problem_statement"]) > 0, num_proc=16)
        train_dataset = build_swesmith_train(train_dataset)

        test_dataset = build_swebench_test(
            load_swebench_dataset("princeton-nlp/SWE-bench_Verified", "test"), split="test"
        )

        train_dataset.to_parquet(os.path.join(OUT, "train.parquet"))
        test_dataset.to_parquet(os.path.join(OUT, "test.parquet"))
    else:
        train_dataset = load_dataset("parquet", data_files=os.path.join(OUT, "train.parquet"), split="train")
        test_dataset = load_dataset("parquet", data_files=os.path.join(OUT, "test.parquet"), split="train")

        def change_sys_prompt(e):
            e["prompt"][0]["content"] = SYSTEM_PROMPT
            return e

        train_dataset = train_dataset.map(change_sys_prompt, num_proc=64)
        test_dataset = test_dataset.map(change_sys_prompt, num_proc=64)

        train_dataset.to_parquet(os.path.join(OUT, "train2.parquet"))
        test_dataset.to_parquet(os.path.join(OUT, "test2.parquet"))


if __name__ == "__main__":
    main()
