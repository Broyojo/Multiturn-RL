import os

import docker
from datasets import Dataset
from swebench import build_instance_images
from swebench.harness.constants import LATEST
from swebench.harness.utils import load_swebench_dataset

client = docker.from_env(max_pool_size=1024)


def build(dataset, split):
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
                    {"role": "system", "content": ""},
                    {"role": "user", "content": d["problem_statement"]},
                ],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": split,
                    "index": i,
                    "docker_image": image_map[d["instance_id"]].instance_image_key,
                },
            }
        )

    return Dataset.from_list(verl_dataset)


def main():
    OUT = "./data/swebench/"
    os.makedirs(OUT, exist_ok=True)

    # train_dataset = build(
    #     load_swebench_dataset("princeton-nlp/SWE-bench", "train"), split="train"
    # )
    test_dataset = build(
        load_swebench_dataset("princeton-nlp/SWE-bench", "test"), split="test"
    )

    # train_dataset.to_parquet(os.path.join(OUT, "train.parquet"))
    test_dataset.to_parquet(os.path.join(OUT, "test.parquet"))


if __name__ == "__main__":
    main()
