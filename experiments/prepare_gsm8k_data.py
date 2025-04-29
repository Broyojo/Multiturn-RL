# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import random
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/gsm8k")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "openai/gsm8k"

    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": "You have access to an integrated terminal. To use it, type your input inside <terminal></terminal> delimeters. You should use it to help you solve the math problem."
                    },
                    {
                        "role": "user",
                        "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
                    },
                    {
                        "role": "assistant", 
                        "content": "To solve this problem, I need to calculate how many eggs Janet sells after consuming some herself.\n\n<terminal>echo $((16 - 3 - 4))</terminal>"
                    },
                    {
                        "role": "user",
                        "content": "<output>9\n~$ </output>"
                    },
                    {
                        "role": "assistant",
                        "content": "Janet has 9 eggs left to sell each day. Now I'll calculate her daily earnings at $2 per egg.\n\n<terminal>echo $((9 * 2))</terminal>",
                    },
                    {
                        "role": "user",
                        "content": "<output>18\n~$ </output>"
                    },
                    {
                        "role": "assistant",
                        "content": "Janet makes $18 per day at the farmers' market from selling her duck eggs. #### 18"
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "docker_image": random.choice(["python:3.12", "ubuntu:latest"])
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
