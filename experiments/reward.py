"""
todo:
- run with BatchRewardManager so we can take in the whole batch in compute_score
- allow extra_info to always be passed, not just in validation stage. this way we can know what docker container to use for the data sample
- for all the swebench problems in the batch, we run the swebench evaluation script on them and extract the reward
"""

import json
import os
import re

import pytest
import swesmith.harness.eval
from joblib import Parallel, delayed


def format_reward(solution_str: str):
    """
    format:
    <|im_start|>assistant
    <think>here are some thoughts</think><terminal>this is a terminal command</terminal><|im_end|>
    <|im_start|>user
    <output>...</output><|im_end|>
    <|im_start|>assistant
    <think>ok seems like pretty interesting</think><answer>the answer is this \\boxed{thing}</answer><|im_end|>

    steps:
    1. check format
    2. run swebench evaluator on patch files
    3. get scores
    4. return scores
    """

    solution_str = "<|im_start|>assistant\n" + solution_str

    # extract the inner content of each assistant block
    blocks = re.findall(r"<\|im_start\|>assistant(.*?)<\|im_end\|>", solution_str, re.S)
    if not blocks:
        return 0

    for i, block in enumerate(blocks):
        # must have a <think>...</think>
        if not re.search(r"<think>.*?</think>", block, re.S):
            return 0

        # for all but the last block: exactly <terminal>
        if i < len(blocks) - 1:
            if not re.search(r"<terminal>.*?</terminal>", block, re.S) or re.search(
                r"<answer>.*?</answer>", block, re.S
            ):
                return 0
        # for the last block: exactly <answer>
        else:
            if not re.search(r"<answer>.*?</answer>", block, re.S) or re.search(
                r"<terminal>.*?</terminal>", block, re.S
            ):
                return 0

    return 1


def normalized_score(report):
    solved = len(report["tests_status"]["FAIL_TO_PASS"]["success"])
    unsolved = len(report["tests_status"]["FAIL_TO_PASS"]["failure"])
    regressed = len(report["tests_status"]["PASS_TO_PASS"]["failure"])
    maintained = len(report["tests_status"]["PASS_TO_PASS"]["success"])

    fix_ratio = solved / (solved + unsolved) if solved + unsolved > 0 else 0
    regression_ratio = (
        regressed / (regressed + maintained) if regressed + maintained > 0 else 0
    )

    score = fix_ratio - regression_ratio
    return score


def swesmith_reward(step, index, max_workers=4):
    # print(f"swesmith_reward({step}, {index})")
    run_id = f"step{step}-n{index}"
    predictions = f"./predictions/predictions_{index}.jsonl"
    swesmith.harness.eval.main(
        dataset_path="./data/swebench/swesmith.jsonl",
        predictions_path=predictions,
        run_id=run_id,
        max_workers=max_workers,
    )
    swe_scores = []
    with open(predictions, "r") as f:
        instances = [json.loads(line)["instance_id"] for line in f.readlines()]
    for instance in instances:
        if "report.json" not in os.listdir(
            f"./logs/run_evaluation/{run_id}/{instance}"
        ):
            swe_scores.append(0)
            continue

        with open(f"./logs/run_evaluation/{run_id}/{instance}/report.json", "r") as f:
            report = json.load(f)

        if "tests_status" not in report:
            swe_scores.append(0)
            continue

        score = normalized_score(report)
        swe_scores.append(score)
    return swe_scores


step = 0


# TODO: add swebench evaluation path (should be very similar to swesmith one)
def compute_score(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict],
    **reward_kwargs,
):
    global step

    n = len(os.listdir("./predictions"))
    # TODO: maybe parallelize this more efficiently? this is double layer of threading with GIL...
    swe_scores = Parallel(n_jobs=-1, backend="threading", timeout=300)(
        delayed(swesmith_reward)(step, index=i) for i in range(n)
    )

    # print(swe_scores)
    # print(len(swe_scores))
    # print(len(swe_scores[0]))

    # swe_scores: [[a,b,c,d], [a,b,c,d], [a,b,c,d]] (n x B)
    # swe_scores_flattened: [a,a,a,b,b,b,c,c,c,d,d,d] (nB)

    # 0,0 1,0 2,0
    # 0,1 1,1 2,1
    # 0,2 1,2 2,2
    # 0,3 1,3 2,3

    # print("before flatten:", swe_scores)

    swe_scores_flattened = []
    for i in range(len(swe_scores[0])):
        for j in range(n):
            swe_scores_flattened.append(swe_scores[j][i])

    format_rewards = [format_reward(solution) for solution in solution_strs]

    rewards = [s + f for s, f in zip(swe_scores_flattened, format_rewards)]

    print(f"swe scores: {swe_scores_flattened}")
    print(f"format rewards: {format_rewards}")
    print(f"rewards: {rewards}")

    step += 1

    return rewards


@pytest.mark.parametrize(
    "transcript,expected",
    [
        # valid cases
        ("<|im_start|>assistant\n<think>x</think><answer>y</answer><|im_end|>", 1),
        (
            "<|im_start|>assistant\n<think>t1</think><terminal>cmd1</terminal><|im_end|>\n"
            "<|im_start|>assistant\n<think>t2</think><answer>res</answer><|im_end|>",
            1,
        ),
        (
            "<|im_start|>assistant   \n\n<think>  foo  </think>\n\n<terminal>bar</terminal>\n<|im_end|>\n"
            "<|im_start|>assistant\n<think>baz</think><answer>42</answer><|im_end|>",
            1,
        ),
        # invalid cases
        (
            "<|im_start|>assistant\n<terminal>cmd</terminal><|im_end|>\n"
            "<|im_start|>assistant\n<think>ok</think><answer>res</answer><|im_end|>",
            0,
        ),
        (
            "<|im_start|>assistant\n<think>t1</think><answer>oops</answer><|im_end|>\n"
            "<|im_start|>assistant\n<think>t2</think><answer>res</answer><|im_end|>",
            0,
        ),
        (
            "<|im_start|>assistant\n<think>t1</think><terminal>cmd</terminal><|im_end|>\n"
            "<|im_start|>assistant\n<think>t2</think><terminal>cmd2</terminal><|im_end|>",
            0,
        ),
        (
            "<|im_start|>assistant\n<think>t1</think><|im_end|>\n"
            "<|im_start|>assistant\n<think>t2</think><answer>res</answer><|im_end|>",
            0,
        ),
        ("", 0),
        ("<|im_start|>user\n<output>hi</output><|im_end|>", 0),
        (
            "<|im_start|>assistant\n<think>t1</think><think>t2</think><|im_end|>\n"
            "<|im_start|>assistant\n<think>t2</think><answer>res</answer><|im_end|>",
            0,
        ),
    ],
    ids=[
        "single_valid_answer",
        "two_valid_blocks",
        "whitespace_and_newlines",
        "missing_think",
        "answer_in_non_last",
        "terminal_in_last_only",
        "non_last_missing_terminal",
        "empty_transcript",
        "other_roles_only",
        "double_think",
    ],
)
def test_format_reward(transcript, expected):
    assert format_reward(transcript) == expected


if __name__ == "__main__":
    # print(compute_score("openai/gsm8k", r"\boxed{1}</answer>", "1.2"))
    # print(compute_score("openai/gsm8k", r"\boxed{1}", "5.0-4.0"))
    # print(compute_score("openai/gsm8k", r"\boxed{${1,2,3,4}$}", "${1,3} \\cup {2,4}$"))
    pass
