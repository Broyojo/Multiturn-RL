"""
todo:
- run with BatchRewardManager so we can take in the whole batch in compute_score
- allow extra_info to always be passed, not just in validation stage. this way we can know what docker container to use for the data sample
- for all the swebench problems in the batch, we run the swebench evaluation script on them and extract the reward
"""

import re

import pytest


def compute_score(
    data_source: str, solution_str: str, ground_truth: str, extra_info=None
) -> float:
    print(solution_str)
    print("=" * 100)
    return 0
    # gold = parse(ground_truth)
    # answer = parse(
    #     solution_str,
    #     extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
    # )
    # return 1 if verify(gold, answer) else 0


def format_reward(solution_str: str):
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


# def compute_score(
#     data_sources: list[str],
#     solution_strs: list[str],
#     ground_truths: list[str],
#     extra_infos: list[dict],
#     **reward_kwargs,
# ):
#     # math: math500, aime25, aime24 ToRL
#     # coding: TACO
#     # swebench: swebench

#     """
#     format reward:

#     <|im_start|>assistant
#     <think>here are some thoughts</think><terminal>this is a terminal command</terminal><|im_end|>
#     <|im_start|>user
#     <output>...</output><|im_end|>
#     <|im_start|>assistant
#     <think>ok seems like pretty interesting</think><answer>the answer is this \\boxed{thing}</answer><|im_end|>
#     """


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
