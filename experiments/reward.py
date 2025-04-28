from verl.utils.reward_score import gsm8k

"""
todo:
- run with BatchRewardManager so we can take in the whole batch in compute_score
- allow extra_info to always be passed, not just in validation stage. this way we can know what docker container to use for the data sample
- for all the swebench problems in the batch, we run the swebench evaluation script on them and extract the reward
"""

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    if data_source == "openai/gsm8k":
        return gsm8k.compute_score(solution_str, ground_truth)
    raise NotImplementedError(f"Reward function is not implemented for {data_source=}")