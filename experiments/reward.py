from verl.utils.reward_score import gsm8k


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    if data_source == "openai/gsm8k":
        return gsm8k.compute_score(solution_str, ground_truth)
    raise NotImplementedError(f"Reward function is not implemented for {data_source=}")