import json
import os
import re
import traceback
from pathlib import Path, PurePosixPath
from typing import cast
from uuid import uuid4

import docker
import ray
import swebench
import swebench.harness
import swesmith.harness.eval
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    LOG_INSTANCE,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    SWEbenchInstance,
)
from swebench.harness.docker_build import setup_logger
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec
from swesmith.constants import KEY_PATCH
from terminal import RemoteTerminal


def format_reward(messages: list[dict[str, str]]):
    assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
    if not assistant_messages:
        return 1.0

    for i, msg in enumerate(assistant_messages[:-1]):
        content = msg.get("content", "")
        pattern = r"^\s*<think>.*?</think>\s*<terminal>.*?</terminal>\s*$"
        if not re.match(pattern, content, re.DOTALL):
            return 0.0

    last_msg = assistant_messages[-1].get("content", "")
    pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    if not re.match(pattern, last_msg, re.DOTALL):
        return 0.0

    return 1.0


def normalized_score(report):
    solved = len(report["tests_status"]["FAIL_TO_PASS"]["success"])
    unsolved = len(report["tests_status"]["FAIL_TO_PASS"]["failure"])
    regressed = len(report["tests_status"]["PASS_TO_PASS"]["failure"])
    maintained = len(report["tests_status"]["PASS_TO_PASS"]["success"])

    fix_ratio = solved / (solved + unsolved) if solved + unsolved > 0 else 0
    regression_ratio = regressed / (regressed + maintained) if regressed + maintained > 0 else 0

    score = fix_ratio - regression_ratio
    return score


@ray.remote
def swesmith_eval(patch, data_row, step):
    assert patch[KEY_INSTANCE_ID] == data_row[KEY_INSTANCE_ID]

    try:
        run_id = f"train-step{step}-{patch[KEY_INSTANCE_ID]}-{os.environ['EXPERIMENT'].replace('/', '__')}-{uuid4()}"
        instance_id = patch[KEY_INSTANCE_ID]

        swesmith.harness.eval.run_evaluation(pred=patch, instance=data_row, run_id=run_id)

        if "report.json" not in os.listdir(f"./logs/run_evaluation/{run_id}/{instance_id}"):
            return 0

        with open(f"./logs/run_evaluation/{run_id}/{instance_id}/report.json") as f:
            report = json.load(f)

        if "tests_status" not in report:
            return 0

        return normalized_score(report)
    except Exception:
        print(traceback.format_exc())
        return 0.0


@ray.remote
def swebench_eval(patch, data_row, step):
    assert patch[KEY_INSTANCE_ID] == data_row[KEY_INSTANCE_ID]

    try:
        run_id = f"eval-step{step}-{patch[KEY_INSTANCE_ID]}-{os.environ['EXPERIMENT'].replace('/', '__')}-{uuid4()}"
        instance_id = patch[KEY_INSTANCE_ID]

        client = docker.from_env(timeout=300)
        test_spec = make_test_spec(cast(SWEbenchInstance, data_row), namespace=None, instance_image_tag="latest")
        result = swebench.harness.run_evaluation.run_instance(
            test_spec=test_spec,
            pred=patch,
            rm_image=False,
            force_rebuild=False,
            client=client,
            run_id=run_id,
            timeout=300,
            rewrite_reports=False,
        )
        if result is None:
            return 0

        report = result[1][instance_id]
        if "tests_status" not in report:
            return 0

        return normalized_score(report)
    except Exception:
        print(traceback.format_exc())
        return 0.0


async def swebench_remote_eval(patch: dict, terminal: RemoteTerminal, data_row: dict, step: int):
    assert patch[KEY_INSTANCE_ID] == data_row[KEY_INSTANCE_ID]

    run_id = f"eval-step{step}-{patch[KEY_INSTANCE_ID]}-{os.environ['EXPERIMENT'].replace('/', '__')}-{uuid4()}"
    instance_id = patch[KEY_INSTANCE_ID]

    test_spec = make_test_spec(cast(SWEbenchInstance, data_row), namespace=None, instance_image_tag="latest")

    instance_id = test_spec.instance_id
    model_name_or_path = patch.get(KEY_MODEL, "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id

    report_path = log_dir / LOG_REPORT
    if report_path.exists():
        return instance_id, json.loads(report_path.read_text())

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_INSTANCE
    logger = setup_logger(instance_id, log_file)

    base_commit = data_row["base_commit"]
    git_diff = (await terminal.exec_run(command=f"bash -c 'git diff {base_commit}'")).output.decode(
        "utf-8", errors="replace"
    )
    patch[KEY_PATCH] = git_diff

    eval_file = Path(log_dir / "eval.sh")
    eval_file.write_text(test_spec.eval_script)
    await terminal.copy_to_container(eval_file, PurePosixPath("/eval.sh"))
    test_output = (await terminal.exec_run("/bin/bash /eval.sh", timeout=100)).output
    test_output_path = log_dir / LOG_TEST_OUTPUT

    with open(test_output_path, "w") as f:
        f.write(test_output)
        # if timed_out:
        #     f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
        #     raise EvaluationError(
        #         instance_id,
        #         f"Test timed out after {timeout} seconds.",
        #         logger,
        #     )

    report = get_eval_report(
        test_spec=test_spec,
        prediction=patch,
        test_log_path=test_output_path,
        include_tests_status=True,
    )


def compute_score(
    data_sources: list[str],
    solution_strs: list[str],
    scores: list[float],
    extra_infos: list[dict],
    **reward_kwargs,
) -> list[float]:
    # just a dummy compute score to pass along the scores from rollout
    return scores


if __name__ == "__main__":
    print(
        format_reward(
            [
                {
                    "role": "assistant",
                    "content": "<think>hi</think><answer>my answer <terminal>is this</terminal>",
                }
            ]
        )
    )
