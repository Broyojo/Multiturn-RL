import time

import wandb
from datasets import load_dataset
from joblib import Parallel, delayed
from terminal import Terminal
from tqdm import tqdm


def make_trajectory(messages, extra_info):
    terminal = Terminal(image=extra_info["docker_image"], commit=extra_info["base_commit"])
    return {"messages": list(messages), "terminal": terminal, "extra_info": extra_info}


def create_trajectories(batch, n_jobs, n=5):
    start = time.time()
    try:
        trajs = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(make_trajectory)(messages, extra_info)
            for messages, extra_info in tqdm(
                [
                    (messages, extra_info)
                    for messages, extra_info in zip(
                        batch["prompt"],
                        batch["extra_info"],
                        strict=False,
                    )
                    for _ in range(n)
                ],
                desc="Starting containers...",
            )
        )
    except Exception:
        return [], float("inf")
    return trajs, time.time() - start


def run_experiment():
    run = wandb.init()

    n_jobs = wandb.config.n_jobs

    ds = (
        load_dataset("parquet", data_files=["./data/swebench/train.parquet"], split="train")
        .shuffle()
        .select(range(100))
    )

    trajectories, execution_time = create_trajectories(ds, n_jobs)

    Parallel(n_jobs=-1, backend="threading")(
        delayed(lambda t: t["terminal"].stop())(traj) for traj in tqdm(trajectories, desc="Extracting patches...")
    )

    wandb.log(
        {
            "execution_time": execution_time,
            "n_jobs": n_jobs,
            "throughput": (100 * 5) / execution_time,
        }
    )

    return execution_time


sweep_config = {
    "method": "random",
    "metric": {"name": "execution_time", "goal": "minimize"},
    "parameters": {"n_jobs": {"values": [16, 32, 64, 96, 128, 160, 192, 224, 240]}},
}

sweep_id = wandb.sweep(sweep_config, project="trajectory_optimization")

wandb.agent(sweep_id, function=run_experiment, count=9)  # Test each value once


api = wandb.Api()
sweep = api.sweep(f"broyojo/trajectory_optimization/{sweep_id}")
best_run = sweep.best_run()
best_n_jobs = best_run.config["n_jobs"]

print(f"Best n_jobs value: {best_n_jobs}")
print(f"Best execution time: {best_run.summary['execution_time']} seconds")

ds = load_dataset("parquet", data_files=["./data/swebench/train.parquet"], split="train").shuffle().select(range(63))
final_time = create_trajectories(ds, best_n_jobs)
print(f"Optimized execution time: {final_time} seconds")
