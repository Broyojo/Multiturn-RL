"""
Create a DPO style preference dataset (torchtune compatible) from all SWE-agent trajectories under a given user.

[
  {
    "chosen_conversations": [
      {
        "content": "What do I do when I have a hole in my trousers?",
        "role": "user"
      },
      { "content": "Fix the hole.", "role": "assistant" }
    ],
    "rejected_conversations": [
      {
        "content": "What do I do when I have a hole in my trousers?",
        "role": "user"
      },
      { "content": "Take them off.", "role": "assistant" }
    ]
  }
]

Usage: (from SWE-agent directory)
python -m swesmith.train.traj_mgr.create_dpo_data
"""

import json
import os

from swesmith.train.traj_mgr.utils import MAP_STYLE_TO_FUNC
from tqdm.auto import tqdm


if __name__ == "__main__":
    USER = "john-b-yang"
    TRAJS_EXP_PREFIX = "swesmith_gen_"
    PATH_TO_TRAJS = f"trajectories/{USER}/"
    PATH_TO_EVAL_DIR = f"/home/{USER}/swe-smith/logs/run_evaluation/"
    OUT_FILE = input("Enter output file name (e.g. swesmith_dpo_250414.json): ")
    style = "xml"

    transform_traj = MAP_STYLE_TO_FUNC[style]

    print("Aggregating trajectories...")
    inst_id_to_trajs = {}
    for run_id in os.listdir(PATH_TO_TRAJS):
        traj_dir = os.path.join(PATH_TO_TRAJS, run_id)
        eval_dir = os.path.join(PATH_TO_EVAL_DIR, run_id)
        if not os.path.exists(eval_dir):
            continue

        inst_ids = [
            x for x in os.listdir(traj_dir) if os.path.isdir(os.path.join(traj_dir, x))
        ]
        print(f"> Found {len(inst_ids)} trajectories in {traj_dir}")

        for inst_id in tqdm(inst_ids):
            if inst_id not in os.listdir(eval_dir):
                continue
            if "report.json" not in os.listdir(os.path.join(eval_dir, inst_id)):
                continue
            report = json.load(open(os.path.join(eval_dir, inst_id, "report.json")))
            is_resolved = (
                report.get("resolved", False)
                if inst_id not in report
                else report[inst_id].get("resolved", False)
            )

            traj_path = os.path.join(traj_dir, inst_id, f"{inst_id}.traj")
            traj = transform_traj(json.load(open(traj_path, "r")))

            if inst_id not in inst_id_to_trajs:
                inst_id_to_trajs[inst_id] = []
            inst_id_to_trajs[inst_id].append({"traj": traj, "resolved": is_resolved})
    print(f"Done! Found {len(inst_id_to_trajs)} instances with 1+ trajectories\n")

    # Create DPO dataset from instance IDs with at least one resolved and one unresolved trajectory
    print("Creating DPO dataset...")
    dpo_dataset = []
    unique_inst_ids = 0
    for inst_id, trajs in tqdm(inst_id_to_trajs.items()):
        resolved_trajs = [x for x in trajs if x["resolved"]]
        unresolved_trajs = [x for x in trajs if not x["resolved"]]
        if len(resolved_trajs) == 0 or len(unresolved_trajs) == 0:
            continue
        unique_inst_ids += 1
        for resolved, unresolved in zip(resolved_trajs, unresolved_trajs):
            dpo_dataset.append(
                {
                    "chosen_conversations": resolved["traj"]["messages"],
                    "rejected_conversations": unresolved["traj"]["messages"],
                }
            )

    # Write DPO dataset to file
    with open(OUT_FILE, "w") as f:
        f.write(json.dumps(dpo_dataset, indent=2))
    print(
        f"Wrote DPO dataset to {OUT_FILE} ({len(dpo_dataset)} data points from {unique_inst_ids} instances)"
    )
