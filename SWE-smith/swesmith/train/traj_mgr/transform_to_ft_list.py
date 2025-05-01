#! /usr/bin/env python3


"""
Similar to transform_to_ft.py, but takes a list of paths to trajectories to transform.
All filtering must have already been done on the list of paths.

Example usage:
python transform_to_ft_list.py --traj_list traj_list.json --out_path ft_list.jsonl

See transform_to_ft.py for more details on the format of the output.
"""

import argparse
import json

from pathlib import Path
from swesmith.train.traj_mgr.utils import transform_traj_xml
from tqdm.auto import tqdm


def main(traj_list_file: Path, out_path: Path) -> None:
    traj_paths: list[str] = json.loads(Path(traj_list_file).read_text())
    print(f"Transforming {len(traj_paths)} trajectories")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for traj_path in tqdm(traj_paths):
            traj = json.loads(Path(traj_path).read_text())
            traj_xml = transform_traj_xml(traj)
            f.write(json.dumps(traj_xml) + "\n")

    print(f"Wrote {len(traj_paths)} trajectories to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traj_list",
        type=Path,
        required=True,
        help="Path to file containing list of trajectories to transform",
    )
    parser.add_argument(
        "--out_path", type=Path, required=True, help="Path to output .jsonlfile"
    )
    args = parser.parse_args()
    main(traj_list_file=args.traj_list, out_path=args.out_path)
