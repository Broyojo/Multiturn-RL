#!/bin/bash

python -m swesmith.train.traj_mgr.clean_trajs trajectories/

python -m swesmith.train.traj_mgr.combine_trajs

python -m swesmith.train.traj_mgr.transform_to_ft