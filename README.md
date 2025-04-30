# Multiturn-RL
Multiturn RL experiments

## Setup
```bash
$ uv sync
```

## Run
```bash
$ cd experiments
$ ./train.sh
```

## TODO
- [X] Make custom chat scheduler for terminal environment
- [ ] Create custom reward functions for various tasks (SWE-Bench, TACO, math, etc.)
- [ ] Collect data
  - [ ] ToRL dataset?
- [ ] Run experiments



## Data
*1. Math
   - NuminaMath
   - AIME
   - MATH
   - etc.
2. Coding
   - TACO
*3. **SWE**
   - **SWE-Bench**



Notes:
- should we do regular RL training before integrating with terminal? seems like it will take a while before the model starts using the terminal
