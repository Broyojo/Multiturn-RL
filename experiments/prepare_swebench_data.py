import os

import docker
from datasets import Dataset, load_dataset
from swebench import build_instance_images, run_threadpool
from swebench.harness.constants import LATEST
from swebench.harness.utils import load_swebench_dataset
from swesmith.build_repo.download_images import (
    SWEFT_ORG,
    TAG,
    get_docker_hub_login,
    get_docker_repositories,
    get_dockerhub_token,
)

client = docker.from_env(max_pool_size=1024)

# TODO: improve this system prompt and make it simpler and prime the model less on what to type or do
SYSTEM_PROMPT = """
You are an AI assistant specialized in software engineering. Your mission is to solve issues in GitHub repositories.

HOW TO INTERACT WITH THE TERMINAL:
- You can run commands in the Docker container by using:
  <terminal>your command here
  </terminal>

- You'll see the output like this:
  <output>
  command results will appear here
  </output>

- Important: Always include a newline at the end of your commands to execute them properly
- You can use control characters (^C, ^D, etc.) when needed

IMPORTANT - GIT VERSION CONTROL:
- Any new files you create MUST be added to git using appropriate commands
- After creating new files, always run: git add <filename> or git add . 
- Consider making atomic commits with meaningful commit messages that describe your changes
- Before submitting your final answer, ensure all your changes are properly tracked in git

IMPORTANT - CONCURRENT OPERATION:
The terminal operates in real-time and doesn't wait for commands to fully complete before showing output. For example:
- If you run a long command like "sudo apt install package", you might only see the beginning of the installation process in the output
- You can continue thinking and planning your next steps while commands are still running
- You don't need to wait for a command to finish before moving to your next <think> section

YOUR WORKFLOW SHOULD BE:
1. Think about the issue and plan your approach
2. Execute terminal commands to explore, debug, and solve the problem
3. When creating new files, add them to git immediately after creation
4. Alternate between thinking and executing commands until the issue is resolved
5. Before concluding, verify all changes are tracked in git
6. Provide your final solution in <answer></answer> tags

EXAMPLE STRUCTURE:
<think>I'll first examine the repository structure to understand the codebase.</think>
<terminal>ls -la
</terminal>
<think>Now I see the files. Let me check the specific code causing the issue.</think>
<terminal>cat file_with_issue.py
</terminal>
<think>I understand the problem. I'll create a new utility file to help fix the issue.</think>
<terminal>touch utils.py
echo "def fix_problem():\n    return 'fixed'" > utils.py
</terminal>
<think>I've created a new file. I need to add it to git.</think>
<terminal>git add utils.py
git status
</terminal>
<think>I've implemented the fix. Now let me test it.</think>
<terminal>python test.py
</terminal>
<think>The fix works. Let me commit the changes to git.</think>
<terminal>git commit -m "Add utility function to fix the issue"
</terminal>
<answer>I've resolved the issue by creating a utility function in a new file (utils.py) that addresses [specific problem] in [file]. The solution involved [brief explanation of what was changed]. I've added the new file to git, committed the changes with an appropriate message, and tested the fix to confirm it works.</answer>
""".strip()


def download_swesmith_images():
    username, password = get_docker_hub_login()
    token = get_dockerhub_token(username, password)

    # Get list of swesmith repositories
    repos = get_docker_repositories(SWEFT_ORG, token)
    repos = [r for r in repos if r["name"].startswith("swesmith")]

    for idx, r in enumerate(repos):
        print("-", r["name"])
        if idx == 4:
            print(f"(+ {len(repos) - 5} more...)")
            break

    def download(r):
        client.images.pull(f"{SWEFT_ORG}/{r['name']}:{TAG}")
        # Rename images via tagging
        new_name = f"{r['name'].replace('_1776_', '__')}:{TAG}"
        client.images.get(f"{SWEFT_ORG}/{r['name']}:{TAG}").tag(new_name)

    success, failure = run_threadpool(
        download, [(r,) for r in repos], max_workers=os.cpu_count()
    )

    return success, failure


def build_swesmith_train(ds):
    success, _ = (
        download_swesmith_images()
    )  # TODO: filter for successful images downloaded
    verl_ds = []
    swesmith_ds = []
    for i, d in enumerate(ds):
        swesmith_ds.append(d)
        verl_ds.append(
            {
                "data_source": "swesmith",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": d["problem_statement"]},
                ],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": "train",
                    "index": i,
                    "base_commit": d["base_commit"],
                    "instance_id": d["instance_id"],
                    "docker_image": d["image_name"],
                },
            }
        )
    Dataset.from_list(swesmith_ds).to_json(
        "./data/swebench/swesmith.jsonl", batch_size=len(swesmith_ds)
    )
    return Dataset.from_list(verl_ds)


def build_swebench_test(dataset, split):
    successful, _ = build_instance_images(
        client=client, dataset=dataset, max_workers=8, tag=LATEST
    )

    image_map = {t[0].instance_id: t[0] for t in successful}

    verl_dataset = []
    for i, d in enumerate(dataset):
        if d["instance_id"] not in image_map:
            print(f"instance id {d['instance_id']} not found!!")
            continue

        verl_dataset.append(
            {
                "data_source": "swebench",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": d["problem_statement"]},
                ],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": split,
                    "index": i,
                    "base_commit": d["base_commit"],
                    "instance_id": d["instance_id"],
                    "docker_image": image_map[d["instance_id"]].instance_image_key,
                },
            }
        )

    return Dataset.from_list(verl_dataset)


def main():
    OUT = "./data/swebench/"
    os.makedirs(OUT, exist_ok=True)

    train_dataset = load_dataset("SWE-bench/SWE-smith", split="train")
    train_dataset = train_dataset.filter(
        lambda e: len(e["problem_statement"]) > 0, num_proc=16
    )
    train_dataset = build_swesmith_train(train_dataset)

    test_dataset = build_swebench_test(
        load_swebench_dataset("princeton-nlp/SWE-bench", "test"), split="test"
    )

    train_dataset.to_parquet(os.path.join(OUT, "train.parquet"))
    test_dataset.to_parquet(os.path.join(OUT, "test.parquet"))


"""
some notes:

swesmith/swebench expects the following format:

jsonl file, with each line like this:
{
    KEY_INSTANCE_ID: <instance id>
    KEY_MODEL: <model>
    KEY_PREDICTION: <patch>
}

this should be saved in a jsonl file after the trajectory if finished

then, the reward function either runs the swebench evaluation script or the swesmith evaluation script depending on if it is training or testing
"""

if __name__ == "__main__":
    main()
