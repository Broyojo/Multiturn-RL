from datasets import load_dataset

t = 0
ds = load_dataset("SWE-bench/SWE-smith")
filtered = ds.filter(lambda e: len(e["problem_statement"]) > 0, num_proc=16)
print(filtered)
print(filtered["train"][69]["problem_statement"])
