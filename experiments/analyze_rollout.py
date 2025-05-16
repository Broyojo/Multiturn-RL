import json
import os

import matplotlib.pyplot as plt

DIR = "./rollouts/validation/run10"

mean_scores = []

for file in sorted(os.listdir(DIR), key=lambda s: int(s.split(".")[0])):
    print(file)
    path = os.path.join(DIR, file)
    with open(path) as f:
        data = [json.loads(line) for line in f.readlines()]
    count = 0
    for x in data:
        if x["score"] == 1.0:
            count += 1
    mean_scores.append(count / len(data))

plt.plot(mean_scores)
plt.savefig("mean_acc.png")
