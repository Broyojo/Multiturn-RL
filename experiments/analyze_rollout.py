import json
import os

import matplotlib.pyplot as plt
import numpy as np


# Read the JSONL file and count occurrences of </terminal>
def count_terminal_occurrences(file_path):
    counts = []
    with open(file_path) as f:
        for line in f:
            item = json.loads(line)
            rollout = item.get("output", "")
            count = rollout.count("</terminal>")
            counts.append(count)
    return counts


# Walk through the rollouts directory and collect counts from all JSONL files
def aggregate_terminal_counts(directory):
    aggregated_counts = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                counts = count_terminal_occurrences(file_path)
                aggregated_counts.extend(counts)
    return aggregated_counts


# Print summary statistics
def print_summary_statistics(counts):
    print("Summary Statistics:")
    print(f"Count: {len(counts)}")
    print(f"Mean: {np.mean(counts):.2f}")
    print(f"Median: {np.median(counts):.2f}")
    print(f"Standard Deviation: {np.std(counts):.2f}")
    print(f"Min: {np.min(counts)}")
    print(f"Max: {np.max(counts)}")


# Generate the whisker plot
def generate_whisker_plot(counts, title="Whisker Plot of </terminal> Counts"):
    plt.figure(figsize=(8, 6))
    plt.boxplot(counts, vert=False)
    plt.title(title)
    plt.xlabel("Count of </terminal>")
    plt.savefig("tool_calls.png")


# Directory path to your rollouts
directory = "./rollouts/validation/"

# Aggregate occurrences, print summary, and generate plot
counts = aggregate_terminal_counts(directory)
print_summary_statistics(counts)
generate_whisker_plot(counts)
