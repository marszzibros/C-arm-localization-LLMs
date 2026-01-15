import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import numpy as np
def parse_output(text):
    text = text.strip()
    nums = [int(n) for n in re.findall(r'\d+', text)]
    return nums
def compute_hit_at_2(row):
    pred = row["pred"]    # list of ints
    target = row["target"]  # list of ints

    # Treat empty lists as "missing" / non-evaluable
    if not pred or not target:
        return np.nan

    # Use only top-2 predicted labels
    pred_top2 = set(pred[:2])
    target_set = set(target)

    return 1 if len(pred_top2 & target_set) > 0 else 0
def compute_recall_at_2(row):
    pred = row["pred"]    # list of ints
    target = row["target"]  # list of ints

    if not pred or not target:
        return np.nan

    pred_top2 = set(pred[:2])
    target_set = set(target)

    true_positives = len(pred_top2 & target_set)
    num_actual_positives = len(target_set)

    return true_positives / num_actual_positives

def compute_precision_at_2(row):
    pred = row["pred"]
    target = row["target"]

    if not pred or not target:
        return np.nan

    pred_top2 = set(pred[:2])
    target_set = set(target)

    true_positives = len(pred_top2 & target_set)
    predicted_positives = len(pred_top2)

    if predicted_positives == 0:
        return np.nan

    return true_positives / predicted_positives
def compute_f1_at_2(row):
    precision = compute_precision_at_2(row)
    recall = compute_recall_at_2(row)

    if precision is np.nan or recall is np.nan:
        return np.nan

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)
df = pd.read_csv("../../results/real_localization/test_classification.csv", index_col=0)

df["target"] = df["target"].apply(parse_output)
df["pred"] = df["pred"].apply(parse_output)

df["hit_at_2"] = df.apply(compute_hit_at_2, axis=1)
hit_at_2 = df["hit_at_2"].mean(skipna=True)
print(f"Hit@2: {hit_at_2:.4f}")

df["recall_at_2"] = df.apply(compute_recall_at_2, axis=1)
recall_at_2 = df["recall_at_2"].mean(skipna=True)
print(f"Recall@2: {recall_at_2:.4f}")

df["precision_at_2"] = df.apply(compute_precision_at_2, axis=1)
precision_at_2 = df["precision_at_2"].mean(skipna=True)
print(f"Precision@2: {precision_at_2:.2f}")
df["f1_at_2"] = df.apply(compute_f1_at_2, axis=1)
f1_at_2 = df["f1_at_2"].mean(skipna=True)
print(f"F1@2: {f1_at_2:.2f}")
