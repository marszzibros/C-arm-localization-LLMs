import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import numpy as np
def parse_output(text):
    text = text.strip()

    match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+)\]', text)
    if not match:
        match = re.search(r'(\d+),\s*(\d+)', text)
    
    if match:
        return int(match.group(1)), int(match.group(2))

    return None
def compute_hit_at_2(row):
    pred = row["pred"]
    target = row["target"]
    
    # Skip rows with missing target or missing pred
    if pd.isna(target) or pred is None or (isinstance(pred, float) and pd.isna(pred)):
        return np.nan  # or False, depending on how you want to handle missing
    
    # Ensure pred is a collection
    if isinstance(pred, (list, tuple, set)):
        pred_set = set(pred)
    else:
        pred_set = {pred}
    
    return target in pred_set

# top 1 accuracy
def compute_top_1_accuracy(row):
    pred = row["pred"]        # list of predictions
    target = row["target"]    # scalar label

    if pred is None or target is None or pred == []:
        return np.nan

    top1 = pred[0]            # first (top-1) prediction

    return 1 if top1 == target else 0

df = pd.read_csv("../../results/synthetic_localization/test_classification.csv")


df["target"] = df["target"].apply(lambda s: int(str(s).strip("[]").strip()))
df["pred"] = df["pred"].apply(parse_output)


df["hit_at_2"] = df.apply(compute_hit_at_2, axis=1)
hit_at_2 = df["hit_at_2"].mean(skipna=True)
print(f"Hit@2: {hit_at_2:.4f}")

df["top_1_accuracy"] = df.apply(compute_top_1_accuracy, axis=1)
top_1_accuracy = df["top_1_accuracy"].mean(skipna=True)
print(f"Top-1 Accuracy: {top_1_accuracy:.4f}")

