import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# accuracy precision recall f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
import re

def build_model_id(model_id):
    # number and b combination detection (i.e. 4b, 32b etc)
    model_id = model_id.lower()
    size = re.search(r'(\d+)(b)', model_id)[0]
    str_id = []

    if "medgemma" in model_id:
        str_id.append("MedGemma")
    elif "gemma" in model_id:
        str_id.append("Gemma")
    elif "qwen2.5" in model_id:
        str_id.append("Qwen2.5")
    str_id.append(size)


    if "real" in model_id:
        str_id.append("(Finetuned)")
    return " ".join(str_id)

def parse_output_data(text):
    # should be in <label_id>INTEGER,INTEGER</label_id> format with multiple labels possible
    text = text.strip()
    match = re.search(r'<label_id>([\d, ]+)</label_id>', text)
    if match:
        inner = match.group(1)
        label_list = list(int(i.strip()) for i in inner.split(",") if i.strip() != "")
        if len(set(label_list)) != len(label_list):
            return None
        if len(label_list) == 0:
            return None
        return set(label_list)
    else:
        return None
def parse_output(text):
    text = text.strip()

    match = re.search(r'\[(\d+),\s*(\d+)\]', text)
    if not match:
        match = re.search(r'(\d+),\s*(\d+)', text)
    
    if match:
        return int(match.group(1)), int(match.group(2))

    return None
def compute_hit_at_2(row):
    pred = row["pred"]    # list of ints
    target = row["target"]  # list of ints

    # Treat empty lists as "missing" / non-evaluable
    if not pred or not target:
        return np.nan

    # Use only top-2 predicted labels

    pred_top2 = set(pred)
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
df = pd.read_csv("../../data/real.csv")
df['file_path'] = df['SOPInstanceUID']
df = df[df['mode'] == 'test']
df.drop(columns=['SOPInstanceUID'], inplace=True)
# count each landmark occurence landmark is in set format
df['target'] = df['Target'].apply(parse_output_data)


# ---------- evaluation ----------
folder_path = "../../results/real_localization/"
model_ids = os.listdir(folder_path)
for i in model_ids:
    if "test" in i:
        model_ids.remove(i)
model_ids.sort()

for model_id in model_ids:

    str_id = build_model_id(model_id)
    result_df = pd.read_csv(os.path.join(folder_path, model_id))
    result_df = result_df.merge(
        df[['file_path', 'target']],
        left_on='filename',
        right_on='file_path',
        how='left',
        suffixes=('', '_df')
    )


    # overwrite result['target'] only when match exists
    result_df = result_df.drop(columns=['file_path'])    

    result_df["pred"] = result_df["output"].apply(parse_output)

    result_df["hit_at_2"] = result_df.apply(compute_hit_at_2, axis=1)
    hit_at_2 = result_df["hit_at_2"].mean(skipna=True)

    n_total = len(result_df)
    n_valid = result_df["hit_at_2"].notna().sum()
    hit_at_2 = result_df["hit_at_2"].mean(skipna=True)
    print(str_id)
    print(f"Hit@2: {hit_at_2:.2f}")
    result_df["recall_at_2"] = result_df.apply(compute_recall_at_2, axis=1)
    recall_at_2 = result_df["recall_at_2"].mean(skipna=True)
    print(f"Recall@2: {recall_at_2:.2f}")
    result_df["precision_at_2"] = result_df.apply(compute_precision_at_2, axis=1)
    precision_at_2 = result_df["precision_at_2"].mean(skipna=True)
    print(f"Precision@2: {precision_at_2:.2f}")
    result_df["f1_at_2"] = result_df.apply(compute_f1_at_2, axis=1)
    f1_at_2 = result_df["f1_at_2"].mean(skipna=True)
    print(f"F1@2: {f1_at_2:.2f}")

    print("Valid rows:", n_valid, "/", n_total)
    print()