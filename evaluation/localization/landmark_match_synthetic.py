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

    if "medgemma" not in model_id and "gemma" in model_id and "sft" in model_id:
        str_id.append("Gemma")
        str_id.append(size)
        str_id.append("(Finetuned)")
    elif "medgemma" in model_id:
        str_id.append("MedGemma")
        str_id.append(size)
    elif "gemma" in model_id:
        str_id.append("Gemma")
        str_id.append(size)
    elif "qwen2.5" in model_id and "sft" in model_id:
        str_id.append("Qwen2.5")
        str_id.append(size) 
        str_id.append("(Finetuned)")
    elif "qwen2.5" in model_id:
        str_id.append("Qwen2.5")
        str_id.append(size)
    return " ".join(str_id)
def parse_output(text):
    text = text.strip()

    match = re.search(r'\[(\d+),\s*(\d+)\]', text)
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

def compute_top_1_accuracy(row):
    pred = row["pred"]        # list of predictions
    target = row["target"]    # scalar label

    if pred is None or target is None or pred == []:
        return np.nan

    top1 = pred[0]            # first (top-1) prediction

    return 1 if top1 == target else 0

df = pd.read_csv("../../data/classification.csv", index_col=0)

folder_path = "../../results/synthetic_localization/"
model_ids = os.listdir(folder_path)
for i in model_ids:
    if "test" in i:
        model_ids.remove(i)
model_ids.sort()

for model_id in model_ids:

    str_id = build_model_id(model_id)
    print(str_id)
    result_df = pd.read_csv(os.path.join(folder_path, model_id))


    result_df['pred'] = result_df['output'].apply(parse_output)
    result_df['target'] = result_df['filename'].apply(lambda x: int(df[df['file_path'] == x]['landmark'].values[0]))

    # result_df['pred'] = result_df['pred'].fillna(-1)
    # result_df['target'] = result_df['target'].fillna(-1)

    # Hit@2: if either of the two landmarks is predicted correctly
    result_df["hit_at_2"] = result_df.apply(compute_hit_at_2, axis=1)
    hit_at_2 = result_df["hit_at_2"].mean(skipna=True)

    n_total = len(result_df)
    n_valid = result_df["hit_at_2"].notna().sum()

    hit_at_2 = result_df["hit_at_2"].mean(skipna=True)

    print(f"Hit@2: {hit_at_2:.2f}")
    

    result_df["top_1_accuracy"] = result_df.apply(compute_top_1_accuracy, axis=1)
    top_1_accuracy = result_df["top_1_accuracy"].mean(skipna=True)
    print(f"Top-1 Accuracy: {top_1_accuracy:.2f}")

    print("Valid rows:", n_valid, "/", n_total)
    print() 

    # Recall@2: proportion of true landmarks
    # Here, the true landmark is only one, so it's equivalent to Hit@2

