import pandas as pd
import os
import numpy as np

result_folder = "../../results/retrieval"

gt_df = pd.read_csv("../../data/test.csv", index_col=0)

# regex; starts with [ and ends with ] and seperated by ", " and each item contains [number: landmark (might contain comman and space)]
# extract numbers; be cautious of unrelated numbers; e.g. T1 T12 etc...
def extract_landmark_numbers(landmark_str):
    import re
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, landmark_str)
    landmark_numbers = []
    for match in matches:
        items = match.split(', ')
        for item in items:
            number = item.split(':')[0].strip()
            if number.isdigit():
                landmark_numbers.append(int(number))
    return landmark_numbers

def precision_at_k(groundtruth, prediction, k):
    if len(prediction) == 0:
        return 0.0
    top_k_predictions = prediction[:k]
    relevant_items = set(groundtruth)
    retrieved_relevant_items = sum(1 for item in top_k_predictions if item in relevant_items)
    precision = retrieved_relevant_items / len(top_k_predictions)
    return precision

def recall_at_k(groundtruth, prediction, k):
    if len(prediction) == 0 or len(groundtruth) == 0:
        return 0.0
    top_k_predictions = prediction[:k]
    relevant_items = set(groundtruth)
    retrieved_relevant_items = sum(1 for item in top_k_predictions if item in relevant_items)
    recall = retrieved_relevant_items / len(relevant_items)
    return recall

def average_precision(groundtruth, prediction):

    relevant_items = set(groundtruth)
    score = 0.0
    num_hits = 0

    for i, p in enumerate(prediction):
        if p in relevant_items:
            num_hits += 1
            score += num_hits / (i + 1) 
    return score / len(relevant_items)

df_dict = []
for i, file_name in enumerate(os.listdir(result_folder)):
    result_df = pd.read_csv(os.path.join(result_folder, file_name))
    result_list = []
    file_list = []
    malformed_count = 0
    if "test" in file_name:
        pass
    else:
        for idx, row in result_df.iterrows():

            landmarks = extract_landmark_numbers(row['output'])
            if len(landmarks) == 3:
                result_list.append(landmarks)
                file_list.append(row['filename'])
            else:
                malformed_count += 1
        df_dict.append({'model': file_name, 'results': result_list, 'malformed_count': malformed_count, 'files': file_list})
test_df = pd.DataFrame(df_dict)


all_models = []
for index, row in test_df.iterrows():

    results = row['results']
    files = row['files']
    merged_data = []
    for file_name, landmarks in zip(files, results):
        gt_row = gt_df[gt_df['filename'] == file_name]
        if not gt_row.empty:
            gt_landmarks_str = gt_row.iloc[0]['top3_landmarks']
            gt_landmarks = extract_landmark_numbers(gt_landmarks_str)
            merged_data.append({'filename': file_name, 'predicted': landmarks, 'ground_truth': gt_landmarks})
    merged_df = pd.DataFrame(merged_data)
    all_models.append({"model": row['model'], "data": merged_df})

output=[]
for model in all_models:
    model_data = model['data']
    for k in [1,2,3]:
        precisions = []
        recalls = []
        for i, item in model_data.iterrows():
            precisions.append(precision_at_k(item['ground_truth'], item['predicted'], k))
            recalls.append(recall_at_k(item['ground_truth'], item['predicted'], k))

        output.append({"precisions": np.mean(precisions), "recalls":np.mean(recalls),  "k":k, "model":  model['model']})

        
df_output = pd.DataFrame(output)
df_output['model_type'] = df_output['model'].apply(lambda x: 'Baseline' if 'unsloth' in x.lower() or 'medgemma' in x.lower() else 'Ours')
replace_to_space = ["stage1_models_SFT_", "_language16.csv", "-it.csv", "unsloth_", "-VL", "-Instruct.csv"]
for item in replace_to_space:
    df_output['model'] = df_output['model'].str.replace(item, '', regex=False)
df_output['model'] = df_output['model'].str.replace("gemma_", 'gemma-3_', regex=False)
df_output.to_csv("synthetic_retrieval_p_r_k.csv", index=False)