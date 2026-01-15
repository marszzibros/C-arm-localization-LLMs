import pandas as pd

df = pd.read_csv("data/navigation_dataset.csv")
df['file_path'] = df['file_path'].str.replace("/gpfs1/home/j/j/jjung2/scratch/vlm_finetuning/","")

df.to_csv("data/navigation_dataset.csv",index=False)