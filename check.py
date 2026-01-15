import os
import pandas as pd
folder_path = "results/real_localization"

for file in os.listdir(folder_path):
    if "test" in file:
        pass
    else:
        df = pd.read_csv(os.path.join(folder_path, file))
        df['filename'] = df['filename'].str.replace("/gpfs1/home/j/j/jjung2/","")
        df.to_csv(os.path.join(folder_path, file),index=False)