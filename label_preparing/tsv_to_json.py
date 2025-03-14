import pandas as pd
import json
import os

name = "alya.almsouti"
tsv_file = f"/l/users/{name}/AI702/Datasets/ClipsDataset/Filtered_openasl-v1.0.tsv" 

df = pd.read_csv(tsv_file, sep="\t")
df['video'] = df['vid'] + '.mp4'
df['pose'] = df['vid'] + '.pkl'
df['text'] = df['raw-text']
columns_needed = ["video", "pose", "text"]
df = df[columns_needed]

json_data = df.to_dict(orient="records")

json_file = f"/l/users/{name}/AI702/Datasets/ClipsDataset/Filtered_openasl-v1.0.json"   # Replace with your desired output filename
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)

os.remove(tsv_file)
print(f"Conversion complete! JSON file saved as {json_file}")
