import pandas as pd
import json

# Load the TSV file
tsv_file = "Filtered_openasl-v1.0.tsv"  # Replace with your TSV file path
df = pd.read_csv(tsv_file, sep="\t")
df['video'] = df['vid'] + '.mp4'
df['pose'] = df['vid'] + '.pkl'
df['text'] = df['raw-text']
# Select the required columns
columns_needed = ["video", "pose", "text"]
df = df[columns_needed]

# Convert to JSON format
json_data = df.to_dict(orient="records")

# Save to a JSON file
json_file = "Filtered_openasl-v1.0.json"  # Replace with your desired output filename
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)

print(f"Conversion complete! JSON file saved as {json_file}")
