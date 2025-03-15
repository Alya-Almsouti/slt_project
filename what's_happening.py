import torch
from datasets import S2T_Dataset_news
import json

# Manually set dataset path (instead of using config.py)
dataset_path = "/l/users/alya.almsouti/AI702/Datasets/ClipsDataset/Labels.json"  # 🔹 Replace with correct path

# Define dummy args object
args = type('', (), {})()  # Creating a dummy object for args
args.rgb_support = False  # Disable RGB for simplicity
args.max_length = 256  # Max sequence length
args.dataset = 'Open_ASL'  # Change if needed

print(f"Loading dataset from: {dataset_path}")

# Initialize dataset
dataset = S2T_Dataset_news(dataset_path, args, phase='train')

# Fetch first sample
sample = dataset[4]
name_sample, pose_sample, text, _, support_rgb_dict = sample

# Print details
print("\n🔹 Sample Summary:")
print(f"Video Name: {name_sample}")
print(f"Text: {text}")

# Inspect pose_sample
print("\n🔹 Pose Sample Details:")
for key, value in pose_sample.items():
    print(f"{key}: Shape={value.shape}, Type={type(value)}")

# Inspect support_rgb_dict (if RGB support is enabled)
if args.rgb_support:
    print("\n🔹 RGB Support Dictionary:")
    for key, value in support_rgb_dict.items():
        print(f"{key}: Shape={value.shape}, Type={type(value)}")
