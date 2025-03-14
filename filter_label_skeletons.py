import json
import os
import pickle
import random
import pathlib
import numpy as np
from config import rgb_dirs, pose_dirs
from tqdm import tqdm  # Import tqdm for progress bar
name = "alya.almsouti"
# Paths
annotations_path = f"/l/users/{name}/AI702/Datasets/ClipsDataset/Filtered_openasl-v1.0.json"  # Update this with your actual path
filtered_annotations_path = f"/l/users/{name}/AI702/Datasets/ClipsDataset/Filtered2_openasl-v1.0.json"  # Output file after filtering
pose_dir = pose_dirs["CSL_News"]

# Load annotations
annotations_path = pathlib.Path(annotations_path)
with annotations_path.open(encoding="utf-8") as f:
    annotations = json.load(f)

filtered_annotations = []

for sample in tqdm(annotations, desc="Filtering Annotations", unit="sample"):
    pose_name = sample["pose"]
    pose_path = os.path.join(pose_dir, pose_name)

    try:
        # Load the pose data
        with open(pose_path, "rb") as f:
            pose = pickle.load(f)

        # Ensure 'keypoints' and 'scores' exist
        if "keypoints" not in pose or "scores" not in pose:
            # print(f"Skipping {pose_name} - Missing 'keypoints' or 'scores'")
            continue

        skeletons = pose["keypoints"]
        confs = pose["scores"]
        # print(skeletons.shape)

        # Ensure skeletons and confs are non-empty
        if len(skeletons) == 0 or len(confs) == 0:
            # print(f"Skipping {pose_name} - Empty keypoints or scores")
            continue

        # Ensure all frames have the correct number of keypoints
        skeleton_shapes = [np.array(skel).shape for skel in skeletons]
        if not all(shape == (133, 2) for shape in skeleton_shapes):
            # print(f"Skipping {pose_name} - Inconsistent keypoints shape {skeleton_shapes}")
            continue

        # Ensure confs array has valid shape
        conf_shapes = [np.array(conf).shape for conf in confs]
        if not all(shape == (133,) for shape in conf_shapes):
            # print(f"Skipping {pose_name} - Inconsistent scores shape {conf_shapes}")
            continue

        # If everything is okay, add to filtered list
        filtered_annotations.append(sample)

    except Exception as e:
        # print(f"Skipping {pose_name} due to error: {e}")
        continue

# Save the filtered annotations
with open(filtered_annotations_path, "w", encoding="utf-8") as f:
    json.dump(filtered_annotations, f, indent=4)

print(f"Filtered annotations saved to {filtered_annotations_path}")
# Print total valid instances
print(f"\n✅ Filtering complete. Total valid instances remaining: {len(filtered_annotations)}")