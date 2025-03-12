import json
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # For progress bar

# Input and output file paths
input_path = Path("/l/users/amal.saqib/AI702/CSLNews/CSL_News_Labels.json")
output_path = Path("/l/users/amal.saqib/AI702/CSLNews/Filtered_CSL_News_Labels.json")
pose_path = '/l/users/amal.saqib/AI702/CSLNews/pose_format'
rgb_path = '/l/users/amal.saqib/AI702/CSLNews/rgb_format'
# Function to check if the required files exist
def is_valid_entry(entry):
    if "pose" in entry and "video" in entry:
        pose , vid = entry['pose'], entry['video']
        pose_full = f'{pose_path}/{pose}'
        rgb_full = f'{rgb_path}/{vid}'
        Path(pose_full).exists() and Path(rgb_full).exists()
    return False
# Function to process a chunk of data
def process_chunk(chunk):
    return [entry for entry in chunk if is_valid_entry(entry)]

def main():
    # Load JSON file
    with input_path.open("r", encoding="utf-8") as f:
        annotation = json.load(f)

    # Ensure the number of workers is within a valid range
    num_workers = min(cpu_count(), len(annotation))  # Avoid excessive workers
    chunk_size = (len(annotation) + num_workers - 1) // num_workers  # Handles uneven chunking

    # Split data into chunks
    chunks = [annotation[i:i + chunk_size] for i in range(0, len(annotation), chunk_size)]

    # Use multiprocessing Pool
    with Pool(num_workers) as pool:
        results = pool.map(process_chunk, chunks)

    # Flatten results
    filtered_data = [entry for sublist in results for entry in sublist]

    # Save the filtered data
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Filtering complete! {len(filtered_data)} valid entries saved to {output_path}")

if __name__ == "__main__":
    main()
