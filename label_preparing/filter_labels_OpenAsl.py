import pandas as pd
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # For progress bar

name = 'alya.almsouti'
input_path = Path(f"/l/users/{name}/AI702/Datasets/ClipsDataset/openasl-v1.0.tsv")
output_path = Path(f"/l/users/{name}/AI702/Datasets/ClipsDataset/Filtered_openasl-v1.0.tsv")
pkl_path = Path(f"/l/users/{name}/AI702/Datasets/ClipsDataset/pose_format")

# Function to check if the required .pkl file exists
def is_valid_entry(vid):
    return (pkl_path / f"{vid}.pkl").exists()

def process_chunk(chunk):
    return chunk[chunk['vid'].apply(is_valid_entry)]

def main():
    df = pd.read_csv(input_path, sep="\t")
    
    # Ensure the number of workers is within a valid range
    num_workers = min(cpu_count(), len(df))  # Avoid excessive workers
    chunk_size = (len(df) + num_workers - 1) // num_workers  # Handles uneven chunking

    # Split data into chunks
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    # Use multiprocessing Pool
    with Pool(num_workers) as pool:
        results = pool.map(process_chunk, chunks)

    # Concatenate results
    filtered_df = pd.concat(results, ignore_index=True)

    # Save the filtered data
    filtered_df.to_csv(output_path, sep="\t", index=False)

    print(f"Filtering complete! {len(filtered_df)} valid entries saved to {output_path}")

if __name__ == "__main__":
    main()
