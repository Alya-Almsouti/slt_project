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
    # Filter rows that have a valid .pkl file
    valid = chunk[chunk['vid'].apply(is_valid_entry)]
    # Compute duration from 'start' and 'end' columns (converted to timedelta)
    durations = pd.to_timedelta(valid['end']) - pd.to_timedelta(valid['start'])

    # Filter rows where duration is less than or equal to 10.24 seconds which means 256 frames
    valid = valid[durations <= pd.Timedelta(seconds=10.24)]
    return valid

def main():
    df = pd.read_csv(input_path, sep="\t")
    
    # Determine an appropriate number of workers
    num_workers = min(cpu_count(), len(df))
    chunk_size = (len(df) + num_workers - 1) // num_workers

    # Split data into chunks
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    # Process chunks in parallel
    with Pool(num_workers) as pool:
        results = pool.map(process_chunk, chunks)

    # Concatenate the filtered results
    filtered_df = pd.concat(results, ignore_index=True)

    # Save the filtered data
    filtered_df.to_csv(output_path, sep="\t", index=False)
    print(f"Filtering complete! {len(filtered_df)} valid entries saved to {output_path}")

if __name__ == "__main__":
    main()
