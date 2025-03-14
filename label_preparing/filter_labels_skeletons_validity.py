import json
import os
import pickle
import pathlib
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager, cpu_count

name = "alya.almsouti"

annotations_path = f"/l/users/{name}/AI702/Datasets/ClipsDataset/Filtered_openasl-v1.0.json"
filtered_annotations_path = f"/l/users/{name}/AI702/Datasets/ClipsDataset/Labels.json"
pose_dir = f"/l/users/{name}/AI702/Datasets/ClipsDataset/pose_format"

# Load annotations
annotations_path = pathlib.Path(annotations_path)
with annotations_path.open(encoding="utf-8") as f:
    annotations = json.load(f)

def process_sample(sample):
    pose_name = sample["pose"]
    pose_path = os.path.join(pose_dir, pose_name)

    try:
        with open(pose_path, "rb") as f:
            pose = pickle.load(f)

        if "keypoints" not in pose or "scores" not in pose:
            return None, 1, 0, 0, 0, 0

        skeletons = pose["keypoints"]
        confs = pose["scores"]

        if len(skeletons) == 0 or len(confs) == 0:
            return None, 0, 1, 0, 0, 0

        skeleton_shapes = [np.array(skel).shape for skel in skeletons]
        if not all(shape == (133, 2) for shape in skeleton_shapes) or any(np.all(np.array(skel) == 0) for skel in skeletons):
            return None, 0, 0, 1, 0, 0

        conf_shapes = [np.array(conf).shape for conf in confs]
        if not all(shape == (133,) for shape in conf_shapes):
            return None, 0, 0, 0, 1, 0

        return sample, 0, 0, 0, 0, 0
    except Exception as error:
        print(error)
        return None, 0, 0, 0, 0, 1

# Run parallel processing
if __name__ == "__main__":
    with Manager() as manager:
        pool = Pool(processes=cpu_count())
        results = list(tqdm(pool.imap(process_sample, annotations), total=len(annotations), desc="Filtering Annotations", unit="sample"))
        pool.close()
        pool.join()
    
    filtered_annotations = [res[0] for res in results if res[0] is not None]
    cnt1, cnt2, cnt3, cnt4, cnt5 = sum(res[1] for res in results), sum(res[2] for res in results), sum(res[3] for res in results), sum(res[4] for res in results),  sum(res[5] for res in results)
    
    with open(filtered_annotations_path, "w", encoding="utf-8") as f:
        json.dump(filtered_annotations, f, indent=4)
    
    print(f"Filtered annotations saved to {filtered_annotations_path}")
    print(f"\n✅ Filtering complete. Total valid instances remaining: {len(filtered_annotations)}")
    print('Invalid 1:', cnt1)
    print('Invalid 2:', cnt2)
    print('Invalid 3:', cnt3)
    print('Invalid 4:', cnt4)
    print('Invalid 5:', cnt5)

