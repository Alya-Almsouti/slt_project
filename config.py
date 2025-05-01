mt5_path = "./pretrained_weight/mt5-base"
name = 'alya.almsouti'
# label paths
train_label_paths = {
                    "Open_ASL": f"/l/users/{name}/AI702/Datasets/ClipsDataset/manifests/subset_train.json",
                    "sample": "data_sample/sample.json"
                    }

dev_label_paths = {
                    "Open_ASL": f"/l/users/{name}/AI702/Datasets/ClipsDataset/manifests/subset_val.json",
                    "sample": "data_sample/sample.json"
                    }

test_label_paths = {
                    "Open_ASL": f"/l/users/{name}/AI702/Datasets/ClipsDataset/manifests/subset_test.json",
                    "sample": "data_sample/sample.json"
                    }


# video paths
rgb_dirs = {
            "Open_ASL": f"/l/users/{name}/AI702/Datasets/ClipsDataset/videos",
            "sample": "data_sample"
            }

# pose paths
pose_dirs = {
            "CSL_News": f"/l/users/{name}/AI702/Datasets/ClipsDataset/pose_format/pose_format",
            "sample": "data_sample"
            }