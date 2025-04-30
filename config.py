mt5_path = "./pretrained_weight/mt5-base"
name = 'alya.almsouti'
# label paths
train_label_paths = {
                    "Open_ASL": f"/l/users/{name}/AI702/Datasets/ClipsDataset/manifests/subset_train.json",
                    }

dev_label_paths = {
                    "Open_ASL": f"/l/users/{name}/AI702/Datasets/ClipsDataset/manifests/subset_val.json",
                    }

test_label_paths = {
                    "Open_ASL": f"/l/users/{name}/AI702/Datasets/ClipsDataset/manifests/subset_test.json",
                    }


# video paths
rgb_dirs = {
            "Open_ASL": f"/l/users/{name}/AI702/Datasets/ClipsDataset/videos",
            }

# pose paths
pose_dirs = {
            "CSL_News": f"/l/users/{name}/AI702/Datasets/ClipsDataset/pose_format/pose_format",
            }