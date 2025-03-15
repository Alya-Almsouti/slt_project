mt5_path = "./pretrained_weight/mt5-base"
name = 'amal.saqib'
# label paths
train_label_paths = {
                    "CSL_News": f"/l/users/amal.saqib/AI702/Datasets/CSLNews/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.train",
                    "WLASL": "./data/WLASL/labels-2000.train",
                    "Open_ASL": f"/l/users/{name}/AI702/Datasets/ClipsDataset/small_openasl.json"
                    }

dev_label_paths = {
                    "CSL_News": f"/l/users/amal.saqib/AI702/Datasets/CSLNews/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.train",
                    "WLASL": "./data/WLASL/labels-2000.train",
                    "Open_ASL": f"/l/users/{name}/AI702/Datasets/ClipsDataset/small_openasl.json"
                    }

test_label_paths = {
                    "CSL_News": f"/l/users/amal.saqib/AI702/Datasets/CSLNews/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.train",
                    "WLASL": "./data/WLASL/labels-2000.train",
                    "Open_ASL": f"/l/users/{name}/AI702/Datasets/ClipsDataset/small_openasl.json"
                    }


# video paths
rgb_dirs = {
            "CSL_News": f"/l/users/{name}/AI702/Datasets/ClipsDataset/videos",
            "CSL_Daily": './dataset/CSL_Daily/sentence-crop',
            "WLASL": "./dataset/WLASL/rgb_format",
            "Open_ASL": f"/l/users/{name}/AI702/Datasets/ClipsDataset/videos"
            }

# pose paths
pose_dirs = {
            "CSL_News": f"/l/users/{name}/AI702/Datasets/ClipsDataset/pose_format",
            "CSL_Daily": './dataset/CSL_Daily/pose_format',
            "WLASL": "./dataset/WLASL/pose_format",
            "Open_ASL": f"/l/users/{name}/AI702/Datasets/ClipsDataset/pose_format"
            }