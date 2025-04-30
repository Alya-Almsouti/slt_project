# SLT Project – Sign Language Translation

Sign Language Translation (SLT) plays a crucial role in making communication more inclusive for the Deaf and Hard-of-Hearing community. By leveraging AI and deep learning, SLT systems aim to automatically convert sign language videos into spoken language, bridging the accessibility gap in education, media, and daily communication.


## 🔄 Updates on Uni-Sign Repo

While this project adopts the training and validation pipelines from the original [Uni-Sign](https://github.com/yc930401/Uni-Sign), we made significant architectural and design modifications to improve modularity, extensibility, and compatibility.

### 📁 Dataset Compatibility (`dataset.py`)

- The dataset class was redesigned to support the [OpenASL](https://github.com/chevalierNoir/OpenASL) dataset.
- Changes include customized video/keypoint loading and preprocessing logic tailored for OpenASL's structure.

---

### 🧠 Modular Model Design (`models.py`)

We refactored the model to support plug-and-play **visual and keypoint feature extractors**, enabling easy experimentation with different backbones and input modalities:

#### 🎞️ RGB Video Feature Extractors

The model supports interchangeable video encoders via `args.vid_extractor`.

**Supported options:**

- `resnet18`
- `EfficientNetV2`
- `ViT`
- `ConvNeXt`
- `i3d`
- `MobileNetV3`

**How to use:**

- Set `--rgb_support` in bash file
- Choose extractor with `--vid_extractor resnet` (or any from the list)

**To add your own:**

1. Implement the extractor in `vid_extractors.py`
2. Register it in `models.py` with an `if video_extractor == 'your_extractor'` clause


---

#### 🕴️ Skeleton Keypoint Feature Extractors

The model also supports modular skeleton-based extractors, such as `UniSignGNNSkeletonExtractor`.

**How to use:**

- Set `--skeleton_support` in bash file
- Choose extractor with `--skeleton_extractor unisign`

**To add your own:**

1. Implement your extractor in `keypoints_extractor.py`
2. Register it in `models.py` with an `if skeleton_extractor == 'your_extractor'` clause

---

#### 🔗 Multi-Modal Feature Fusion

If both RGB and skeleton inputs are enabled:

- Extracted features are concatenated.
- A linear fusion layer (`fusion_layer`) maps them into the input space of the mT5 encoder.

---

This modular structure enables rapid experimentation with different input modalities and backbone architectures for sign language translation.


## 📦 Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/Alya-Almsouti/slt_project.git
cd slt_project

# 2. Create a new conda environment
conda create -n slt_env python=3.10
conda activate slt_env

# 3. Install PyTorch3D (follow instructions based on your CUDA version)
→ https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download the OpenASL dataset (follow guide)
 → https://github.com/chevalierNoir/OpenASL


# 7. Download the pretrained mT5 model
python download_scripts/install_mt5_model.py
```
## 🚀 Usage

Once setup is complete, you can start training or evaluating models using:

```bash
#for training:
bash script/train.sh 

#for evaluation:
bash script/eval.sh
```

## 📁 Project Structure

- `models.py` – Main model architecture, including feature fusion and MT5 integration  
- `vid_extractors.py` – Video-based feature extractor classes (e.g., ResNet, ViT, i3d)  
- `keypoints_extractor.py` – Skeleton/keypoint-based feature extractor classes  
- `dataset.py` – Custom dataset class adapted for OpenASL  
- `pre_training.py` – Main training and evaluation pipeline  
- `config.py` – Centralized configuration for paths and model options  
- `download_scripts/` – Scripts to download pretrained models (e.g., mT5)  
- `output_vis/` –  Scripts to see model performance using saved outputs from training 
- `script/` – Shell scripts to launch training and evaluation  


## 📜 License

This project is for academic and research purposes only.

---

### 🔗 Credits

- Original framework: [Uni-Sign](https://github.com/yc930401/Uni-Sign)  
- Dataset: [OpenASL](https://github.com/chevalierNoir/OpenASL)
