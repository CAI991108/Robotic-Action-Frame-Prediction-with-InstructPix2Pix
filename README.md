# Robotic Action Frame Prediction with InstructPix2Pix

This repository contains the code and configuration files for training a multimodal fine-tuned `InstructPix2Pix` model to predict future robotic action frames. The model generates 256×256 resolution images conditioned on a current observation and textual instruction (e.g., *"stack blocks"*, *"beat the blocks with hammer"*). Results achieve *SSIM* up to **0.98** and *PSNR* over **40 dB** on synthetic RoboTwin tasks.

---

## 📋 Table of Contents
- [Environment Setup](#-environment-setup)
- [Data Preparation](#-data-preparation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Reproducibility Notes](#-reproducibility-notes)
- [Citation](#-citation)

---

## 🛠️ Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/CAI991108/robotic-frame-prediction.git
cd robotic-frame-prediction
```
### 2. Create a Conda Environment
```bash
cd instruct-pix2pix
conda env create -f environment.yaml
conda activate ip2p
bash scripts/download_checkpoints.sh
```
### Download Pretrained Models
```bash
bash scripts/download_pretrained_sd.sh  # Stable Diffusion v1.5
bash scripts/download_checkpoints.sh   # InstructPix2Pix
```
---

## 📊 Data Preparation
### 1. Generate RoboTwin Dataset
- Follow [RoboTwin's official guide](https://github.com/TianxingChen/RoboTwin/tree/main) to generate episodes for three tasks:
    - `block_hammer_beat`
    - `block_handover`
    - `block_stack_easy`
- Place generated data in `./RoboTwin_data`.
### 2. Preprocess Data
- Use the provided script to preprocess the RoboTwin dataset:
```bash
# Step 1: Extract frames and map to instructions
python ./RoboTwin/test.py --root_dir <your_RoboTwin_data_dir> --output_jsonl instructpix2pix_dataset.jsonl

# Step 2: Convert to InstructPix2Pix-compatible format
python ./instruct-pix2pix/data/instructpix2pix/data_prepare.py --input_jsonl instructpix2pix_dataset.jsonl --output_dir <your_output_dir>
```
---
## 🚀 Training
### 1. Configure Paths
- Edit the `./instruct-pix2pix/configs/train.yaml` file to set the paths for the dataset and checkpoints:
```yaml
data:
  params:
    batch_size: 2   # Batch size for training
    num_workers: 2   # Number of workers for data loading
    train:
      params:
        path: ./data/instructpix2pix  # Path to preprocessed data
```
### 2. Start Training
```bash
python ./instruct-pix2pix/main.py \
  --name default \
  --base configs/train.yaml \
  --train \
  --gpus 0,1  # Use 2 GPUs
```
### Key Training Parameters:
- **Batch size**: `2` per GPU (effective `16` with gradient accumulation)
- **Learning rate**: `1e-4` (AdamW optimizer)
- **Epochs**: `100`

---
## 📈 Evaluation
### Visualize Predictions and SSIM/PSNR Scores
```bash
python ./instruct-pix2pix/eval.py --ckpt logs/train_default/checkpoints/last.ckpt
```
---

## 🔄 Reproducibility Notes
### Hardware
- **GPUs**: 2× NVIDIA RTX 2080 Ti (22GB VRAM each)
- **RAM**: 134 GB
### Common Issues
- **Dependency Conflicts**: Ensure exact versions in `requirements.txt` and `environment.yaml`.
- **OOM Errors**: Reduce batch size, enable `--half-precision` or set `use_ema: false`.
- **Dataset Paths**: Verify paths in `train.yaml` and `data_prepare.py`.