# CSE 144 Final Project — Transfer Learning Challenge

**Team:** Hansa Atreya (hatreya) · Zhiyuan Song (zsong41) · Paola Alvarez (palvare9)
**Course:** CSE 144 Applied Machine Learning, Spring 2026, UC Santa Cruz
**Kaggle Competition:** https://www.kaggle.com/competitions/ucsc-cse-144-spring-2026-final-project

## Kaggle Leaderboard
![leaderboard screenshot](assets/leaderboard.png)
## Model Weights
[Download from Google Drive](https://drive.google.com/file/d/1a8c4x00rq9HMVfNBnzXWyyzfIJBGSKqZ/view)

## Setup

```bash
git clone https://github.com/Lamuymuy/cse144-final-project
cd cse144-final-project
pip install -r requirements.txt
```

Requires PyTorch ≥ 2.0 and torchvision ≥ 0.15.

## Data

Download from the Kaggle competition page and place in the project root, or attach the dataset to your Kaggle Notebook:

/kaggle/input/competitions/ucsc-cse-144-spring-2026-final-project/
├── train/
│   ├── 0/       #10 images per class folder, folders named 0–99
│   ├── 1/
│   └── ...
└── test/
├── 0.jpg ... 999.jpg

If running **locally**, update `DATA_PATH` at the top of `src/train.py` and `src/predict.py` to your local data directory.

> **Note on label mapping:** `ImageFolder` sorts class folders lexicographically (`"0", "1", "10", "11"...`), which scrambles Kaggle labels. `dataset.py` explicitly remaps so folder `"0"` → label 0 through folder `"99"` → label 99. Without this fix, predictions are random regardless of model quality.

## Training

Run on a Kaggle Notebook with the competition dataset attached:

```bash
python src/train.py
```

This will:
1. Load all 1,000 training images with the full augmentation pipeline
2. Fine-tune ResNet-18 (pretrained on ImageNet) for 50 epochs 
3. Write `submission.csv` to the working directory

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 32 |
| Epochs | 50 |
| Frozen epochs | 10 |
| Seed | 42 |

Expected training time: ~15–20 minutes on a Kaggle T4 GPU.

## Inference

To generate `submission.csv` from saved weights without retraining:

1. Download the model weights from the Google Drive link above and place at `assets/resnet18_aug_lr1e-4_20ep.pth`
2. Run:

```bash
python src/predict.py
```

Or specify custom paths:

```bash
python src/predict.py --weights assets/resnet18_aug_lr1e-4_20ep.pth \
                      --test_dir /path/to/test \
                      --out submission.csv
```

This loads the trained ResNet-18, runs inference on all 1,000 test images, and writes `submission.csv`.

## Results

| Configuration | Kaggle Public Leaderboard |
|---|---|
| ResNet-18, no augmentation, lr=1e-3, 10 epochs | 0.51818 |
| EfficientNet-B0, no augmentation, lr=1e-3, 10 epochs | 0.53636 |
| ResNet-18 + augmentation, lr=1e-4, 20 epochs | 0.70000 |
| + RandomGrayscale | 0.71818 |
| + epochs → 30 | 0.73636 |
| + epochs → 40 | 0.74545 |
| **+ epochs → 50, ** | **0.75454**  |

## Reproducing Results

### Option 1 — Retrain from scratch
```bash
python src/train.py
```
Trains ResNet-18 for 50 epochs and writes `submission.csv` directly. Takes ~15 minutes on a Kaggle T4 GPU.

### Option 2 — Inference from d weights (faster)
1. Download weights from the Google Drive link above and place at `assets/resnet18_aug_lr1e-4_20ep.pth`
2. Run:
```bash
python src/predict.py
```
Skips training and writes `submission.csv` immediately using the pretrained weights.
