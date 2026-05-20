# cse144-final-project
# CSE 144 Final Project — Transfer Learning

**Team:** Your Name, Teammate Name  
**Course:** CSE 144, Spring 2026, UC Santa Cruz

## Kaggle Leaderboard
![leaderboard screenshot](assets/leaderboard.png)

## Model Weights
[Download from Google Drive](LINK_HERE)

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python src/train.py --epochs 20 --lr 0.001 --seed 42
```

## Inference
```bash
python src/predict.py --checkpoint weights/best_model.pth
```
