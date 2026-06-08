# cse144-final-project
# CSE 144 Final Project — Transfer Learning

**Team:** Your Name, Teammate Name  
**Course:** CSE 144, Spring 2026, UC Santa Cruz

## Kaggle Leaderboard
![leaderboard screenshot](assets/leaderboard.png)

## Model Weights
[Download from Google Drive](https://drive.google.com/file/d/1a8c4x00rq9HMVfNBnzXWyyzfIJBGSKqZ/view)

## Setup
```bash
pip install -r requirements.txt
```
## Data
```bash
Data from kaggle competition page /train and /test
```

## Training
```bash
python src/train.py --epochs 20 --lr 0.001 --seed 42
```

## Inference
```bash
python src/predict.py --checkpoint weights/best_model.pth
```
