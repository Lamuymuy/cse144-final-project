import os, glob
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from utils import set_seed, get_device

DATA_PATH = "/kaggle/input/competitions/ucsc-cse-144-spring-2026-final-project"
TEST_DIR  = f"{DATA_PATH}/test"

set_seed(42)
device = get_device()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ImageFolder(f"{DATA_PATH}/train", transform=transform)
dataset.class_to_idx = {cls: int(cls) for cls in dataset.classes}
dataset.targets = [int(dataset.classes[t]) for t in dataset.targets]
dataset.samples = [(s, int(dataset.classes[t])) for s, t in dataset.samples]
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
classes = dataset.classes
print("Label mapping sample:", list(dataset.class_to_idx.items())[:5])

# Model - swapped from ResNet to Efficientnet to improve score https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)
model = model.to(device)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(15): # increased epochs from 10 to 15
    model.train()
    total, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.size(0)
    print(f"Epoch {epoch+1}: Acc = {correct/total:.3f}")

# Inference
model.eval()
test_images = sorted(glob.glob(TEST_DIR + "/*.jpg"),
                     key=lambda x: int(os.path.basename(x).replace(".jpg", "")))
ids, preds = [], []
with torch.no_grad():
    for path in test_images:
        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        preds.append(model(img).argmax(1).item())
        ids.append(os.path.basename(path))

sub = pd.DataFrame({"ID": ids, "Label": preds})
sub.to_csv("submission.csv", index=False)
print(f"Shape: {sub.shape}")
print(sub.head(10))
print("Done!")
