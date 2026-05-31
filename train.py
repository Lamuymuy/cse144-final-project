import os, glob
import torch
import torch.nn as nn
import pandas as pd
from torchvision import models
from PIL import Image

from utils import set_seed, get_device
from dataset import get_dataloaders, get_transforms

DATA_PATH = "/kaggle/input/competitions/ucsc-cse-144-spring-2026-final-project"
TEST_DIR  = f"{DATA_PATH}/test"

set_seed(42)
device = get_device()

train_loader, _, classes = get_dataloaders(
    data_dir=DATA_PATH,
    batch_size=32,
    val_split=0.0,
    augment=False,
    seed=42,
)
print("Label mapping sample:", classes[:5])

# Model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
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

# Generate submission - all test images, keep .jpg in ID
model.eval()
test_transform = get_transforms(augment=False)
test_images = sorted(glob.glob(TEST_DIR + "/*.jpg"),
                     key=lambda x: int(os.path.basename(x).replace(".jpg","")))
ids, preds = [], []
with torch.no_grad():
    for path in test_images:
        img = Image.open(path).convert("RGB")
        img = test_transform(img).unsqueeze(0).to(device)
        preds.append(model(img).argmax(1).item())
        ids.append(os.path.basename(path))

sub = pd.DataFrame({"ID": ids, "Label": preds})
sub.to_csv("submission.csv", index=False)
print(f"Shape: {sub.shape}")
print(sub.head(10))
print("Done!")
