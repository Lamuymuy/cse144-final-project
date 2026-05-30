import os, torch, glob
import torch.nn as nn
import pandas as pd
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

DATA_PATH = "/kaggle/input/competitions/ucsc-cse-144-spring-2026-final-project"
TRAIN_DIR = f"{DATA_PATH}/train"
TEST_DIR  = f"{DATA_PATH}/test"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset - verify label mapping
dataset = ImageFolder(TRAIN_DIR, transform=transform)
dataset.class_to_idx = {cls: int(cls) for cls in dataset.classes}
dataset.targets = [int(dataset.classes[t]) for t in dataset.targets]
dataset.samples = [(s, int(dataset.classes[t])) for s, t in dataset.samples]
print("Label mapping sample:", list(dataset.class_to_idx.items())[:5])

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.cuda()

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.size(0)
    print(f"Epoch {epoch+1}: Acc = {correct/total:.3f}")

# Generate submission - all test images, keep .jpg in ID
model.eval()
test_images = sorted(glob.glob(TEST_DIR + "/*.jpg"),
                     key=lambda x: int(os.path.basename(x).replace(".jpg","")))
ids, preds = [], []
with torch.no_grad():
    for path in test_images:
        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0).cuda()
        preds.append(model(img).argmax(1).item())
        ids.append(os.path.basename(path))  # keeps "0.jpg" format

sub = pd.DataFrame({"ID": ids, "Label": preds})
sub.to_csv("submission.csv", index=False)
print(f"Shape: {sub.shape}")
print(sub.head(10))
print("Done!")
