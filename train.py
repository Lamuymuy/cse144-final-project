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

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomCrop(224, padding=16),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# inference uses plain transform (no randomness) so predictions are deterministic
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ImageFolder(f"{DATA_PATH}/train", transform=train_transform)
dataset.class_to_idx = {cls: int(cls) for cls in dataset.classes}
dataset.targets = [int(dataset.classes[t]) for t in dataset.targets]
dataset.samples = [(s, int(dataset.classes[t])) for s, t in dataset.samples]
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
classes = dataset.classes
print("Label mapping sample:", list(dataset.class_to_idx.items())[:5])

# Model - ResNet18: smaller model less prone to overfit 1000 training images
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

# Train - weight_decay for regularization, label_smoothing to prevent overconfidence
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

for epoch in range(50):
    model.train()
    total, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    print(f"Epoch {epoch+1}: Acc = {correct/total:.3f}")

# Save trained weights so they can be uploaded to Drive and reloaded by predict.py
WEIGHTS_PATH = "/kaggle/working/resnet18_aug_lr1e-4_50ep.pth"
torch.save(model.state_dict(), WEIGHTS_PATH)
print(f"Saved weights to {WEIGHTS_PATH}")

# Inference with TTA
model.eval()
tta_transforms = [
    transform,
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
]

test_images = sorted(glob.glob(TEST_DIR + "/*.jpg"),
                     key=lambda x: int(os.path.basename(x).replace(".jpg", "")))
ids, preds = [], []
with torch.no_grad():
    for path in test_images:
        img = Image.open(path).convert("RGB")
        probs = torch.zeros(100).to(device)
        for tf in tta_transforms:
            probs += model(tf(img).unsqueeze(0).to(device)).softmax(dim=1).squeeze()
        preds.append(probs.argmax().item())
        ids.append(os.path.basename(path))
sub = pd.DataFrame({"ID": ids, "Label": preds})
sub.to_csv("submission.csv", index=False)
print(f"Shape: {sub.shape}")
print(sub.head(10))
print("Done!")