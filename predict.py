import os, glob, argparse
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms, models
from PIL import Image
from utils import set_seed, get_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="assets/resnet18_aug_lr1e-4_50ep.pth")
    parser.add_argument("--test_dir",
        default="/kaggle/input/competitions/ucsc-cse-144-spring-2026-final-project/test")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    # Must match train.py's inference transform exactly (no randomness)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Rebuild the same architecture, then load the trained weights
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    test_images = sorted(
        glob.glob(os.path.join(args.test_dir, "*.jpg")),
        key=lambda x: int(os.path.basename(x).replace(".jpg", "")),
    )

    ids, preds = [], []
    with torch.no_grad():
        for path in test_images:
            img = Image.open(path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            preds.append(model(img).argmax(1).item())
            ids.append(os.path.basename(path))

    sub = pd.DataFrame({"ID": ids, "Label": preds})
    sub.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with shape {sub.shape}")

if __name__ == "__main__":
    main()