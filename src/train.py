# Save trained weights so they can be uploaded to Drive and reloaded by predict.py
os.makedirs("assets", exist_ok=True)
WEIGHTS_PATH = "assets/resnet18_aug_lr1e-4_20ep.pth"
torch.save(model.state_dict(), WEIGHTS_PATH)
print(f"Saved weights to {WEIGHTS_PATH}")