import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_transforms(augment=False):
    # ImageNet stats bc model is pretrained 
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]


