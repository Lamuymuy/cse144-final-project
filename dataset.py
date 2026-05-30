import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_transforms(augment=False):
    # ImageNet stats since backbone is pretrained
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:  # train only
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomCrop(224, padding=16),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # val and test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

def get_dataloaders(data_dir, batch_size=32, val_split=0.2,
                    augment=False, num_workers=2, seed=42):

    train_tf = get_transforms(augment=augment)
    val_tf   = get_transforms(augment=False)

    full_train = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_tf)

    n_val   = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train_set, val_set = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    # val_set points to same dataset object as train, override its transform
    val_set.dataset.transform = val_tf

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, full_train.classes
