from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


def get_transforms(image_size: int = 32):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def build_dataloader(dataset_name: str, data_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    tf = TwoCropsTransform(get_transforms(32))
    name = dataset_name.lower()
    if name == "cifar10":
        ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tf)
    elif name == "cifar100":
        ds = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=tf)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
