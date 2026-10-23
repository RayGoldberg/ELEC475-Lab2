"""
ELEC 475: Lab 2 - Unified Dataset & DataLoader Factory
Supports CIFAR-10 (primary) and Tiny ImageNet-200 (secondary benchmark).
"""
import os
import urllib.request
import zipfile
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, ImageFolder
from .transforms import get_train_transforms, get_test_transforms

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

def download_and_extract_tiny_imagenet(data_dir: str = "./data"):
    target_dir = os.path.join(data_dir, "tiny-imagenet-200")
    if os.path.exists(target_dir):
        return target_dir

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading Tiny ImageNet-200 from {TINY_IMAGENET_URL}...")
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
        print("Download complete.")

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction complete.")
    return target_dir

class TinyImageNetVal(Dataset):
    """
    Tiny ImageNet-200 official validation split.
    Images live in val/images/ and labels in val/val_annotations.txt
    (tab separated: filename, wnid, bbox...). Class indices follow the
    train ImageFolder ordering so labels line up with the training set.
    """
    def __init__(self, tiny_dir: str, class_to_idx: dict, transform=None):
        self.transform = transform
        img_dir = os.path.join(tiny_dir, "val", "images")
        ann_path = os.path.join(tiny_dir, "val", "val_annotations.txt")
        self.samples = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                fname, wnid = parts[0], parts[1]
                self.samples.append((os.path.join(img_dir, fname), class_to_idx[wnid]))
        self.targets = [t for _, t in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_dataloaders(
    dataset: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_augmentation: bool = True,
    target_size: int = 64
):
    dataset_name = dataset.lower()

    if dataset_name == "cifar10":
        train_tf = get_train_transforms("cifar10", augment=use_augmentation, target_size=target_size)
        val_tf   = get_test_transforms("cifar10", target_size=target_size)

        train_set = CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
        val_set   = CIFAR10(root=data_dir, train=False, download=True, transform=val_tf)
        num_classes = 10
        classes = CIFAR10_CLASSES

    elif dataset_name in ["tiny_imagenet", "tinyimagenet"]:
        tiny_dir = download_and_extract_tiny_imagenet(data_dir)
        train_dir = os.path.join(tiny_dir, "train")

        train_tf = get_train_transforms("tiny_imagenet", augment=use_augmentation, target_size=target_size)
        val_tf   = get_test_transforms("tiny_imagenet", target_size=target_size)

        train_set = ImageFolder(root=train_dir, transform=train_tf)
        val_set   = TinyImageNetVal(tiny_dir, train_set.class_to_idx, transform=val_tf)
        num_classes = 200
        classes = train_set.classes

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from 'cifar10', 'tiny_imagenet'.")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False
    )

    return train_loader, val_loader, num_classes, classes
