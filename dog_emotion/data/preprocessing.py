"""
to use:
    from dog_emotion.data.preprocessing import get_dataloaders, CLASSES

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="data/images",
        img_size=224,
        batch_size=32,
        num_workers=2,
    )
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

CLASSES = ["alert", "angry", "frown", "happy", "relax"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class DogEmotionDataset(Dataset):
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, data_dir: str, transform=None):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []  # (image_path, label_index)

        for label_idx, class_name in enumerate(CLASSES):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Expected class folder not found: {class_dir}"
                )
            for filename in sorted(os.listdir(class_dir)):
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.VALID_EXTENSIONS:
                    self.samples.append(
                        (os.path.join(class_dir, filename), label_idx)
                    )

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def labels(self) -> list[int]:
        """All labels in order — used for stratified splitting."""
        return [label for _, label in self.samples]

def _build_transforms(img_size: int, augment: bool) -> transforms.Compose:
    base = [
        transforms.Resize((img_size, img_size)),
    ]

    if augment:
        augmentations = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
            ),
        ]
        base.extend(augmentations)

    base.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return transforms.Compose(base)

def _stratified_split(
    labels: list[int],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    indices = np.arange(len(labels))
    labels_arr = np.array(labels)

    temp_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=labels_arr,
        random_state=random_state,
    )

    adjusted_val_ratio = val_ratio / (1.0 - test_ratio)
    train_idx, val_idx = train_test_split(
        temp_idx,
        test_size=adjusted_val_ratio,
        stratify=labels_arr[temp_idx],
        random_state=random_state,
    )

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

def get_dataloaders(
    data_dir: str = "data/images",
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    base_dataset = DogEmotionDataset(data_dir, transform=None)
    labels = base_dataset.labels

    train_idx, val_idx, test_idx = _stratified_split(
        labels,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    train_transform = _build_transforms(img_size, augment=True)
    eval_transform  = _build_transforms(img_size, augment=False)

    train_dataset = DogEmotionDataset(data_dir, transform=train_transform)
    eval_dataset  = DogEmotionDataset(data_dir, transform=eval_transform)

    train_subset = Subset(train_dataset, train_idx)
    val_subset   = Subset(eval_dataset,  val_idx)
    test_subset  = Subset(eval_dataset,  test_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,          # shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Dataset split (random_state={random_state}):")
    print(f"  Train : {len(train_subset):>5} images")
    print(f"  Val   : {len(val_subset):>5} images")
    print(f"  Test  : {len(test_subset):>5} images")
    print(f"  Total : {len(base_dataset):>5} images")
    print(f"Image size : {img_size}x{img_size}, batch size : {batch_size}")
    print(f"Normalization: mean={IMAGENET_MEAN}, std={IMAGENET_STD}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """
    Run from the repo root to verify the pipeline:
        python dog_emotion/data/preprocessing.py
    """
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="data/images",
        img_size=224,
        batch_size=32,
    )

    images, labels = next(iter(train_loader))
    print(f"\nSanity check — first training batch:")
    print(f"  images shape : {images.shape}")
    print(f"  labels shape : {labels.shape}")   
    print(f"  pixel min    : {images.min():.3f}")
    print(f"  pixel max    : {images.max():.3f}")
    print(f"  unique labels in batch: {labels.unique().tolist()}")