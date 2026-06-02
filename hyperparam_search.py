"""
Hyperparameter search over learning rate.
Trains each LR config with early stopping and saves the best model.

Run:
    python hyperparam_search.py
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dog_emotion.models.CNN_model import Dog_Model

# data config
DATA_DIR    = "data/images"
IMG_SIZE    = 224
NUM_CLASSES = 5
BATCH_SIZE  = 32

# search config
LR_VALUES  = [1e-3, 5e-4, 1e-4]
MAX_EPOCHS = 15
PATIENCE   = 5
WEIGHT_DECAY = 1e-4
L1_LAMBDA    = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_loaders():
    full = ImageFolder(root=DATA_DIR, transform=train_transform)
    n = len(full)
    train_n = int(0.8 * n)
    val_n   = int(0.1 * n)
    test_n  = n - train_n - val_n
    train_ds, val_ds, _ = torch.utils.data.random_split(full, [train_n, val_n, test_n])
    val_copy = copy.deepcopy(val_ds.dataset)
    val_copy.transform = val_transform
    val_ds.dataset = val_copy
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, val_loader
