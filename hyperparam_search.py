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


def train_one_lr(lr, train_loader, val_loader):
    model = Dog_Model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0
    epochs_no_improve = 0
    best_state = None
    val_accs = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            loss = loss + L1_LAMBDA * l1_penalty
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, predicted = torch.max(model(inputs), 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        scheduler.step(val_acc)
        val_accs.append(val_acc)
        print(f"  lr={lr}  epoch {epoch+1}/{MAX_EPOCHS}  val acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"  early stop at epoch {epoch+1}")
            break

    return best_val_acc, val_accs, best_state
