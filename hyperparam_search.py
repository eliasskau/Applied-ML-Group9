"""
Grid search over learning_rate, weight_decay (L2), and l1_lambda.
Trains each config for a limited number of epochs with early stopping
and prints a summary sorted by best val accuracy.

Run:
    python hyperparam_search.py
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dog_emotion.models.CNN_model import Dog_Model

# ---- data config ----
DATA_DIR   = "data/images"
IMG_SIZE   = 224
NUM_CLASSES = 5
BATCH_SIZE  = 32
MAX_EPOCHS  = 20    # keep short for search; bump up for final run
PATIENCE    = 5

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

# ---- grid ----
PARAM_GRID = {
    "lr":           [1e-3, 5e-4],
    "weight_decay": [1e-4, 1e-3],   # L2
    "l1_lambda":    [0.0, 1e-5],    # L1 (0.0 = off)
}


def get_loaders():
    full = ImageFolder(root=DATA_DIR, transform=train_transform)
    n = len(full)
    train_n = int(0.8 * n)
    val_n   = int(0.1 * n)
    test_n  = n - train_n - val_n
    train_ds, val_ds, test_ds = torch.utils.data.random_split(full, [train_n, val_n, test_n])
    val_ds.dataset  = copy.deepcopy(val_ds.dataset)
    val_ds.dataset.transform  = val_transform
    test_ds.dataset = copy.deepcopy(test_ds.dataset)
    test_ds.dataset.transform = val_transform
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, val_loader


def train_one_config(lr, weight_decay, l1_lambda, train_loader, val_loader):
    model = Dog_Model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0
    epochs_no_improve = 0
    val_accs = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if l1_lambda > 0:
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_penalty
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        scheduler.step(val_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"  early stop at epoch {epoch+1}")
            break

    return best_val_acc, val_accs, best_state


def main():
    train_loader, val_loader = get_loaders()

    import itertools
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))

    results = []
    all_curves = []

    for combo in combos:
        params = dict(zip(keys, combo))
        print(f"\nTrying: {params}")
        best_val, val_accs, best_state = train_one_config(
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            l1_lambda=params["l1_lambda"],
            train_loader=train_loader,
            val_loader=val_loader,
        )
        print(f"  best val acc: {best_val:.2f}%")
        results.append((best_val, params, best_state))
        all_curves.append((params, val_accs))

    # sort by best val acc
    results.sort(key=lambda x: x[0], reverse=True)

    print("\n--- results (sorted) ---")
    for rank, (acc, params, _) in enumerate(results, 1):
        print(f"{rank}. val acc: {acc:.2f}%  params: {params}")

    # save best model
    best_acc, best_params, best_state = results[0]
    torch.save(best_state, "models/best_hyperparam_CNN.pth")
    print(f"\nbest config: {best_params}  val acc: {best_acc:.2f}%")
    print("saved to models/best_hyperparam_CNN.pth")

    # plot all val curves
    plt.figure(figsize=(10, 6))
    for params, val_accs in all_curves:
        label = f"lr={params['lr']} wd={params['weight_decay']} l1={params['l1_lambda']}"
        plt.plot(range(1, len(val_accs) + 1), val_accs, label=label)
    plt.xlabel("epoch")
    plt.ylabel("val accuracy (%)")
    plt.title("hyperparam search - val accuracy")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("models/hyperparam_search.png")
    plt.show()
    print("plot saved to models/hyperparam_search.png")


if __name__ == "__main__":
    main()
