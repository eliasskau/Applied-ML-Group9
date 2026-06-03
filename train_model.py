import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from dog_emotion.models.CNN_model import Dog_Model
from torch.utils.data import DataLoader, Subset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# data

DATA_DIR = "data/images"
IMG_SIZE = 224
classes = ["angry", "happy", "relaxed", "sad"]
num_classes = len(classes)


# hyper param
learning_rate= 0.0005
num_epochs = 100
batch_size = 32
weight_decay = 1e-4   # L2 regularization
l1_lambda = 1e-5      # L1 regularization

# early stopping
PATIENCE = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    full_dataset = ImageFolder(root=DATA_DIR, transform=train_transform)
    labels = [s[1] for s in full_dataset.samples]

    # stratified split: preserves class distribution in each split
    train_idx, temp_idx = train_test_split(
        range(len(full_dataset)), test_size=0.2, stratify=labels, random_state=SEED)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=SEED)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset   = Subset(full_dataset, val_idx)
    test_dataset  = Subset(full_dataset, test_idx)

    # Apply test transform (no augmentation) to val and test sets using deepcopy to avoid affecting train
    val_copy = copy.deepcopy(val_dataset.dataset)
    val_copy.transform = test_transform
    val_dataset.dataset = val_copy

    test_copy = copy.deepcopy(test_dataset.dataset)
    test_copy.transform = test_transform
    test_dataset.dataset = test_copy

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = Dog_Model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    # weight_decay adds L2 penalty via the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0.0
    epochs_no_improve = 0
    history = {"train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # L1 penalty on weights only (not biases or batchnorm params)
            l1_penalty = sum(p.abs().sum() for name, p in model.named_parameters() if 'weight' in name and 'bn' not in name)
            loss = loss + l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            if i % 50 == 49:
                avg_loss = running_loss / 50
                print(f'Avg loss over last 50 batches: {avg_loss:.3f}')
                running_loss = 0.0

        train_accuracy = 100 * train_correct / train_total

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        scheduler.step(val_accuracy)
        history["train_acc"].append(train_accuracy)
        history["val_acc"].append(val_accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}] Train acc: {train_accuracy:.2f}%  Val acc: {val_accuracy:.2f}%')

        # save best weights
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'models/best_CNN_dog_model.pth')
            print(f'  -> best model saved (val acc: {best_val_acc:.2f}%)')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'  -> no improvement ({epochs_no_improve}/{PATIENCE})')

        if epochs_no_improve >= PATIENCE:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Final evaluation on held-out test set
    model.load_state_dict(torch.load('models/best_CNN_dog_model.pth', weights_only=True))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f'Final Test accuracy: {test_accuracy:.2f}%')

    # bootstrap confidence interval on test accuracy
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, predicted = torch.max(model(inputs), 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    n = len(all_labels)

    bootstrap_accs = []
    rng = np.random.default_rng(42)
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        bootstrap_accs.append(100 * np.mean(all_preds[idx] == all_labels[idx]))

    ci_low  = np.percentile(bootstrap_accs, 2.5)
    ci_high = np.percentile(bootstrap_accs, 97.5)
    print(f'95% confidence interval: ({ci_low:.2f}%, {ci_high:.2f}%)')

    # per-class accuracy
    print('\nPer-class accuracy:')
    for i, cls in enumerate(classes):
        mask = all_labels == i
        cls_acc = 100 * np.mean(all_preds[mask] == all_labels[mask]) if mask.sum() > 0 else 0.0
        print(f'  {cls}: {cls_acc:.2f}% ({mask.sum()} samples)')

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.title('confusion matrix')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.show()
    print('confusion matrix saved to models/confusion_matrix.png')

    # plot train vs val accuracy
    epochs_ran = range(1, len(history["train_acc"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_ran, history["train_acc"], label="train accuracy")
    plt.plot(epochs_ran, history["val_acc"], label="val accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy (%)")
    plt.title("train vs val accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/train_val_accuracy.png")
    plt.show()
    print("plot saved to models/train_val_accuracy.png")