import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# data

DATA_DIR = "data/images"
IMG_SIZE = 384
classes = ["alert", "angry", "frown", "happy", "relax"]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((IMG_SIZE,IMG_SIZE)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

full_dataset = ImageFolder(root=DATA_DIR, transform=transform)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
testloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

#model architecture

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

