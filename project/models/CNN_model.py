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
num_classes = len(classes)

#model architecture
class Dog_Model(nn.Module):
    def __init__(self,num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3,32,3)
        self.conv2 = nn.Conv2d(32,64,3)
        self.conv3 = nn.Conv2d(64,128,3)

        self.pool = nn.MaxPool2d(2,2)

        self.conv4 = nn.Conv2d(128,256,3)
        self.conv5 = nn.Conv2d(256,512,3)

        self.fc1 = nn.Linear(512 * 21 * 21, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x =self.pool(x)
        x = x.view(x.size(0),-1)
        x= F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

