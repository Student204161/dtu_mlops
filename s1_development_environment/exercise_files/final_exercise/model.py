from torch import nn
import torch

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        #torch.set_default_dtype(torch.float32)

        self.conv1 = nn.Conv2d(1, 64,5)
        self.maxpool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(64, 64,3)
        self.maxpool2 = nn.MaxPool2d(2)

        self.fc3 = nn.Linear(64*3*3, 10)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)
        self.batchnorm1 = torch.nn.BatchNorm2d(64)
        self.batchnorm2 = torch.nn.BatchNorm2d(64)
        self.batchnorm3 = torch.nn.BatchNorm1d(10)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = torch.flatten(self.maxpool2(x),start_dim=1)
        x = self.fc3(x)
        x = self.batchnorm3(x)
        x = self.softmax(x)
        return x
