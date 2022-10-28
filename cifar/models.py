import torch
import torch.nn as nn
from anypacking import quant_module

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        conv=nn.Conv2d
        self.conv1 = conv(3,6,5)
        self.conv2 = conv(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VGG_small(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_small, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.nonlinear = nn.ReLU(inplace=True)

        self.layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False), # 0
            nn.BatchNorm2d(128),
            self.nonlinear,

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False), # 1
            self.pooling,
            nn.BatchNorm2d(128),
            self.nonlinear,

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False), # 2
            nn.BatchNorm2d(256),
            self.nonlinear,

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), # 3
            self.pooling,
            nn.BatchNorm2d(256),
            self.nonlinear,

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False), # 4
            nn.BatchNorm2d(512),
            self.nonlinear,

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False), # 5
            self.pooling,
            nn.BatchNorm2d(512),
            self.nonlinear,

            nn.Flatten(),
            nn.Linear(512*4*4, num_classes)
        )


    def forward(self, x):
        return self.layers(x)

class VGG_tiny(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_tiny, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.nonlinear = nn.ReLU(inplace=True)

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False), # 0
            nn.BatchNorm2d(64),
            self.nonlinear,

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), # 1
            self.pooling,
            nn.BatchNorm2d(64),
            self.nonlinear,

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), # 2
            nn.BatchNorm2d(128),
            self.nonlinear,

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False), # 3
            self.pooling,
            nn.BatchNorm2d(128),
            self.nonlinear,

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False), # 4
            nn.BatchNorm2d(256),
            self.nonlinear,

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), # 5
            self.pooling,
            nn.BatchNorm2d(256),
            self.nonlinear,

            nn.Flatten(),
            nn.Linear(256*4*4, num_classes)
        )


    def forward(self, x):
        return self.layers(x)
