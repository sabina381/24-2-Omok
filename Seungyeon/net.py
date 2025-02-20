# import
import torch.nn as nn
import torch.nn.functional as F

from config import *

# ResNet
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), bias=False, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        sc = x

        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        x = self.conv(x)
        x = self.bn(x)
        x += sc
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, action_size=ACTION_SIZE, conv_units=CONV_UNITS):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=STATE_DIM, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)
        self.bn = nn.BatchNorm2d(conv_units)
        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=1)
        self.residual_block = ResidualBlock(conv_units, conv_units)

        self.policy_head = nn.Sequential(
            nn.Conv2d(conv_units, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2*action_size, action_size),
            nn.Softmax(dim=1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(conv_units, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(action_size, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)

        # residual block
        for i in range(RESIDUAL_NUM):
            x = self.residual_block(x)

        # pooling
        x = self.pool(x)

        # policy, value 출력
        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value

# Basic Net
class Net(nn.Module):
    def __init__(self, action_size=ACTION_SIZE, conv_units=CONV_UNITS):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=STATE_DIM, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)
        self.conv2 = nn.Conv2d(in_channels=conv_units, out_channels=conv_units * 2, kernel_size=(3,3), bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_units)
        self.bn2 = nn.BatchNorm2d(conv_units * 2)
        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=1)

        self.policy_head = nn.Sequential(
            nn.Conv2d(conv_units* 2, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2*action_size, action_size),
            nn.Softmax(dim=1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(conv_units * 2, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(action_size, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # policy, value 출력
        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value