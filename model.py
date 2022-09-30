import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super().__init__()
        # input 28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv1_bn = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),  bias=False)
        self.conv2_bn = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),  bias=False)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),  bias=False)
        self.conv4_bn = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(100352, num_classes)
        self.fc1_bn = nn.BatchNorm1d(10)

    def forward(self, x):
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        flat = conv4.reshape(conv4.shape[0], -1)
        logits = self.fc1_bn(self.fc1(flat))
        return F.log_softmax(logits, dim=1)

