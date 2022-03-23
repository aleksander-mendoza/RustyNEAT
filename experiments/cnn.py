import math
import re
import rusty_neat
import os
from rusty_neat import ndalgebra as nd
from rusty_neat import htm
from rusty_neat import ecc
import pandas as pd
import copy
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
import time

MNIST, LABELS = torch.load('htm/data/mnist.pt')
MNIST = MNIST.type(torch.float)


class MnistDataset(Dataset):
    def __init__(self, imgs, lbls):
        self.imgs = imgs
        self.lbls = lbls

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.lbls[idx]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 49, (6, 6))
        # self.conv2 = nn.Conv2d(49, 49, (6, 6))
        # self.conv3 = nn.Conv2d(49, 49, (6, 6))
        self.top = nn.Linear(49 * 13 * 13, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.top(x)
        return x


net = Net()
criterion = nn.NLLLoss()
optim = torch.optim.Adam(net.parameters())
BS = 32

train_dataset = MnistDataset(MNIST, LABELS)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BS, shuffle=True)

a = time.time()
for data in trainloader:
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data

    # zero the parameter gradients
    optim.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    outputs = F.log_softmax(outputs, dim=1)
    loss = criterion(outputs, labels)
    loss.backward()
    optim.step()
b = time.time()

print("Took ", b - a, "seconds")
print("Speed:", len(MNIST) / (b - a), "samples per second")
print("Input shape:", 28*28)
print("Speed:", 28*28 * len(MNIST) / (b - a), "pixels per second")
