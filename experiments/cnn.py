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

fig, axs = None, None

MNIST, LABELS = torch.load('htm/data/mnist.pt')
MNIST = MNIST.type(torch.float)
SAMPLES = [20, 100, 1000, 6000, 12000, 30000, 40000]


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
        self.conv1 = nn.Conv2d(1, 8, (6, 6))
        self.conv2 = nn.Conv2d(8, 32, (6, 6))
        self.top = nn.Linear(32 * 18 * 18, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.top(x)
        return x


net = Net()
criterion = nn.NLLLoss()
optim = torch.optim.Adam(net.parameters())
BS = 32

for samples in SAMPLES:
    train_dataset = MnistDataset(MNIST[:samples], LABELS[:samples])
    test_dataset = MnistDataset(MNIST[samples:], LABELS[samples:])
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BS, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BS, shuffle=True)

    for epoch in range(4):  # loop over the dataset multiple times
        for data in tqdm(trainloader, desc="Training"):
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
    with torch.no_grad():
        train_correct = 0
        test_correct = 0
        for inputs, labels in tqdm(trainloader, desc="Eval trainset"):
            outputs = net(inputs)
            train_correct += (labels == outputs.argmax(dim=1)).sum()

        for inputs, labels in tqdm(testloader, desc="Eval testset"):
            outputs = net(inputs)
            test_correct += (labels == outputs.argmax(dim=1)).sum()
        msg = "samples=" + str(samples) + \
              ", train=" + str(train_correct / len(train_dataset)) + \
              ", test=" + str(test_correct / len(test_dataset))
        print(msg)
        with open("cnn.accuracy","a+") as f:
            print(msg, file=f)
