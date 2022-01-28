import rusty_neat
from rusty_neat import ndalgebra as nd
import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import numpy as np
from torch.utils.data import DataLoader


class D(torch.utils.data.Dataset):
    def __init__(self, imgs, lbls):
        self.imgs = imgs
        self.lbls = lbls

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.lbls[idx]


MNIST, LABELS = torch.load('htm/data/mnist.pt')

train_mnist = MNIST[:40000]
train_labels = LABELS[:40000]
train_d = D(train_mnist, train_labels)

eval_mnist = MNIST[40000:60000]
eval_labels = LABELS[40000:60000]
eval_d = D(eval_mnist, eval_labels)

linear = torch.nn.Linear(28 * 28, 10)
loss = torch.nn.NLLLoss()
optim = torch.optim.Adam(linear.parameters())

train_dataloader = DataLoader(train_d, batch_size=64, shuffle=True)
test_dataloader = DataLoader(eval_d, batch_size=64, shuffle=True)

for epoch in range(10):
    accuracy = 0
    total = 0
    for x, y in train_dataloader:
        optim.zero_grad()
        x = x.reshape(x.shape[0], -1)
        x = x.type(torch.float32)
        x = linear(x)
        x = torch.log_softmax(x, dim=1)
        d = loss(x, y)
        accuracy += (x.argmax(1) == y).sum()
        total += x.shape[0]
        d.backward()
        optim.step()
    print(accuracy / total)

# tensor(0.8376)
# tensor(0.8743)
# tensor(0.8798)
# tensor(0.8874)
# tensor(0.8898)
# tensor(0.8895)
# tensor(0.8873)
# tensor(0.8925)
# tensor(0.8902)
# tensor(0.8936)
