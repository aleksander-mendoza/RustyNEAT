import rusty_neat
from rusty_neat import ndalgebra as nd
from rusty_neat import ecc
from rusty_neat import htm
import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import numpy as np
from torch.utils.data import DataLoader

sep = ecc.CpuEccSparse(output=[1, 1], kernel=[28, 28], stride=[1, 1],
                       in_channels=1, out_channels=4 * 28 * 28, k=28,
                       connections_per_output=15)
enc = htm.EncoderBuilder()
enc_img = enc.add_image(28, 28, 1, 0.8)


class D(torch.utils.data.Dataset):
    def __init__(self, imgs, lbls):
        self.imgs = imgs
        self.lbls = lbls

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        sdr = htm.CpuSDR()
        enc_img.encode(sdr, img.unsqueeze(2).numpy())
        img_bits = sep.infer(sdr)
        img = torch.zeros(sep.out_shape[2])
        img[list(img_bits)] = 1
        return img, self.lbls[idx]


MNIST, LABELS = torch.load('htm/data/mnist.pt')

train_mnist = MNIST[:40000]
train_labels = LABELS[:40000]
train_d = D(train_mnist, train_labels)

eval_mnist = MNIST[40000:60000]
eval_labels = LABELS[40000:60000]
eval_d = D(eval_mnist, eval_labels)

linear = torch.nn.Linear(sep.out_shape[2], 10)
loss = torch.nn.NLLLoss()
optim = torch.optim.Adam(linear.parameters())

train_dataloader = DataLoader(train_d, batch_size=64, shuffle=True)
test_dataloader = DataLoader(eval_d, batch_size=64, shuffle=True)

for epoch in range(100):
    accuracy = 0
    total = 0
    for x, y in train_dataloader:
        optim.zero_grad()
        x = linear(x)
        x = torch.log_softmax(x, dim=1)
        d = loss(x, y)
        accuracy += (x.argmax(1) == y).sum()
        total += x.shape[0]
        d.backward()
        optim.step()
    print(accuracy / total)
# tensor(0.8037)
# tensor(0.8686)
# tensor(0.8866)
# tensor(0.8985)
# tensor(0.9071)
# tensor(0.9144)
# tensor(0.9195)
# tensor(0.9241)
# tensor(0.9284)
# tensor(0.9315)
# tensor(0.9342)
# tensor(0.9371)
# tensor(0.9397)
# tensor(0.9415)
# tensor(0.9437)
# tensor(0.9451)
# tensor(0.9463)
# tensor(0.9485)
# tensor(0.9502)
# tensor(0.9520)
# tensor(0.9531)
