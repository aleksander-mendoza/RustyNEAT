import math
import re
import rusty_neat
import os
from rusty_neat import ndalgebra as nd
from rusty_neat import htm
from rusty_neat import ecc
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import copy
from matplotlib import pyplot as plt
import torch
import numpy as np
import json
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
import time

fig, axs = None, None

MNIST, LABELS = torch.load('htm/data/mnist.pt')
MACHINE_TYPE = ecc.CpuEccMachine
DENSE_TYPE = ecc.CpuEccDense
m = ecc.CpuEccMachine(output=[1, 1], kernels=[[6, 6]], strides=[[1, 1]],
                      channels=[1, 49], k=[1])


def preprocess_mnist():
    enc = htm.EncoderBuilder()
    img_w, img_h, img_c = 28, 28, 1
    enc_img = enc.add_image(img_w, img_h, img_c, 0.8)
    sdr_mnist = [htm.CpuSDR() for _ in MNIST]
    for sdr, img in zip(sdr_mnist, MNIST):
        enc_img.encode(sdr, img.unsqueeze(2).numpy())
    sdr_dataset = htm.CpuSdrDataset([28, 28, 1], sdr_mnist)
    return sdr_dataset


mnist = preprocess_mnist()

s = 10000
a = time.time()
mnist.train_machine_with_patches(s, m)
b = time.time()
print("Took ", b - a, "seconds")
print("Speed:", s / (b - a), "samples per second")
print("Input shape:", m.in_shape)
print("Speed:", m.in_volume * s / (b - a), "pixels per second")
