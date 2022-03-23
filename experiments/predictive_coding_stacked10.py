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

fig, axs = None, None

MNIST, LABELS = torch.load('htm/data/mnist.pt')
L2 = True
METRIC_STR = "l2" if L2 else "l1"
SAMPLES = 60000
DIR = 'predictive_coding_stacked10/' + METRIC_STR + "/" + str(SAMPLES)
ENTROPY_MAXIMISATION = True
MACHINE_TYPE = ecc.CpuEccMachine
DENSE_TYPE = ecc.CpuEccDense
SPLITS = [20, 100, 1000, 0.1]


def preprocess_mnist():
    enc = htm.EncoderBuilder()
    img_w, img_h, img_c = 28, 28, 1
    enc_img = enc.add_image(img_w, img_h, img_c, 0.8)
    sdr_mnist = [htm.CpuSDR() for _ in MNIST]
    for sdr, img in zip(sdr_mnist, MNIST):
        enc_img.encode(sdr, img.unsqueeze(2).numpy())
    sdr_dataset = htm.CpuSdrDataset([28, 28, 1], sdr_mnist)
    return sdr_dataset


class MachineShape:

    def __init__(self, channels, kernels, strides, drifts, k):
        assert len(channels) == len(kernels) + 1
        assert len(kernels) == len(strides) == len(drifts) == len(k)
        self.channels = channels
        self.k = k
        self.kernels = kernels
        self.strides = strides
        self.drifts = drifts

    def composed_conv(self, idx):
        strides = [[s, s] for s in self.strides]
        kernels = [[k, k] for k in self.kernels]
        return htm.conv_compose_array(strides=strides[:idx + 1], kernels=kernels[:idx + 1])

    def code_name_part(self, idx):
        k = "k" + str(self.kernels[idx])
        s = "s" + str(self.strides[idx])
        c = "c" + str(self.channels[idx])
        d = "d" + str(self.drifts[idx])
        k_ = "k" + str(self.k[idx]) if self.k[idx] > 1 else ""
        return k + s + c + k_ + d

    def code_name(self, idx):
        path = ''.join([self.code_name_part(i) + "_" for i in range(idx)])
        return path + "c" + str(self.channels[idx])

    def parent_code_name(self):
        if len(self) == 0:
            return None
        else:
            return self.code_name(len(self) - 1)

    def save_file(self, idx):
        return DIR + "/" + self.code_name(idx)

    def kernel(self, idx):
        return [self.kernels[idx], self.kernels[idx]]

    def stride(self, idx):
        return [self.strides[idx], self.strides[idx]]

    def drift(self, idx):
        return [self.drifts[idx], self.drifts[idx]]

    def __len__(self):
        return len(self.kernels)

    def load_layer(self, idx):
        mf = self.save_file(idx + 1) + ".model"
        if os.path.exists(mf):
            return DENSE_TYPE.load(mf)
        else:
            return None

    def load_or_save_params(self, idx, **kwrds):
        f = self.save_file(idx + 1) + " params.txt"
        if os.path.exists(f):
            with open(f, "r") as f:
                d2 = json.load(f)
                kwrds.update(d2)
        else:
            with open(f, "w+") as f:
                json.dump(kwrds, f)
        return kwrds


class Mnist:

    def __init__(self, machine_shape: MachineShape, layer_idx):
        self.machine_shape = machine_shape
        self.layer_idx = layer_idx
        self.file = machine_shape.save_file(layer_idx) + " data.pickle"
        self.mnist = None
        self.composed_stride, self.composed_kernel = machine_shape.composed_conv(layer_idx)

    def load(self):
        self.mnist = htm.CpuSdrDataset.load(self.file)

    def save_mnist(self):
        self.mnist.save(self.file)


class SingleColumnMachine:

    def __init__(self, machine_shape: MachineShape, w, h, threshold=None):
        assert w * h == machine_shape.channels[-1]
        self.threshold = threshold
        self.machine_shape = machine_shape
        self.m = MACHINE_TYPE()
        self.w = w
        self.h = h
        top = self.machine_shape.load_layer(len(machine_shape) - 1)
        if top is None:
            l = DENSE_TYPE([1, 1],
                      kernel=self.machine_shape.kernel(-1),
                      stride=self.machine_shape.stride(-1),
                      in_channels=self.machine_shape.channels[-2],
                      out_channels=self.machine_shape.channels[-1],
                      k=self.machine_shape.k[-1])
            prev_k = self.machine_shape.k[-2] if len(self.machine_shape.k) > 1 else 1
            if self.threshold == 'in':
                l.threshold = prev_k / l.in_channels
            elif type(self.threshold) is float:
                l.threshold = self.threshold
            else:
                l.threshold = prev_k / l.out_channels
            self.m.push(l)
        else:
            self.m.push(top)
        for idx in reversed(range(len(machine_shape) - 1)):
            self.m.prepend_repeated_column(machine_shape.load_layer(idx))

    def train(self, train_len, eval_len, interval, channels,log=None):
        idx = len(self.machine_shape) - 1
        single_column = self.m.get_layer(idx)
        single_column.plasticity = 1
        drift = self.machine_shape.drift(idx)
        train_data = SDR_MNIST.mnist.subdataset(0, train_len)
        eval_data = SDR_MNIST.mnist.subdataset(train_len, train_len+eval_len)
        compound_stride, compound_kernel = self.machine_shape.composed_conv(idx)
        compound_stride, compound_kernel = compound_stride[:2], compound_kernel[:2]
        full_machine_out_shape = htm.conv_out_size(SDR_MNIST.mnist.shape,compound_stride, compound_kernel)
        deep_conv_out_shape = htm.conv_out_size(htm.conv_out_size(SDR_MNIST.mnist.shape,[1,1], [6,6]), [1,1], [6,6])

        class D(torch.utils.data.Dataset):
            def __init__(self, imgs, lbls):
                self.imgs = imgs
                self.lbls = lbls

            def __len__(self):
                return len(self.imgs)

            def __getitem__(self, idx):
                return self.imgs.to_f32_numpy(idx), self.lbls[idx]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, channels[0], (6, 6))
                self.conv2 = torch.nn.Conv2d(channels[0], channels[1], (6, 6))

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                return x

        train_lbls = LABELS[0:train_len].numpy()
        eval_lbls = LABELS[train_len:train_len+eval_len].numpy()
        train_d = D(train_data, train_lbls)
        eval_d = D(eval_data, eval_lbls)
        loss = torch.nn.NLLLoss()
        BS = 20
        train_dataloader = DataLoader(train_d, batch_size=BS, shuffle=True)
        eval_dataloader = DataLoader(eval_d, batch_size=BS, shuffle=True)
        deep_conv = Net()
        deep_conv_optim = torch.optim.Adam(deep_conv.parameters())
        conv_linear = torch.nn.Linear(deep_conv_out_shape[0]*deep_conv_out_shape[1]*channels[1], 10)
        conv_linear_optim = torch.optim.Adam(conv_linear.parameters())
        all_ecc_accuracies = []
        all_deep_accuracies = []
        while True:

            full_layer = single_column.repeat_column(full_machine_out_shape)
            produced_data = train_data.batch_infer(full_layer)
            produced_d = D(produced_data, train_lbls)
            produced_dataloader = DataLoader(produced_d, batch_size=BS, shuffle=True)
            ecc_linear = torch.nn.Linear(produced_data.volume, 10)
            optim = torch.optim.Adam(ecc_linear.parameters())

            for epoch in range(4):
                for x, y in tqdm(produced_dataloader, desc="train classifier"):
                    optim.zero_grad()
                    bs = x.shape[0]
                    x = x.reshape(bs, -1)
                    x = ecc_linear(x)
                    x = torch.log_softmax(x, dim=1)
                    d = loss(x, y)
                    d.backward()
                    optim.step()
            eval_accuracy = 0
            eval_total = 0
            produced_data = eval_data.batch_infer(full_layer)
            produced_d = D(produced_data, eval_lbls)
            produced_dataloader = DataLoader(produced_d, batch_size=BS, shuffle=True)
            for x, y in tqdm(produced_dataloader, desc="eval ecc"):
                bs = x.shape[0]
                x = x.reshape(bs, -1)
                x = ecc_linear(x)
                eval_accuracy += (x.argmax(1) == y).sum().item()
                eval_total += x.shape[0]
            all_ecc_accuracies.append(eval_accuracy/eval_total)
            eval_accuracy = 0
            eval_total = 0
            with torch.no_grad():
                for x, y in tqdm(eval_dataloader, desc="eval deep"):
                    bs = x.shape[0]
                    x = x.squeeze().unsqueeze(1)
                    x = F.relu(deep_conv(x))
                    x = x.reshape(bs, -1)
                    x = conv_linear(x)
                    eval_accuracy += (x.argmax(1) == y).sum().item()
                    eval_total += x.shape[0]
            all_deep_accuracies.append(eval_accuracy / eval_total)
            print("interval=", interval)
            print("channels=", channels)
            print("deep=", all_deep_accuracies)
            print("ecc=", all_ecc_accuracies)
            train_data.train_with_patches(number_of_samples=interval, drift=drift,
                                          patches_per_sample=1,
                                          ecc=single_column, log=log,
                                          decrement_activities=ENTROPY_MAXIMISATION)
            remaining = interval
            for epoch in range(4):
                for x, y in tqdm(train_dataloader, desc="train conv"):
                    deep_conv_optim.zero_grad()
                    conv_linear_optim.zero_grad()
                    bs = x.shape[0]
                    x = x.squeeze().unsqueeze(1)
                    x = F.relu(deep_conv(x))
                    x = x.reshape(bs,-1)
                    x = conv_linear(x)
                    x = torch.log_softmax(x, dim=1)
                    d = loss(x, y)
                    d.backward()
                    if remaining>0:
                        deep_conv_optim.step()
                    conv_linear_optim.step()
                    remaining -= bs



SDR_MNIST = Mnist(MachineShape([1], [], [], [], []), 0)
if os.path.exists(SDR_MNIST.file):
    SDR_MNIST.load()
else:
    SDR_MNIST.mnist = preprocess_mnist()
    SDR_MNIST.save_mnist()


def run_experiments():
    factorizations = {
        100: (10, 10),
        200: (10, 20),
        256: (16, 16),
        144: (12, 12),
        9: (3, 3),
        6: (3, 2),
        8: (4, 2),
        24: (6, 4),
        32: (8, 4),
        16: (4, 4),
        48: (8, 6),
        49: (7, 7),
        64: (8, 8),
        1: (1, 1),
        20: (5, 4),
        25: (5, 5),
        40: (8, 5),
        400: (20, 20)
    }
    i49 = (49, 6, 1, 1, 1, None, 1)
    e144 = (144, 5, 1, 1, 1, None, 1)
    e200 = (200, 5, 1, 1, 1, None, 1)
    e256 = (256, 5, 1, 1, 1, None, 1)

    def e(channels, kernel, k=1):
        return channels, kernel, 1, 1, 1, None, k

    def c(channels, drift, k=1):
        return channels, 1, 1, drift, 6, 'in', k

    experiments = [
        # (1, [e(49, 6), c(9, 3), e(100, 6), c(9, 5), e(144, 6), c(16, 7), e(256, 6), c(20, 8), e(256, 6)]),
        # (1, [e(49, 6), c(9, 3), e(100, 6, k=4), c(9, 5), e(144, 6, k=3), c(16, 7), e(256, 6, k=4), c(20, 8), e(256, 6, k=4)]),
        # (1, [e(49, 6), e(100, 6,k=4), e(144, 6,k=3), e(256, 6,k=4), e(256, 6,k=4)]),
        # (1, [e(49, 6), e(100, 6), e(144, 6, k=3), e(256, 6, k=4), e(256, 6, k=4)]),
        # (1, [e(49, 6), e(100, 6), e(144, 6), e(256, 6, k=4), e(256, 6, k=4)]),
        # (1, [e(49, 6), e(100, 6), e(144, 6), e(256, 6), e(256, 6, k=4)]),
        #    (1, [e(100, 28)]),
        #    (1, [e(256, 28)]),
        #    (1, [e(400, 28)]),
        #    (1, [e(6, 6), e(6, 6), e(6, 6), e(6, 6), e(6, 6)]),
        #    (1, [e(8, 6), e(2 * 8, 6), e(3 * 8, 6), e(4 * 8, 6), e(4 * 8, 6)]),
        #    (1, [e(8, 6), e(2 * 8, 6), e(3 * 8, 6), e(4 * 8, 6), e(5 * 8, 6)]),
        # (1, [e(8, 6), e(2 * 8, 3), e(3 * 8, 3), e(4 * 8, 3), e(4 * 8, 3), e(4 * 8, 3)]),
        # (1, [e(16, 6), e(2 * 16, 3), e(3 * 16, 3), e(4 * 16, 3), e(4 * 16, 3), e(4 * 16, 3)]),
        # (1, [e(16, 6), e(2 * 16, 6), e(3 * 16, 6), e(4 * 16, 6), e(4 * 16, 6)]),
        (1, [e(49, 6)]),
        # (1, [i49, c9(3), e144, c9(5), e144, c9(7), e144, c9(10), e144, c9(7)]),
        # (1, [i49, c9(3), e144, c9(5), e144, c16(7), e144, c16(10), e144, c16(7), e144, c16(3)]),
        # (1, [i49, c9(3), e144, c9(5), e144, c16(7), e200, c16(10), e200, c20(7), e200, c20(3)]),
        # (1, [i49, c9(3), e144, c9(5), e144, c16(7), e200, c20(10), e256, c25(7), e200, c20(3)]),
    ]
    overwrite_benchmarks = False
    overwrite_data = False
    interval = 500
    for experiment in experiments:
        first_channels, layers = experiment
        kernels, strides, channels, drifts, ks = [], [], [first_channels], [], []
        for layer in layers:
            channel, kernel, stride, drift, snapshots_per_sample, threshold, k = layer
            kernels.append(kernel)
            strides.append(stride)
            channels.append(channel)
            drifts.append(drift)
            ks.append(k)
            s = MachineShape(channels, kernels, strides, drifts, ks)
            code_name = s.save_file(len(kernels))
            save_file = code_name + " data.pickle"
            if overwrite_benchmarks or overwrite_data or not os.path.exists(save_file):
                w, h = factorizations[channel]
                m = SingleColumnMachine(s, w, h, threshold=threshold)
                print(save_file)
                m.train(train_len=12000,eval_len=5000,channels=[8,16], interval=interval)


# print_comparison_across_sample_sizes()

run_experiments()
# edb = ExperimentDB()
# edb.experiment_on_all("softmax", overwrite_benchmarks=True)
# edb.compare_metrics()
# edb.print_accuracy2_results([26, 2], with_drift=False, with_k=False)
# edb.print_accuracy2_results([26, 2], with_drift=False, with_k=False)
