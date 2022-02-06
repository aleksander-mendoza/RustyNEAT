import math

import rusty_neat
import os
from rusty_neat import ndalgebra as nd
from rusty_neat import htm
from rusty_neat import ecc
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


def visualise_connection_heatmap(in_w, in_h, ecc_net, out_w, out_h, pause=None):
    global fig, axs
    if fig is None:
        fig, axs = plt.subplots(out_w, out_h)
        for a in axs:
            for b in a:
                b.set_axis_off()
    for i in range(out_w):
        for j in range(out_h):
            w = ecc_net.get_weights(i + j * out_w)
            w = np.array(w)
            w = w.reshape(in_w, in_h)
            w.strides = (8, 56)
            axs[i, j].imshow(w)
    if pause is None:
        plt.show()
    else:
        plt.pause(pause)


def visualise_recursive_weights(w, h, ecc_net):
    fig, axs = plt.subplots(w, h)
    for a in axs:
        for b in a:
            b.set_axis_off()
    for i in range(w):
        for j in range(h):
            weights = np.array(ecc_net.get_weights(i + j * w))
            weights[i + j * w] = 0
            weights = weights.reshape([w, h]).T
            axs[i, j].imshow(weights)
    plt.show()


def compute_confusion_matrix_fit(conf_mat, ecc_net, metric_l2=True):
    assert ecc_net.out_grid == [1, 1]
    fit = torch.empty(ecc_net.out_channels)
    for i in range(ecc_net.out_channels):
        corr = conf_mat[i] / conf_mat[i].sum()
        wei = torch.tensor(ecc_net.get_weights(i))
        if metric_l2:
            fit[i] = corr @ wei
        else:  # l1
            fit[i] = (corr - wei).abs().sum()
    return fit


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

    def __init__(self, channels, kernels, strides):
        assert len(channels) == len(kernels) + 1 == len(strides) + 1
        self.channels = channels
        self.kernels = kernels
        self.strides = strides

    def composed_conv(self, idx):
        strides = [[s, s] for s in self.strides]
        kernels = [[k, k] for k in self.kernels]
        return htm.conv_compose_array(strides=strides[:idx + 1], kernels=kernels[:idx + 1])

    def save_file(self, idx):
        prefix = "predictive_coding_stacked8/"
        path = ''.join(["k" + str(k) + "s" + str(s) + "c" + str(c) + "_" for k, c, s in
                        zip(self.kernels[:idx], self.channels[:idx], self.strides[:idx])])
        return prefix + path + "o" + str(self.channels[idx])

    def kernel(self, idx):
        return [self.kernels[idx], self.kernels[idx]]

    def stride(self, idx):
        return [self.strides[idx], self.strides[idx]]

    def __len__(self):
        return len(self.kernels)

    def load_layer(self, idx):
        mf = self.save_file(idx + 1) + ".model"
        if os.path.exists(mf):
            return ecc.CpuEccDense.load(mf)
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
        self.m = ecc.CpuEccMachine()
        self.w = w
        self.h = h
        top = self.machine_shape.load_layer(len(machine_shape) - 1)
        if top is None:
            l = ecc.CpuEccDense([1, 1],
                                kernel=self.machine_shape.kernel(-1),
                                stride=self.machine_shape.stride(-1),
                                in_channels=self.machine_shape.channels[-2],
                                out_channels=self.machine_shape.channels[-1],
                                k=1)
            if self.threshold == 'in':
                l.threshold = 1 / l.in_channels
            elif type(self.threshold) is float:
                l.threshold = self.threshold
            self.m.push(l)
        else:
            self.m.push(top)
        for idx in reversed(range(len(machine_shape) - 1)):
            self.m.prepend_repeated_column(machine_shape.load_layer(idx))

    def train(self, plot=False, save=True, log=None, drift=[1, 1],
              snapshots_per_sample=1, iterations=1000000,
              interval=100000, test_patches=20000):
        idx = len(self.machine_shape) - 1
        layer = self.m.get_layer(idx)
        params = self.machine_shape.load_or_save_params(
            idx,
            w=self.w,
            h=self.h,
            drift=drift,
            snapshots_per_sample=snapshots_per_sample,
            iterations=iterations,
            interval=interval,
            test_patches=test_patches,
        )
        w = params['w']
        h = params['h']
        drift = params['drift']
        snapshots_per_sample = params['snapshots_per_sample']
        iterations = params['iterations']
        interval = params['interval']
        test_patches = params['test_patches']
        mnist = Mnist(self.machine_shape, idx)
        mnist.load()
        compound_stride, compound_kernel = self.machine_shape.composed_conv(idx)
        compound_stride, compound_kernel = compound_stride[:2], compound_kernel[:2]
        test_patch_indices = mnist.mnist.gen_rand_conv_subregion_indices_with_ecc(layer, test_patches)
        test_input_patches = mnist.mnist.conv_subregion_indices_with_ecc(layer, test_patch_indices)
        test_img_patches = SDR_MNIST.mnist.conv_subregion_indices_with_ker(compound_kernel, compound_stride,
                                                                           test_patch_indices)
        print("PATCH_SIZE=", test_img_patches.shape)
        all_missed = []
        all_total_sum = []
        if plot:
            fig, axs = plt.subplots(self.w, self.h)
            for a in axs:
                for b in a:
                    b.set_axis_off()

        def eval_ecc():
            test_outputs, s_sums, missed = test_input_patches.batch_infer_and_measure_s_expectation(layer)
            receptive_fields = test_img_patches.measure_receptive_fields(test_outputs)
            receptive_fields = receptive_fields.squeeze(2)
            all_missed.append(missed / test_patches)
            all_total_sum.append(s_sums)
            print("missed=", all_missed)
            print("sums=", all_total_sum)
            if plot:
                for i in range(w):
                    for j in range(h):
                        axs[i, j].imshow(receptive_fields[:, :, i + j * w])
                plt.pause(0.01)
                if save:
                    img_file_name = self.machine_shape.save_file(idx + 1) + " before.png"
                    if s > 0 or os.path.exists(img_file_name):
                        img_file_name = self.machine_shape.save_file(idx + 1) + " after.png"
                    plt.savefig(img_file_name)

        for s in tqdm(range(int(math.ceil(iterations / interval))), desc="training"):
            eval_ecc()
            mnist.mnist.train_with_patches(interval, drift, snapshots_per_sample, layer, log)
            if save:
                layer.save(self.machine_shape.save_file(idx + 1) + ".model")
        eval_ecc()
        with open(self.machine_shape.save_file(idx + 1) + ".log", "a+") as f:
            print("missed=", all_missed, file=f)
            print("sums=", all_total_sum, file=f)
        # if plot:
        #     plt.show()


class FullColumnMachine:

    def __init__(self, machine_shape: MachineShape):
        self.machine_shape = machine_shape
        self.m = ecc.CpuEccMachine()
        bottom = machine_shape.load_layer(0)
        out_size = htm.conv_out_size([28, 28], bottom.stride, bottom.kernel)[:2]
        bottom = bottom.repeat_column(out_size)
        self.m.push(bottom)
        for idx in range(1, len(machine_shape)):
            self.m.push_repeated_column(machine_shape.load_layer(idx))

    def eval_with_classifier_head(self, overwrite_data=False, overwrite_benchmarks=False):
        idx = len(self.machine_shape) - 1
        print("PATCH_SIZE=", self.m.in_shape)
        layer = self.m.get_layer(idx)
        benchmarks_save = self.machine_shape.save_file(idx + 1) + " accuracy.txt"
        out_mnist = Mnist(self.machine_shape, idx + 1)
        if os.path.exists(out_mnist.file) and not overwrite_data:
            out_mnist.load()
        else:
            in_mnist = Mnist(self.machine_shape, idx)
            in_mnist.load()
            out_mnist.mnist = in_mnist.mnist.batch_infer(layer)
            out_mnist.save_mnist()
        if not os.path.exists(benchmarks_save) or overwrite_benchmarks:
            with open(benchmarks_save, 'w+') as f:
                for split in [0.1, 0.2, 0.5, 0.8, 0.9]:
                    train_len = int(len(MNIST) * split)
                    eval_len = len(MNIST) - train_len
                    train_data = out_mnist.mnist.subdataset(0, train_len)
                    eval_data = out_mnist.mnist.subdataset(train_len)
                    train_lbls = LABELS[0:train_len].numpy()
                    eval_lbls = LABELS[train_len:].numpy()
                    lc = train_data.fit_linear_regression(train_lbls, 10)
                    lc.log_weights()
                    train_out_lbl = lc.batch_classify(train_data)
                    eval_out_lbl = lc.batch_classify(eval_data)
                    train_accuracy = (train_out_lbl == train_lbls).mean()
                    eval_accuracy = (eval_out_lbl == eval_lbls).mean()
                    s = "split=" + str(split) + \
                        ", train_len=" + str(train_len) + \
                        ", eval_len=" + str(eval_len) + \
                        ", train_accuracy=" + str(train_accuracy) + \
                        ", eval_accuracy=" + str(eval_accuracy)
                    print(s, file=f)
                    print(s)


SDR_MNIST = Mnist(MachineShape([1], [], []), 0)
if os.path.exists(SDR_MNIST.file):
    SDR_MNIST.load()
else:
    SDR_MNIST.mnist = preprocess_mnist()
    SDR_MNIST.save_mnist()


def run_experiments():
    factorizations = {
        144: (12, 12),
        9: (3, 3),
        16: (4, 4),
        1: (1, 1),
        20: (5, 4)
    }
    i49 = (49, 6, 1, 1, 1, None)
    e144 = (144, 5, 1, 1, 1, None)
    c9 = (9, 1, 1, 3, 6, 'in')
    c16 = (16, 1, 1, 3, 6, 'in')
    c20 = (20, 1, 1, 3, 6, 'in')
    experiments = [
        (1, [i49, c9, e144, c9, e144, c9, e144, c9, e144, c9]),
        (1, [i49, c9, e144, c9, e144, c16, e144, c16, e144, c16, e144, c16]),
        (1, [i49, c9, e144, c9, e144, c16, e144, c16, e144, c20, e144, c20]),
    ]
    for experiment in experiments:
        first_channels, layers = experiment
        kernels, strides, channels = [], [], [first_channels]
        for layer in layers:
            channel, kernel, stride, drift, snapshots_per_sample, threshold = layer
            kernels.append(kernel)
            strides.append(stride)
            channels.append(channel)
            s = MachineShape(channels, kernels, strides)
            save_file = s.save_file(len(kernels)) + " data.pickle"
            if not os.path.exists(save_file):
                w, h = factorizations[channel]
                m = SingleColumnMachine(s, w, h, threshold=threshold)
                m.train(save=True, plot=True, snapshots_per_sample=snapshots_per_sample, drift=[drift, drift])
                m = FullColumnMachine(s)
                m.eval_with_classifier_head()

run_experiments()