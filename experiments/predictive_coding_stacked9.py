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
import numpy as np
import json
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader

fig, axs = None, None


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


class MachineShape:

    def __init__(self, channels, kernels, strides, drifts):
        assert len(channels) == len(kernels) + 1 == len(strides) + 1 == len(drifts) + 1
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.drifts = drifts

    def composed_conv(self, idx):
        strides = [[s, s] for s in self.strides]
        kernels = [[k, k] for k in self.kernels]
        return htm.conv_compose_array(strides=strides[:idx + 1], kernels=kernels[:idx + 1])

    def code_name(self, idx):
        path = ''.join(["k" + str(k) + "s" + str(s) + "c" + str(c) + "d" + str(d) + "_" for k, c, s, d in
                        zip(self.kernels[:idx], self.channels[:idx], self.strides[:idx], self.drifts[:idx])])
        return path + "c" + str(self.channels[idx])

    def save_file(self, idx):
        return "predictive_coding_stacked8/" + self.code_name(idx)

    def kernel(self, idx):
        return [self.kernels[idx], self.kernels[idx]]

    def stride(self, idx):
        return [self.strides[idx], self.strides[idx]]

    def drift(self, idx):
        return [self.drifts[idx], self.drifts[idx]]

    def __len__(self):
        return len(self.kernels)

    def load_model(self, idx):
        mf = self.save_file(idx + 1) + " machine.model"
        if os.path.exists(mf):
            return ecc.CpuEccMachine.load(mf)
        else:
            return None

    def load_or_save_params(self, idx, **kwrds):
        f = self.save_file(idx + 1) + " params2.txt"
        if os.path.exists(f):
            with open(f, "r") as f:
                d2 = json.load(f)
                kwrds.update(d2)
        else:
            with open(f, "w+") as f:
                json.dump(kwrds, f)
        return kwrds


class SingleColumnMachine:

    def __init__(self, machine_shape: MachineShape, w, h):
        assert w * h == machine_shape.channels[-1]
        self.machine_shape = machine_shape
        self.m = self.machine_shape.load_model(len(machine_shape) - 1)
        self.w = w
        self.h = h
        if self.m is None:
            kernels = [[k, k] for k in self.machine_shape.kernels]
            strides = [[k, k] for k in self.machine_shape.strides]
            self.m = ecc.CpuEccMachine([1, 1],
                                       kernels,
                                       strides,
                                       self.machine_shape.channels,
                                       [1] * len(kernels))

    def train(self, plot=False, save=True, log=None, iterations=2000000,
              interval=100000, test_patches=20000):
        idx = len(self.machine_shape) - 1
        params = self.machine_shape.load_or_save_params(
            idx,
            w=self.w,
            h=self.h,
            iterations=iterations,
            interval=interval,
            test_patches=test_patches,
        )
        w = params['w']
        h = params['h']
        iterations = params['iterations']
        interval = params['interval']
        test_patches = params['test_patches']
        test_input_patches = SDR_MNIST.gen_rand_2d_patches(self.m.in_grid, test_patches)
        # test_img_patches = SDR_MNIST.mnist.conv_subregion_indices_with_ker(compound_kernel, compound_stride,
        #                                                                    test_patch_indices)
        print("PATCH_SIZE=", self.m.in_shape)
        all_missed = []
        all_total_sum = []
        if plot:
            fig, axs = plt.subplots(self.w, self.h)
            for a in axs:
                for b in a:
                    b.set_axis_off()

        def eval_ecc():
            test_outputs = test_input_patches.machine_infer(self.m)
            if plot:
                receptive_fields = test_input_patches.measure_receptive_fields(test_outputs)
                receptive_fields = receptive_fields.squeeze(2)
                for i in range(w):
                    for j in range(h):
                        axs[i, j].imshow(receptive_fields[:, :, i + j * w])
                plt.pause(0.01)
                if save:
                    img_file_name = self.machine_shape.save_file(idx + 1) + " machine before.png"
                    if s > 0 or os.path.exists(img_file_name):
                        img_file_name = self.machine_shape.save_file(idx + 1) + " machine after.png"
                    plt.savefig(img_file_name)

        for s in tqdm(range(int(math.ceil(iterations / interval))), desc="training"):
            eval_ecc()
            SDR_MNIST.train_machine_with_patches(interval, self.m, log)
            if save:
                self.m.save(self.machine_shape.save_file(idx + 1) + " machine.model")
        eval_ecc()
        if plot:
            plt.close(fig)
            # plt.show()


SDR_MNIST_FILE = 'predictive_coding_stacked8/c1 data.pickle'
if os.path.exists(SDR_MNIST_FILE):
    SDR_MNIST = htm.CpuSdrDataset.load(SDR_MNIST_FILE)

SingleColumnMachine(MachineShape([1, 20], [5], [2], [1]), 5, 4).train(plot=True)
