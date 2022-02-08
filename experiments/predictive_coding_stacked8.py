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

    def train(self, plot=False, save=True, log=None,
              snapshots_per_sample=1, iterations=2000000,
              interval=100000, test_patches=20000):
        idx = len(self.machine_shape) - 1
        layer = self.m.get_layer(idx)
        drift = self.machine_shape.drift(idx)
        params = self.machine_shape.load_or_save_params(
            idx,
            w=self.w,
            h=self.h,
            snapshots_per_sample=snapshots_per_sample,
            iterations=iterations,
            interval=interval,
            test_patches=test_patches,
        )
        w = params['w']
        h = params['h']
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
        if plot:
            plt.close(fig)
            # plt.show()


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

    def eval_with_classifier_head(self, overwrite_data=False, overwrite_benchmarks=False, epochs=2):
        idx = len(self.machine_shape) - 1
        print("PATCH_SIZE=", self.m.in_shape)
        layer = self.m.get_layer(idx)
        benchmarks_save = self.machine_shape.save_file(idx + 1) + " accuracy2.txt"
        out_mnist = Mnist(self.machine_shape, idx + 1)
        if os.path.exists(out_mnist.file) and not overwrite_data:
            out_mnist.load()
        else:
            in_mnist = Mnist(self.machine_shape, idx)
            in_mnist.load()
            out_mnist.mnist = in_mnist.mnist.batch_infer(layer)
            out_mnist.save_mnist()
        if not os.path.exists(benchmarks_save) or overwrite_benchmarks:

            class D(torch.utils.data.Dataset):
                def __init__(self, imgs, lbls):
                    self.imgs = imgs
                    self.lbls = lbls

                def __len__(self):
                    return len(self.imgs)

                def __getitem__(self, idx):
                    return self.imgs.to_f32_numpy(idx), self.lbls[idx]

            for split in [0.1, 0.2, 0.5, 0.8, 0.9]:
                train_len = int(len(MNIST) * split)
                eval_len = len(MNIST) - train_len
                train_data = out_mnist.mnist.subdataset(0, train_len)
                eval_data = out_mnist.mnist.subdataset(train_len)
                train_lbls = LABELS[0:train_len].numpy()
                eval_lbls = LABELS[train_len:].numpy()
                train_d = D(train_data, train_lbls)
                eval_d = D(eval_data, eval_lbls)
                linear = torch.nn.Linear(out_mnist.mnist.volume, 10)
                loss = torch.nn.NLLLoss()
                optim = torch.optim.Adam(linear.parameters())
                bs = 64
                train_dataloader = DataLoader(train_d, batch_size=bs, shuffle=True)
                eval_dataloader = DataLoader(eval_d, batch_size=bs, shuffle=True)

                for epoch in range(epochs):
                    train_accuracy = 0
                    train_total = 0
                    for x, y in tqdm(train_dataloader, desc="train"):
                        optim.zero_grad()
                        bs = x.shape[0]
                        x = x.reshape(bs, -1)
                        x = linear(x)
                        x = torch.log_softmax(x, dim=1)
                        d = loss(x, y)
                        train_accuracy += (x.argmax(1) == y).sum().item()
                        train_total += x.shape[0]
                        d.backward()
                        optim.step()

                    eval_accuracy = 0
                    eval_total = 0
                    for x, y in tqdm(eval_dataloader, desc="eval"):
                        bs = x.shape[0]
                        x = x.reshape(bs, -1)
                        x = linear(x)
                        eval_accuracy += (x.argmax(1) == y).sum().item()
                        eval_total += x.shape[0]
                    s = "split=" + str(split) + \
                        ", train_len=" + str(train_len) + \
                        ", eval_len=" + str(eval_len) + \
                        ", epoch=" + str(epoch) + \
                        ", train_accuracy=" + str(train_accuracy / train_total) + \
                        ", eval_accuracy=" + str(eval_accuracy / eval_total)
                    with open(benchmarks_save, 'a+') as f:
                        print(s, file=f)
                    print(s)

    def eval_with_naive_bayes(self, overwrite_data=False, overwrite_benchmarks=False):
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
            with open(benchmarks_save, 'a+') as f:
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


SDR_MNIST = Mnist(MachineShape([1], [], [], []), 0)
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
    i49 = (49, 6, 1, 1, 1, None)
    e144 = (144, 5, 1, 1, 1, None)
    e200 = (200, 5, 1, 1, 1, None)
    e256 = (256, 5, 1, 1, 1, None)

    def e(c, k):
        return c, k, 1, 1, 1, None

    def c(c, d):
        return c, 1, 1, d, 6, 'in'

    experiments = [
        (1, [e(49, 6), c(9, 3), e(100, 6), c(9, 5), e(144, 6), c(16, 7), e(256, 6), c(20, 8), e(256, 6)]),
        # (1, [e(100, 28)]),
        # (1, [e(256, 28)]),
        # (1, [e(400, 28)]),
        # (1, [e(6, 6), e(6, 6), e(6, 6), e(6, 6), e(6, 6)]),
        # (1, [e(8, 6), e(2 * 8, 6), e(3 * 8, 6), e(4 * 8, 6), e(4 * 8, 6)]),
        # (1, [e(8, 6), e(2 * 8, 6), e(3 * 8, 6), e(4 * 8, 6), e(5 * 8, 6)]),
        # (1, [e(8, 6), e(2 * 8, 3), e(3 * 8, 3), e(4 * 8, 3), e(4 * 8, 3), e(4 * 8, 3)]),
        # (1, [e(16, 6), e(2 * 16, 3), e(3 * 16, 3), e(4 * 16, 3), e(4 * 16, 3), e(4 * 16, 3)]),
        # (1, [e(16, 6), e(2 * 16, 6), e(3 * 16, 6), e(4 * 16, 6), e(4 * 16, 6)]),
        # (1, [e(49, 6), e(100, 6), e(144, 6), e(256, 6), e(256, 6)]),
        # (1, [i49, c9(3), e144, c9(5), e144, c9(7), e144, c9(10), e144, c9(7)]),
        # (1, [i49, c9(3), e144, c9(5), e144, c16(7), e144, c16(10), e144, c16(7), e144, c16(3)]),
        # (1, [i49, c9(3), e144, c9(5), e144, c16(7), e200, c16(10), e200, c20(7), e200, c20(3)]),
        # (1, [i49, c9(3), e144, c9(5), e144, c16(7), e200, c20(10), e256, c25(7), e200, c20(3)]),
    ]
    for experiment in experiments:
        first_channels, layers = experiment
        kernels, strides, channels, drifts = [], [], [first_channels], []
        for layer in layers:
            channel, kernel, stride, drift, snapshots_per_sample, threshold = layer
            kernels.append(kernel)
            strides.append(stride)
            channels.append(channel)
            drifts.append(drift)
            s = MachineShape(channels, kernels, strides, drifts)
            save_file = s.save_file(len(kernels)) + " data.pickle"
            if not os.path.exists(save_file):
                w, h = factorizations[channel]
                m = SingleColumnMachine(s, w, h, threshold=threshold)
                print(save_file)
                m.train(save=True, plot=True, snapshots_per_sample=snapshots_per_sample)
                m = FullColumnMachine(s)
                print(save_file)
                m.eval_with_naive_bayes()
                print(save_file)
                m.eval_with_classifier_head()
                print(save_file)


def parse_benchmarks(file_name):
    with open('predictive_coding_stacked8/' + file_name, "r") as f:
        accuracy8 = 0
        accuracy1 = 0
        for line in f:
            attributes = line.split(",")
            attributes = [attr.split("=") for attr in attributes]
            attributes = {key.strip(): value for key, value in attributes}
            split_val = float(attributes["split"])
            if split_val == 0.1:
                accuracy1 = max(accuracy1, float(attributes["eval_accuracy"]))
            elif split_val == 0.8:
                accuracy8 = max(accuracy8, float(attributes["eval_accuracy"]))
        return accuracy8, accuracy1


EXTRACT_K_S = re.compile("k([0-9]+)s([0-9]+)c([0-9]+)d([0-9]+)")
HAS_DRIFT = re.compile("d[2-9][0-9]*")


class ExperimentData:
    def __init__(self, experiment_name):
        kernels, strides, channels, drifts = [], [], [], []
        for m in EXTRACT_K_S.finditer(experiment_name):
            k, s, c, d = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            kernels.append(k)
            strides.append(s)
            channels.append(c)
            drifts.append(d)
        channels.append(int(experiment_name.rsplit('c', 1)[1]))
        self.shape = MachineShape(channels, kernels, strides, drifts)
        self.name = experiment_name
        self.comp_stride, self.comp_kernel = self.shape.composed_conv(len(self.shape) - 1)
        self.out_shape = htm.conv_out_size([28, 28], self.comp_stride, self.comp_kernel)[:2]
        self.softmax8, self.softmax1 = parse_benchmarks(experiment_name + " accuracy2.txt")
        self.naive8, self.naive1 = parse_benchmarks(experiment_name + " accuracy.txt")
        self.has_drift = HAS_DRIFT.search(experiment_name) is not None

    def format(self, db):
        k = str(self.comp_kernel[0]) + "x" + str(self.comp_kernel[1])
        o = str(self.out_shape[0]) + "x" + str(self.out_shape[1])
        acc8 = "{:.2f}".format(self.softmax8) + "/" + "{:.2f}".format(self.naive8)
        acc1 = "{:.2f}".format(self.softmax1) + "/" + "{:.2f}".format(self.naive1)
        s = self.shape
        prev_softmax8 = [db[self.shape.code_name(i)].softmax8 for i in range(1, len(self.shape))]
        prev_softmax8.append(self.softmax8)
        path = ' '.join(["k" + str(k) + "c" + str(c) + "d" + str(d) + "({:.2f})".format(s8)
                         for k, c, d, s8
                         in zip(s.kernels, s.channels[1:], s.drifts, prev_softmax8)])
        return ', '.join([acc8, acc1, k, o, path])


class ExperimentDB:

    def __init__(self):
        s = " accuracy2.txt"
        self.experiments = [e[:-len(s)] for e in os.listdir('predictive_coding_stacked8/') if e.endswith(s)]
        self.experiment_data = {n: ExperimentData(n) for n in self.experiments}

    def print_accuracy2_results(self, depth, with_drift=None):

        depth_pat = "*" if type(depth) == list else "{" + str(depth) + "}"
        if with_drift is None or with_drift is True:
            pat = re.compile("(k[0-9]+s[0-9]+c[0-9]+d[0-9]+_)" + depth_pat + "c[0-9]+")
        elif with_drift is False:
            pat = re.compile("(k[0-9]+s[0-9]+c[0-9]+d1_)" + depth_pat + "c[0-9]+")
        else:
            raise Exception("Invalid drift")
        scores = []

        for experiment in self.experiment_data.values():
            if pat.fullmatch(experiment.name):
                if with_drift is True and not experiment.has_drift:
                    continue

                if type(depth) is list:
                    if abs(experiment.comp_kernel[0] - depth[0]) > depth[1]:
                        continue
                scores.append(experiment)
        scores.sort(key=lambda x: x.softmax8)
        print("Depth =", depth, ",  with_drift =", with_drift)
        for exp_data in scores:
            print(exp_data.format(self.experiment_data))


# run_experiments()
# for d in range(16):
edb = ExperimentDB()
edb.print_accuracy2_results([26, 2], with_drift=True)
edb.print_accuracy2_results([26, 2], with_drift=False)
#     print_accuracy2_results(d, with_drift=False)
