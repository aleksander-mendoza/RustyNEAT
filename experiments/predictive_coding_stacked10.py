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
L2 = True
METRIC_STR = "l2_new" if L2 else "l1"
SAMPLES = 60000
DIR = 'predictive_coding_stacked8/' + METRIC_STR + "/" + str(SAMPLES)
ENTROPY_MAXIMISATION = True
MACHINE_TYPE = ecc.CpuEccMachineL2 if L2 else ecc.CpuEccMachine
DENSE_TYPE = ecc.CpuEccDenseL2 if L2 else ecc.CpuEccDense
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
        mnist.mnist = mnist.mnist.subdataset(0, SAMPLES)
        compound_stride, compound_kernel = self.machine_shape.composed_conv(idx)
        compound_stride, compound_kernel = compound_stride[:2], compound_kernel[:2]
        test_patch_indices = mnist.mnist.gen_rand_conv_subregion_indices_with_ecc(layer, test_patches)
        test_input_patches = mnist.mnist.conv_subregion_indices_with_ecc(layer, test_patch_indices)
        test_img_patches = SDR_MNIST.mnist.subdataset(0, SAMPLES).conv_subregion_indices_with_ker(compound_kernel,
                                                                                                  compound_stride,
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
            mnist.mnist.train_with_patches(number_of_samples=interval, drift=drift,
                                           patches_per_sample=snapshots_per_sample,
                                           ecc=layer, log=log,
                                           decrement_activities=ENTROPY_MAXIMISATION)
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
        self.m = MACHINE_TYPE()
        bottom = machine_shape.load_layer(0)
        out_size = htm.conv_out_size([28, 28], bottom.stride, bottom.kernel)[:2]
        bottom = bottom.repeat_column(out_size)
        self.m.push(bottom)
        for idx in range(1, len(machine_shape)):
            self.m.push_repeated_column(machine_shape.load_layer(idx))

    def eval_with_classifier_head(self, overwrite_data=False, overwrite_benchmarks=False, epochs=4):
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

            for split in SPLITS:  # [0.1, 0.2, 0.5, 0.8, 0.9]:
                if type(split) is float:
                    train_len = int(len(MNIST) * split)
                else:
                    train_len = split
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

    def eval_with_naive_bayes(self, overwrite_data=False, overwrite_benchmarks=False, min_deviation_from_mean=None):
        idx = len(self.machine_shape) - 1
        print("PATCH_SIZE=", self.m.in_shape)
        layer = self.m.get_layer(idx)
        i = "I" if min_deviation_from_mean is not None else ""
        benchmarks_save = self.machine_shape.save_file(idx + 1) + " accuracy" + i + ".txt"
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
                for split in SPLITS:
                    train_len = int(len(MNIST) * split) if type(split) is float else split
                    eval_len = len(MNIST) - train_len
                    train_data = out_mnist.mnist.subdataset(0, train_len)
                    eval_data = out_mnist.mnist.subdataset(train_len)
                    train_lbls = LABELS[0:train_len].numpy()
                    eval_lbls = LABELS[train_len:].numpy()
                    lc = train_data.fit_naive_bayes(train_lbls, 10,
                                                    invariant_to_column=min_deviation_from_mean is not None)
                    lc.clear_class_prob()
                    train_out_lbl = lc.batch_classify(train_data, min_deviation_from_mean)
                    eval_out_lbl = lc.batch_classify(eval_data, min_deviation_from_mean)
                    train_accuracy = (train_out_lbl == train_lbls).mean()
                    eval_accuracy = (eval_out_lbl == eval_lbls).mean()
                    s = "split=" + str(split) + \
                        ", train_len=" + str(train_len) + \
                        ", eval_len=" + str(eval_len) + \
                        ", train_accuracy=" + str(train_accuracy) + \
                        ", eval_accuracy=" + str(eval_accuracy)
                    print(s, file=f)
                    print(s)


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
        (1, [e(49, 6), e(100, 6), e(144, 6), e(256, 6), e(256, 6)]),
        # (1, [i49, c9(3), e144, c9(5), e144, c9(7), e144, c9(10), e144, c9(7)]),
        # (1, [i49, c9(3), e144, c9(5), e144, c16(7), e144, c16(10), e144, c16(7), e144, c16(3)]),
        # (1, [i49, c9(3), e144, c9(5), e144, c16(7), e200, c16(10), e200, c20(7), e200, c20(3)]),
        # (1, [i49, c9(3), e144, c9(5), e144, c16(7), e200, c20(10), e256, c25(7), e200, c20(3)]),
    ]
    overwrite_benchmarks = False
    overwrite_data = False
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
                m.train(save=True, plot=True, snapshots_per_sample=snapshots_per_sample)
                m = FullColumnMachine(s)
                print(save_file)
                m.eval_with_naive_bayes(overwrite_data=overwrite_data,
                                        overwrite_benchmarks=overwrite_benchmarks)
                print(save_file)
                m.eval_with_classifier_head(overwrite_benchmarks=overwrite_benchmarks)
                print(save_file)
                m.eval_with_naive_bayes(min_deviation_from_mean=0.01,
                                        overwrite_benchmarks=overwrite_benchmarks)
                print(save_file)


def parse_benchmarks(file_name, splits=[0.1, 0.8]):
    if not file_name.startswith('predictive_coding_stacked8/'):
        file_name = DIR + '/' + file_name
    with open(file_name, "r") as f:
        eval_accuracies = [0.] * len(splits)
        train_accuracies = [0.] * len(splits)
        for line in f:
            attributes = line.split(",")
            attributes = [attr.split("=") for attr in attributes]
            attributes = {key.strip(): value for key, value in attributes}
            split_val = float(attributes["split"])
            if split_val in splits:
                i = splits.index(split_val)
                eval_accuracies[i] = max(eval_accuracies[i], float(attributes["eval_accuracy"]))
                train_accuracies[i] = max(train_accuracies[i], float(attributes["train_accuracy"]))
        return eval_accuracies + train_accuracies


EXTRACT_K_S = re.compile("k([0-9]+)s([0-9]+)c([0-9]+)(k[0-9]+)?d([0-9]+)")
HAS_DRIFT = re.compile("d[2-9][0-9]*")
TABLE_MODE = 'csv'  # alternatives: latex, csv
if TABLE_MODE == 'latex':
    TABLE_FIELD_SEP = ' & '
    TABLE_ROW_SEP = ' \\\\\n\\hline\n'
elif TABLE_MODE == 'csv':
    TABLE_FIELD_SEP = ', '
    TABLE_ROW_SEP = '\n'


class ExperimentData:
    def __init__(self, experiment_name: str):
        self.leaf = True
        self.name = experiment_name
        kernels, strides, channels, drifts, ks = [], [], [], [], []
        for m in EXTRACT_K_S.finditer(experiment_name):
            k, s, c, d = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(5))
            kernels.append(k)
            strides.append(s)
            channels.append(c)
            drifts.append(d)
            k = 1 if m.group(4) is None else int(m.group(4)[1:])
            ks.append(k)
        channels.append(int(experiment_name.rsplit('c', 1)[1]))
        self.shape = MachineShape(channels, kernels, strides, drifts, ks)

        self.comp_stride, self.comp_kernel = self.shape.composed_conv(len(self.shape) - 1)
        self.out_shape = htm.conv_out_size([28, 28], self.comp_stride, self.comp_kernel)[:2]
        self.benchmarks = {
            'softmax': parse_benchmarks(self.name + " accuracy2.txt"),
            'vote': parse_benchmarks(self.name + " accuracyI.txt"),
            'naive': parse_benchmarks(self.name + " accuracy.txt"),
        }
        assert min(ks) > 0
        self.has_k = max(ks) > 1
        self.has_drift = HAS_DRIFT.search(experiment_name) is not None

    def acc(self, benchmark, split):
        return "{:.0%}".format(self.benchmark(benchmark, split, False)) + "/" + \
               "{:.0%}".format(self.benchmark(benchmark, split, True))

    def all_acc(self, split):
        return self.acc("softmax", split) + ";" + self.acc("naive", split) + ";" + self.acc("vote", split)

    def benchmark(self, benchmark, split, train):
        benchmark = self.benchmarks[benchmark]
        if split == 1:
            split = 0
        elif split == 8:
            split = 1
        else:
            raise Exception("Should be either 1 or 8")
        return benchmark[(2 if train else 0) + split]

    def format(self, db, detailed=True, metric=True):
        if detailed:
            s = self.shape
            k = str(self.comp_kernel[0]) + "x" + str(self.comp_kernel[1])
            o = str(self.out_shape[0]) + "x" + str(self.out_shape[1])
            if metric:
                codename = [METRIC_STR]
            else:
                codename = ["Yes" if self.has_drift else "No"]
            codename = codename + \
                       ["k" + str(k) + "c" + str(c) + ("/" + str(k_) if k_ > 1 else "") + "d" + str(d) for
                        k, c, d, k_ in
                        zip(s.kernels, s.channels[1:], s.drifts, self.shape.k)]
            acc8 = [k]
            acc1 = [o]

            for i in range(1, len(self.shape)):
                prev_ex = db[self.shape.code_name(i)]
                acc8.append(prev_ex.all_acc(8))
                acc1.append(prev_ex.all_acc(1))
            acc8.append(self.all_acc(8))
            acc1.append(self.all_acc(1))
            assert len(codename) == len(acc8) == len(acc1)
            return TABLE_ROW_SEP.join(
                [TABLE_FIELD_SEP.join(codename), TABLE_FIELD_SEP.join(acc8), TABLE_FIELD_SEP.join(acc1)])
        else:
            k = str(self.comp_kernel[0]) + "x" + str(self.comp_kernel[1])
            o = str(self.out_shape[0]) + "x" + str(self.out_shape[1])
            acc8 = "{:.2f}/{:.2f}".format(self.benchmark("softmax", 8, False), self.benchmark("naive", 8, False))
            acc1 = "{:.2f}/{:.2f}".format(self.benchmark("softmax", 1, False), self.benchmark("naive", 1, False))
            s = self.shape
            prev_softmax8 = [db[self.shape.code_name(i)].benchmark("softmax", 8, False) for i in
                             range(1, len(self.shape))]
            prev_softmax8.append(self.benchmark("softmax", 8, False))
            path = ' '.join(["k" + str(k) + "c" + str(c) + "d" + str(d) + "({:.2f})".format(s8)
                             for k, c, d, s8
                             in zip(s.kernels, s.channels[1:], s.drifts, prev_softmax8)])
            return ', '.join([acc8, acc1, k, o, path])

    def experiment(self, benchmark, overwrite_benchmarks=False):
        if benchmark == "vote":
            save_file = self.name + " accuracyI.txt"
            if not os.path.exists(save_file):
                m = FullColumnMachine(self.shape)
                print(save_file)
                m.eval_with_naive_bayes(min_deviation_from_mean=0.01, overwrite_benchmarks=overwrite_benchmarks)
        elif benchmark == "softmax":
            save_file = self.name + " accuracy2.txt"
            m = FullColumnMachine(self.shape)
            print(save_file)
            m.eval_with_classifier_head(epochs=4, overwrite_benchmarks=overwrite_benchmarks)


class ExperimentDB:

    def __init__(self):
        s = " accuracy2.txt"
        self.experiments = [e[:-len(s)] for e in os.listdir(DIR + '/') if e.endswith(s)]
        self.experiment_data = {n: ExperimentData(n) for n in self.experiments}
        for ex in self.experiment_data.values():
            s: MachineShape = ex.shape
            parent = self.experiment_data.get(s.parent_code_name())
            if parent is not None:
                parent.leaf = False

    def print_accuracy2_results(self, depth, with_drift=None, with_k=None):
        scores = []
        for experiment in self.experiment_data.values():
            if with_drift is not None and with_drift != experiment.has_drift:
                continue
            if type(depth) is list:
                if abs(experiment.comp_kernel[0] - depth[0]) > depth[1]:
                    continue
            elif type(depth) is int:
                if len(experiment.shape) == depth:
                    continue
            if with_k is not None and with_k != experiment.has_k:
                continue
            scores.append(experiment)
        scores.sort(key=lambda x: x.benchmark('softmax', 8, False))
        print("Depth =", depth, ",  with_drift =", with_drift, ",  with_k =", with_k)
        for exp_data in scores:
            print(exp_data.format(self.experiment_data), end=TABLE_ROW_SEP)

    def experiment_on_all(self, mode, overwrite_benchmarks=False):
        for e in self.experiment_data.values():
            e.experiment(mode, overwrite_benchmarks=overwrite_benchmarks)


def print_comparison_across_sample_sizes():
    ss = [20, 100, 1000, 6000, 12000, 60000]
    files = {
        "k6s1c1d1_c49": "k6c49",
        "k6s1c1d1_k6s1c49d1_c100": "k6c100",
        "k6s1c1d1_k6s1c49d1_k6s1c100d1_c144": "k6c144",
        "k6s1c1d1_k6s1c49d1_k6s1c100d1_k6s1c144d1_c256": "k6c256",
        "k6s1c1d1_k6s1c49d1_k6s1c100d1_k6s1c144d1_k6s1c256d1_c256": "k6c256",
    }
    results = {k: {s: [] for s in ss} for k in files.keys()}
    for s in ss:
        d = 'predictive_coding_stacked8/' + str(s)
        suff = " accuracy2.txt"
        for f in files:
            splits = [20,100,1000,0.1, 0.9]
            b = parse_benchmarks(d + '/' + f + suff, splits=splits)
            b = " ".join(["{:.0%}/{:.0%}".format(e,t) for e,t in zip(b[:len(splits)], b[len(splits):])])
            results[f][s] = b
    results = [(k, v) for k, v in results.items()]
    results.sort(key=lambda a: a[0])
    print(TABLE_FIELD_SEP.join(["%"] + [files[k] for k, _ in results]), end=TABLE_ROW_SEP)
    for s in ss:
        print(s, end=TABLE_FIELD_SEP)
        print(TABLE_FIELD_SEP.join([result[s] for _, result in results]), end=TABLE_ROW_SEP)


# print_comparison_across_sample_sizes()

run_experiments()
# edb = ExperimentDB()
# edb.experiment_on_all("softmax", overwrite_benchmarks=True)
# edb.compare_metrics()
# edb.print_accuracy2_results([26, 2], with_drift=False, with_k=False)
# edb.print_accuracy2_results([26, 2], with_drift=False, with_k=False)
