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

from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader


MNIST, LABELS = torch.load('htm/data/mnist.pt')


def rand_patch(patch_size, img_idx=None):
    if img_idx is None:
        img_idx = int(np.random.rand() * len(MNIST))
    img = MNIST[img_idx]
    r = np.random.rand(2)
    left_bottom = (img.shape - patch_size) * r
    left_bottom = left_bottom.astype(int)
    top_right = left_bottom + patch_size
    return img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]]


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


class Experiment:

    def __init__(self, w, h, channels, kernels, strides, threshold=None):
        assert len(channels) == len(kernels) == len(strides)
        self.w = w
        self.h = h
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.channels.append(w * h)
        self.iterations = 1000000
        self.interval = 100000
        self.test_patches = 20000
        self.num_of_snapshots = 1
        self.drift = np.array([0, 0])
        self.plot = True
        self.sdr = None
        self.m = None
        self.head = None
        self.in_shape = None
        self.out_shape = None
        self.in_grid = None
        self.threshold = threshold
        self.epochs = 5

    def save_file(self, idx=-1):
        if idx < 0:
            idx = len(self) + idx
        prefix = "predictive_coding_stacked7_"
        path = ''.join(["k" + str(k) + "s" + str(s) + "c" + str(c) + "_" for k, c, s in
                        zip(self.kernels[:idx + 1], self.channels[:idx + 1], self.strides[:idx + 1])])
        return prefix + path + "o" + str(self.channels[idx + 1])

    def load_layer(self, idx=-1):
        mf = self.save_file(idx) + ".model"
        if os.path.exists(mf):
            return ecc.CpuEccDense.load(mf)
        else:
            return None

    def kernel(self, idx):
        return [self.kernels[idx], self.kernels[idx]]

    def stride(self, idx):
        return [self.strides[idx], self.strides[idx]]

    def __len__(self):
        return len(self.kernels)

    def make_single_column_machine(self):
        self.m = ecc.CpuEccMachine()
        top = self.load_layer(-1)
        if top is None:
            l = ecc.CpuEccDense([1, 1], self.kernel(-1), self.stride(-1),
                                in_channels=self.channels[-2],
                                out_channels=self.channels[-1],
                                k=1)
            if self.threshold == 'in':
                l.threshold = 1 / l.in_channels
            elif type(self.threshold) is float:
                l.threshold = self.threshold
            self.m.push(l)
        else:
            self.m.push(top)
        for idx in reversed(range(len(self) - 1)):
            self.m.prepend_repeated_column(self.load_layer(idx))
        self.update_shapes()

    def update_shapes(self):
        self.in_shape = np.array(self.m.in_shape)
        self.out_shape = np.array(self.m.out_shape)
        self.in_grid = np.array(self.m.in_grid)

    def make_full_column_machine(self):
        self.m = ecc.CpuEccMachine()
        bottom = self.load_layer(0)
        out_size = htm.conv_out_size([28, 28], bottom.stride, bottom.kernel)[:2]
        bottom = bottom.repeat_column(out_size)
        self.m.push(bottom)
        for idx in range(1, len(self)):
            self.m.push_repeated_column(self.load_layer(idx))
        self.update_shapes()

    def save(self, idx=-1):
        if idx == -1 and self.head is not None:
            self.head.save(self.save_file(idx) + ".model")
        else:
            if idx < 0:
                idx = len(self) + idx
            self.m.save_layer(idx, self.save_file(idx) + ".model")

    def sums(self, output_sdr):
        if self.head is None:
            return self.m.sums_for_sdr(len(self.kernels) - 1, output_sdr)
        else:
            return self.head.sums_for_sdr(output_sdr)

    def experiment(self):
        self.make_single_column_machine()
        self.head = self.m.pop()
        with open(self.save_file() + " params.txt", "w+") as f:
            print({
                'drift': self.drift,
                'num_of_snapshots': self.num_of_snapshots,
                'w:': self.w,
                'h:': self.h,
                'test_patches': self.test_patches,
                'iterations': self.iterations
            }, file=f)
        patch_size = np.array(self.in_grid)
        print("PATCH_SIZE=", self.in_shape)
        if self.plot:
            fig, axs = plt.subplots(self.w, self.h)
            for a in axs:
                for b in a:
                    b.set_axis_off()
        enc = htm.EncoderBuilder()
        img_w, img_h, img_c = self.in_shape
        img_encoder = enc.add_image(img_w, img_h, img_c, 0.8)

        def normalise_img(img):
            sdr = htm.CpuSDR()
            img_encoder.encode(sdr, img.unsqueeze(2).numpy())
            return img, sdr

        test_patches = [normalise_img(rand_patch(patch_size)) for _ in range(self.test_patches)]
        test_patches = [test_patch for test_patch in test_patches if len(test_patch[1]) > 2]
        all_processed = []
        all_total_sum = []
        stats_shape = list(self.in_grid) + [self.w * self.h]
        for s in tqdm(range(self.iterations), desc="training"):
            img_idx = int(np.random.rand() * len(MNIST))
            img = MNIST[img_idx]
            r = np.random.rand(2)
            min_left_bottom = (img.shape - patch_size - self.drift) * r
            min_left_bottom = min_left_bottom.astype(int)
            self.sdr = None
            for _ in range(self.num_of_snapshots):
                left_bottom = min_left_bottom + (np.random.rand(2) * (self.drift + 1)).astype(int)
                assert np.all(left_bottom >= 0)
                assert np.all(left_bottom <= img.shape - patch_size), left_bottom
                top_right = left_bottom + patch_size
                snapshot = img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]]
                snapshot, sdr = normalise_img(snapshot)
                self.m.infer(sdr)
                pre_sdr = self.m.last_output_sdr()
                fin_sdr = self.head.infer(pre_sdr)
                if self.sdr is None or len(self.sdr) == 0:
                    self.sdr = fin_sdr
                    self.head.decrement_activities(fin_sdr)
                self.head.learn(pre_sdr, self.sdr)
            if s % self.interval == 0:
                if SAVE:
                    self.save()
                stats = torch.zeros(stats_shape)
                processed = 0
                total_sum = 0
                for img, sdr in tqdm(test_patches, desc="eval"):
                    self.m.infer(sdr)
                    pre_sdr = self.m.last_output_sdr()
                    output_sdr = self.head.infer(pre_sdr)
                    for top in output_sdr:
                        stats[:, :, top] += img
                        processed += 1
                    total_sum += self.sums(output_sdr)
                all_processed.append(processed / len(test_patches))
                all_total_sum.append(total_sum)
                print("processed=", all_processed)
                print("sums=", all_total_sum)
                if self.plot:
                    for i in range(self.w):
                        for j in range(self.h):
                            axs[i, j].imshow(stats[:, :, i + j * self.w])
                    plt.pause(0.01)
                    if SAVE:
                        img_file_name = self.save_file() + " before.png"
                        if s > 0 or os.path.exists(img_file_name):
                            img_file_name = self.save_file() + " after.png"
                        plt.savefig(img_file_name)
        if self.plot:
            plt.show()

    def measure_cross_correlation(self):
        save = self.save_file() + " conf_mat.pt"
        self.make_single_column_machine()
        if os.path.exists(save):
            with open(save, "rb") as f:
                confusion_matrix = torch.load(f)
        else:
            patch_size = self.in_shape[:2]
            print("PATCH_SIZE=", self.in_shape)
            enc = htm.EncoderBuilder()
            img_w, img_h, img_c = self.in_shape
            img_encoder = enc.add_image(img_w, img_h, img_c, 0.8)

            def normalise_img(img):
                sdr = htm.CpuSDR()
                img_encoder.encode(sdr, img.unsqueeze(2).numpy())
                return img, sdr

            curr = set()
            confusion_matrix = torch.zeros([self.w * self.h, self.w * self.h])
            for s in tqdm(range(self.test_patches), desc="measuring"):
                img_idx = int(np.random.rand() * len(MNIST))
                img = MNIST[img_idx]
                r = np.random.rand(2)
                min_left_bottom = (img.shape - patch_size - self.drift) * r
                min_left_bottom = min_left_bottom.astype(int)
                for _ in range(self.num_of_snapshots):
                    left_bottom = min_left_bottom + (np.random.rand(2) * (self.drift + 1)).astype(int)
                    assert np.all(left_bottom >= 0)
                    assert np.all(left_bottom < img.shape - patch_size), left_bottom
                    top_right = left_bottom + patch_size
                    snapshot = img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]]
                    snapshot, sdr = normalise_img(snapshot)
                    self.m.infer(sdr)
                    out_sdr = self.m.last_output_sdr()
                    if len(out_sdr) > 0:
                        curr.add(out_sdr.item())
                curr_list = list(curr)
                for i in range(len(curr_list)):
                    for j in range(i + 1, len(curr_list)):
                        confusion_matrix[curr_list[i], curr_list[j]] += 1
                        confusion_matrix[curr_list[j], curr_list[i]] += 1
                curr.clear()

            with open(save, "wb+") as f:
                torch.save(confusion_matrix, f)
        if self.plot:
            plt.imshow(confusion_matrix)
            plt.show()
            fig, axs = plt.subplots(self.w, self.h)
            for a in axs:
                for b in a:
                    b.set_axis_off()
            for i in range(self.w):
                for j in range(self.h):
                    img = confusion_matrix[i + j * self.w]
                    img = img.reshape([self.w, self.h])
                    img = img.T
                    axs[i, j].imshow(img)
            plt.savefig(self.save_file() + " conf_mat.png")
            plt.show()
        return confusion_matrix

    def eval_with_classifier_head(self):
        self.make_full_column_machine()
        print("PATCH_SIZE=", self.in_shape)
        benchmarks_save = self.save_file() + " accuracy.txt"
        enc = htm.EncoderBuilder()
        img_w, img_h, img_c = self.in_shape
        enc_img = enc.add_image(img_w, img_h, img_c, 0.8)
        out_volume = self.out_shape.prod()

        class D(torch.utils.data.Dataset):
            def __init__(self, imgs, lbls, m):
                self.imgs = imgs
                self.lbls = lbls
                self.m = m

            def __len__(self):
                return len(self.imgs)

            def __getitem__(self, idx):
                img = self.imgs[idx]
                sdr = htm.CpuSDR()
                enc_img.encode(sdr, img.unsqueeze(2).numpy())
                self.m.infer(sdr)
                img_bits = self.m.last_output_sdr()
                img = torch.zeros(out_volume)
                img[list(img_bits)] = 1
                return img, self.lbls[idx]

        train_mnist = MNIST[:40000]
        train_labels = LABELS[:40000]
        train_d = D(train_mnist, train_labels, self.m)

        eval_mnist = MNIST[40000:60000]
        eval_labels = LABELS[40000:60000]
        eval_d = D(eval_mnist, eval_labels, self.m)

        linear = torch.nn.Linear(out_volume, 10)
        loss = torch.nn.NLLLoss()
        optim = torch.optim.Adam(linear.parameters())

        train_dataloader = DataLoader(train_d, batch_size=64, shuffle=True)
        eval_dataloader = DataLoader(eval_d, batch_size=64, shuffle=True)

        for epoch in range(self.epochs):
            train_accuracy = 0
            train_total = 0
            for x, y in tqdm(train_dataloader, desc="train"):
                optim.zero_grad()
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
                x = linear(x)
                eval_accuracy += (x.argmax(1) == y).sum().item()
                eval_total += x.shape[0]
            with open(benchmarks_save, "a+") as f:
                print("epoch=", epoch, "train=", train_accuracy / train_total, "eval=", eval_accuracy / eval_total,
                      file=f)
            print("train=", train_accuracy / train_total, "eval=", eval_accuracy / eval_total)


SAVE = True
e = Experiment(4, 4, [1, 49, 9, 144, 9, 144], [6, 1, 5, 1, 5, 1], [1, 1, 1, 1, 1, 1])
e.threshold = 0.0000001
e.plot = False
e.num_of_snapshots = 6
e.drift = np.array([8, 8])
e.epochs = 1
e.experiment()
e.eval_with_classifier_head()

# 
# e = Experiment(4, 4, [1, 49, 9, 144, 9, 144], [6, 1, 5, 1, 5, 1], [1, 1, 1, 1, 1, 1])
# e.threshold = 0.0000001
# e.plot = False
# e.num_of_snapshots = 6
# e.drift = np.array([8, 8])
# e.epochs = 1
# e.experiment()
# e.eval_with_classifier_head()
