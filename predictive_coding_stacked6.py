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

SAVE = True
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


def visualise_connection_heatmap(in_w, in_h, ecc_net, out_w, out_h):
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
    plt.show()


class Experiment:

    def __init__(self, w, h, experiment_id):
        self.w = w
        self.h = h
        self.input_shape = None
        self.iterations = 1000000
        self.interval = 100000
        self.test_patches = 20000
        self.save_file = "predictive_coding_stacked6_" + str(experiment_id)
        self.num_of_snapshots = 1
        self.drift = np.array([0, 0])

    def save(self):
        pass

    def run(self, sdr, learn, update_activity):
        pass

    def take_action(self):
        pass

    def sums(self, output_sdr):
        pass

    def experiment(self):
        patch_size = self.input_shape[:2]
        print("PATCH_SIZE=", self.input_shape)
        fig, axs = plt.subplots(self.w, self.h)
        for a in axs:
            for b in a:
                b.set_axis_off()
        enc = htm.EncoderBuilder()
        img_w, img_h, img_c = self.input_shape
        img_encoder = enc.add_image(img_w, img_h, img_c, 0.8)

        def normalise_img(img):
            sdr = htm.CpuSDR()
            img_encoder.encode(sdr, img.unsqueeze(2).numpy())
            return img, sdr

        test_patches = [normalise_img(rand_patch(patch_size)) for _ in range(self.test_patches)]
        test_patches = [test_patch for test_patch in test_patches if test_patch[0].sum() > 2]
        all_processed = []
        all_total_sum = []
        for s in tqdm(range(self.iterations), desc="training"):
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
                self.run(sdr, learn=True, update_activity=True)
            self.take_action()
            if s % self.interval == 0:
                if SAVE:
                    self.save()
                stats = torch.zeros([self.input_shape[0], self.input_shape[1], self.w * self.h])
                processed = 0
                total_sum = 0
                for img, sdr in tqdm(test_patches, desc="eval"):
                    output_sdr = self.run(sdr, learn=False, update_activity=False)
                    self.take_action()
                    for top in output_sdr:
                        stats[:, :, top] += img
                        processed += 1
                    total_sum += self.sums(output_sdr)
                all_processed.append(processed / len(test_patches))
                all_total_sum.append(total_sum)
                print("processed=", all_processed)
                print("sums=", all_total_sum)
                for i in range(self.w):
                    for j in range(self.h):
                        axs[i, j].imshow(stats[:, :, i + j * self.w])
                plt.pause(0.01)
                if SAVE:
                    img_file_name = self.save_file + " before.png"
                    if s > 0 or os.path.exists(img_file_name):
                        img_file_name = self.save_file + " after.png"
                    plt.savefig(img_file_name)
        plt.show()

    def eval_with_classifier_head(self):
        print("PATCH_SIZE=", self.input_shape)

        enc = htm.EncoderBuilder()
        img_w, img_h, img_c = self.input_shape
        enc_img = enc.add_image(img_w, img_h, img_c, 0.8)

        class D(torch.utils.data.Dataset):
            def __init__(self, imgs, lbls, ecc_net):
                self.imgs = imgs
                self.lbls = lbls
                self.ecc_net = ecc_net

            def __len__(self):
                return len(self.imgs)

            def __getitem__(self, idx):
                img = self.imgs[idx]
                sdr = htm.CpuSDR()
                enc_img.encode(sdr, img.unsqueeze(2).numpy())
                img_bits = self.ecc_net.run(sdr, learn=False, update_activity=False)
                img = torch.zeros(m.out_volume)
                img[list(img_bits)] = 1
                return img, self.lbls[idx]

        train_mnist = MNIST[:40000]
        train_labels = LABELS[:40000]
        train_d = D(train_mnist, train_labels)

        eval_mnist = MNIST[40000:60000]
        eval_labels = LABELS[40000:60000]
        eval_d = D(eval_mnist, eval_labels)

        linear = torch.nn.Linear(m.out_volume, 10)
        loss = torch.nn.NLLLoss()
        optim = torch.optim.Adam(linear.parameters())

        train_dataloader = DataLoader(train_d, batch_size=64, shuffle=True)
        eval_dataloader = DataLoader(eval_d, batch_size=64, shuffle=True)

        for epoch in range(100):
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
            print("train=", train_accuracy / train_total, "eval=", eval_accuracy / eval_total)


class Experiment1(Experiment):
    def __init__(self):
        super().__init__(15, 15, "experiment1")
        self.layer1_save_file = self.save_file + " layer1.model"
        self.input_to_layer1_save_file = self.save_file + " input_to_layer1.model"
        if os.path.exists(self.input_to_layer1_save_file):
            self.input_to_layer1 = ecc.ConvWeights.load(self.input_to_layer1_save_file)
        else:
            self.input_to_layer1 = ecc.ConvWeights([1, 1, self.w * self.h], [18, 18], [1, 1], 1)

        if os.path.exists(self.layer1_save_file):
            self.layer1 = ecc.CpuEccPopulation.load(self.layer1_save_file)
        else:
            self.layer1 = ecc.CpuEccPopulation(self.input_to_layer1.out_shape, 1)

        self.input_shape = np.array(self.input_to_layer1.in_shape)

    def run(self, sdr, learn, update_activity):
        layer1_sdr = self.input_to_layer1.run(sdr, self.layer1, update_activity=update_activity, learn=learn)
        return layer1_sdr

    def save(self):
        self.layer1.save(self.layer1_save_file)
        self.input_to_layer1.save(self.input_to_layer1_save_file)

    def sums(self, output_sdr):
        return self.layer1.sums_for_sdr(output_sdr)


class Experiment2(Experiment):
    def __init__(self, w, h):
        super().__init__(w, h, "experiment2oc" + str(w * h))
        self.layer1_save_file = self.save_file + " layer1.model"

        if os.path.exists(self.layer1_save_file):
            self.layer1 = ecc.CpuEccDense.load(self.layer1_save_file)
        else:
            self.layer1 = ecc.CpuEccDense([1, 1], [6, 6], [1, 1], 1, w * h, 1)

        self.input_shape = np.array(self.layer1.in_shape)

    def run(self, sdr, learn, update_activity):
        if update_activity:
            layer1_sdr = self.layer1.run(sdr, learn=learn)
        else:
            layer1_sdr = self.layer1.infer(sdr, learn=learn)
        return layer1_sdr

    def save(self):
        self.layer1.save(self.layer1_save_file)

    def sums(self, output_sdr):
        return self.layer1.sums_for_sdr(output_sdr)


class Experiment3(Experiment):
    def __init__(self, kernel):
        super().__init__(15, 15, "experiment3k" + str(kernel[0]) + "x" + str(kernel[1]))
        ex2 = Experiment2(5, 5)
        self.layer2_save_file = self.save_file + " layer2.model"
        if os.path.exists(self.layer2_save_file):
            self.layer2 = ecc.CpuEccDense.load(self.layer2_save_file)
        else:
            self.layer2 = ecc.CpuEccDense([1, 1], kernel, [1, 1], ex2.layer1.out_channels, self.w * self.h, 1)
        self.layer1 = ex2.layer1.repeat_column(self.layer2.in_shape)
        self.input_shape = np.array(self.layer1.in_shape)

    def run(self, sdr, learn, update_activity):
        sdr = self.layer1.infer(sdr)
        if update_activity:
            return self.layer2.run(sdr, learn=learn)
        else:
            return self.layer2.infer(sdr, learn=learn)

    def save(self):
        self.layer2.save(self.layer2_save_file)

    def sums(self, output_sdr):
        return self.layer2.sums_for_sdr(output_sdr)


class Experiment22(Experiment):
    def __init__(self):
        super().__init__(3, 3, "experiment22")
        ex2 = Experiment2(7, 7)
        self.layer2_save_file = self.save_file + " layer2.model"

        if os.path.exists(self.layer2_save_file):
            self.layer2 = ecc.CpuEccDense.load(self.layer2_save_file)
        else:
            self.layer2 = ecc.CpuEccDense([1, 1], [1, 1], [1, 1], ex2.layer1.out_channels, self.w * self.h, 1)
            self.layer2.set_threshold(1 / self.layer2.in_channels)
        self.layer1 = ex2.layer1
        self.input_shape = ex2.input_shape
        self.sdr = None
        self.drift = np.array([3, 3])
        self.num_of_snapshots = 6

    def run(self, sdr, learn, update_activity):
        pre_sdr = self.layer1.infer(sdr)
        fin_sdr = self.layer2.infer(pre_sdr)
        if self.sdr is None:
            self.sdr = fin_sdr
        if learn:
            self.layer2.learn(pre_sdr, self.sdr)
        return self.sdr

    def take_action(self):
        self.sdr = None

    def save(self):
        # visualise_connection_heatmap(7, 7, self.layer2, 3, 3)
        self.layer2.save(self.layer2_save_file)

    def sums(self, output_sdr):
        return self.layer2.sums_for_sdr(output_sdr)


Experiment22().experiment()
