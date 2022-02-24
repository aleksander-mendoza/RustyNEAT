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

    def __init__(self, w, h, experiment_id):
        self.w = w
        self.h = h
        self.input_shape = None
        self.output_shape = None
        self.iterations = 1000000
        self.interval = 100000
        self.test_patches = 20000
        self.save_file = "predictive_coding_stacked6_" + str(experiment_id)
        self.num_of_snapshots = 1
        self.drift = np.array([0, 0])
        self.plot = True

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
        if self.plot:
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
        test_patches = [test_patch for test_patch in test_patches if len(test_patch[1]) > 2]
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
                if self.plot:
                    for i in range(self.w):
                        for j in range(self.h):
                            axs[i, j].imshow(stats[:, :, i + j * self.w])
                    plt.pause(0.01)
                    if SAVE:
                        img_file_name = self.save_file + " before.png"
                        if s > 0 or os.path.exists(img_file_name):
                            img_file_name = self.save_file + " after.png"
                        plt.savefig(img_file_name)
        if self.plot:
            plt.show()

    def measure_cross_correlation(self):
        save = self.save_file + " conf_mat.pt"
        if os.path.exists(save):
            with open(save, "rb") as f:
                confusion_matrix = torch.load(f)
        else:
            patch_size = self.input_shape[:2]
            print("PATCH_SIZE=", self.input_shape)
            enc = htm.EncoderBuilder()
            img_w, img_h, img_c = self.input_shape
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
                    out_sdr = self.run(sdr, learn=True, update_activity=True)
                    if len(out_sdr) > 0:
                        curr.add(out_sdr.item())
                self.take_action()
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
            plt.savefig(self.save_file + " conf_mat.png")
            plt.show()
        return confusion_matrix

    def eval_with_classifier_head(self):
        print("PATCH_SIZE=", self.input_shape)
        benchmarks_save = self.save_file + " accuracy.txt"
        enc = htm.EncoderBuilder()
        img_w, img_h, img_c = self.input_shape
        enc_img = enc.add_image(img_w, img_h, img_c, 0.8)
        out_volume = self.output_shape.prod()

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
                img = torch.zeros(out_volume)
                img[list(img_bits)] = 1
                return img, self.lbls[idx]

        train_mnist = MNIST[:40000]
        train_labels = LABELS[:40000]
        train_d = D(train_mnist, train_labels, self)

        eval_mnist = MNIST[40000:60000]
        eval_labels = LABELS[40000:60000]
        eval_d = D(eval_mnist, eval_labels, self)

        linear = torch.nn.Linear(out_volume, 10)
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
            with open(benchmarks_save, "a+") as f:
                print("epoch=", epoch, "train=", train_accuracy / train_total, "eval=", eval_accuracy / eval_total, file=f)
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
        self.output_shape = np.array(self.input_to_layer1.out_shape)

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


class Experiment21(Experiment):
    def __init__(self, w, h):
        super().__init__(w, h, "experiment21oc" + str(w * h))
        ex2 = Experiment2(7, 7)
        ex2.plot = False
        ex2.num_of_snapshots = 6
        ex2.drift = np.array([3, 3])
        self.conf_mat = ex2.measure_cross_correlation()
        self.layer2_save_file = self.save_file + " layer2.model"
        if os.path.exists(self.layer2_save_file):
            self.layer2 = ecc.CpuEccDense.load(self.layer2_save_file)
        else:
            self.layer2 = ecc.CpuEccDense([1, 1], [1, 1], [1, 1], ex2.layer1.out_channels, self.w * self.h, 1)
            self.layer2.threshold = 1 / self.layer2.in_channels
        self.layer1 = ex2.layer1
        self.input_shape = ex2.input_shape
        self.sdr = None
        self.drift = ex2.drift
        self.plot = False
        self.num_of_snapshots = ex2.num_of_snapshots

    def run(self, sdr, learn, update_activity):
        pre_sdr = self.layer1.infer(sdr)
        fin_sdr = self.layer2.infer(pre_sdr)
        if self.sdr is None or len(self.sdr) == 0:
            self.sdr = fin_sdr
            if update_activity:
                self.layer2.decrement_activities(fin_sdr)
        if learn:
            self.layer2.learn(pre_sdr, self.sdr)
        return self.sdr

    def take_action(self):
        self.sdr = None

    def save(self):
        visualise_connection_heatmap(7, 7, self.layer2, 3, 3, 0.001)
        fit = compute_confusion_matrix_fit(self.conf_mat, self.layer2, metric_l2=True)
        print("fit l2=", fit)
        print(fit.sum())
        fit = compute_confusion_matrix_fit(self.conf_mat, self.layer2, metric_l2=False)
        print("fit l1=", fit)
        print(fit.sum())
        # visualise_connection_heatmap(7, 7, self.layer2, 3, 3)
        self.layer2.save(self.layer2_save_file)

    def sums(self, output_sdr):
        return self.layer2.sums_for_sdr(output_sdr)


class Experiment22(Experiment):
    def __init__(self, w, h):
        super().__init__(w, h, "experiment22")
        ex2 = Experiment2(7, 7)
        ex2.num_of_snapshots = 6
        ex2.drift = np.array([3, 3])
        self.conf_mat = ex2.measure_cross_correlation()
        self.layer2_save_file = self.save_file + " layer2.model"
        self.layer1_to_layer2_save_file = self.save_file + " layer1_to_layer2.model"
        self.layer2_to_layer2_save_file = self.save_file + " layer2_to_layer2.model"
        if os.path.exists(self.layer2_save_file):
            self.layer1_to_layer2 = ecc.ConvWeights.load(self.layer1_to_layer2_save_file)
            self.layer2_to_layer2 = ecc.ConvWeights.load(self.layer2_to_layer2_save_file)
            self.layer2 = ecc.CpuEccPopulation.load(self.layer2_save_file)
        else:
            c = ex2.layer1.out_channels
            self.layer1_to_layer2 = ecc.ConvWeights([1, 1, c], [1, 1], [1, 1], self.w * self.h)
            self.layer2_to_layer2 = ecc.ConvWeights([1, 1, c], [1, 1], [1, 1], c)
            self.layer2 = ecc.CpuEccPopulation([1, 1, c], 1)
            self.layer2.threshold = 1 / c
        self.layer1 = ex2.layer1
        self.input_shape = ex2.input_shape
        self.sdr = htm.CpuSDR()
        self.drift = np.array([3, 3])
        self.num_of_snapshots = 6

    def run(self, sdr, learn, update_activity):
        pre_sdr = self.layer1.infer(sdr)
        self.layer1_to_layer2.reset_and_forward(pre_sdr, self.layer2)
        self.layer2_to_layer2.forward(self.sdr, self.layer2)
        new_state = self.layer2.determine_winners_top1_per_region()
        self.layer2.decrement_activities(new_state)
        if learn:
            self.layer1_to_layer2.learn(pre_sdr, new_state)
            self.layer2_to_layer2.learn(self.sdr, new_state)
        self.sdr = new_state
        return new_state

    def take_action(self):
        self.sdr = htm.CpuSDR()

    def save(self):
        # visualise_connection_heatmap(7, 7, self.layer2, 3, 3)
        self.layer2.save(self.layer2_save_file)
        self.layer1_to_layer2.save(self.layer1_to_layer2_save_file)
        self.layer2_to_layer2.save(self.layer2_to_layer2_save_file)

    def sums(self, output_sdr):
        return self.layer2.sums_for_sdr(output_sdr)


class Experiment23(Experiment):
    def __init__(self, w, h, no_layer2=False):
        super().__init__(0, 0, "experiment23oc" + str(w * h) + ("_no_l2" if no_layer2 else ""))
        ex21 = Experiment21(w, h)
        self.m = ex21.layer1.repeat_column([23, 23]).to_machine()
        if not no_layer2:
            self.m.push(ex21.layer2.repeat_column([23, 23]))
        self.input_shape = np.array(self.m.in_shape)
        self.output_shape = np.array(self.m.out_shape)
        self.plot = False

    def run(self, sdr, learn, update_activity):
        self.m.infer(sdr)
        return self.m.last_output_sdr()


class Experiment11(Experiment):
    def __init__(self):
        super().__init__(15, 15, "experiment11")
        ex1 = Experiment1()
        self.layer1 = ecc.CpuEccDense.new_from(ex1.input_to_layer1, ex1.layer1)
        self.layer1 = self.layer1.repeat_column([11, 11])
        self.input_shape = np.array(self.layer1.in_shape)
        self.output_shape = np.array(self.layer1.out_shape)

    def run(self, sdr, learn, update_activity):
        return self.layer1.infer(sdr)


class Experiment4(Experiment):
    def __init__(self, w, h):
        super().__init__(w, h, "experiment4oc" + str(w * h))
        self.input_to_layer1_save_file = self.save_file + " input_to_layer1.model"
        self.layer1_to_layer1_save_file = self.save_file + " layer1_to_layer1.model"
        self.layer1_save_file = self.save_file + " layer1.model"
        if os.path.exists(self.input_to_layer1_save_file):
            self.layer1 = ecc.CpuEccPopulation.load(self.layer1_save_file)
            self.input_to_layer1 = ecc.ConvWeights.load(self.input_to_layer1_save_file)
            self.layer1_to_layer1 = ecc.ConvWeights.load(self.layer1_to_layer1_save_file)
        else:
            self.layer1 = ecc.CpuEccPopulation([1, 1, self.w * self.h], 1)
            self.input_to_layer1 = ecc.ConvWeights(self.layer1.shape, [5, 5], [1, 1], 1)
            self.layer1_to_layer1 = ecc.ConvWeights(self.layer1.shape, [1, 1], [1, 1], self.layer1.channels)
        self.stored_sums = ecc.WeightSums(self.layer1.shape)
        self.layer1_sdr = htm.CpuSDR()
        self.input_shape = np.array(self.input_to_layer1.in_shape)
        self.output_shape = np.array(self.input_to_layer1.out_shape)
        self.num_of_snapshots = 6
        self.drift = np.array([3, 3])

    def run(self, input_sdr, learn, update_activity):
        self.input_to_layer1.reset_and_forward(input_sdr, self.layer1, parallel=PARALLEL)
        self.layer1_to_layer1.forward(self.layer1_sdr, self.layer1, parallel=PARALLEL)
        new_sdr = self.layer1.determine_winners_top1_per_region()
        if update_activity:
            self.layer1.decrement_activities(new_sdr)
        if learn:
            self.input_to_layer1.learn(input_sdr, new_sdr, self.stored_sums, PARALLEL)
            self.layer1_to_layer1.learn(self.layer1_sdr, new_sdr, self.stored_sums, PARALLEL)
            self.input_to_layer1.normalize_with_stored_sums(new_sdr, self.stored_sums, PARALLEL)
            self.layer1_to_layer1.normalize_with_stored_sums(new_sdr, self.stored_sums, PARALLEL)
            self.stored_sums.clear(new_sdr, PARALLEL)
        self.layer1_sdr = new_sdr
        return new_sdr

    def save(self):
        self.input_to_layer1.save(self.input_to_layer1_save_file)
        self.layer1_to_layer1.save(self.layer1_to_layer1_save_file)
        self.layer1.save(self.layer1_save_file)

    def take_action(self):
        self.layer1_sdr = htm.CpuSDR()

    def sums(self, output_sdr):
        return self.layer1.sums_for_sdr(output_sdr)


class Experiment24(Experiment):
    def __init__(self, w, h):
        super().__init__(w, h, "experiment24oc" + str(w * h))
        ex21 = Experiment21(9, 9)
        self.layer3_save_file = self.save_file + " layer3.model"
        if os.path.exists(self.layer3_save_file):
            self.layer3 = ecc.CpuEccDense.load(self.layer3_save_file)
        else:
            self.layer3 = ecc.CpuEccDense([1, 1], [5, 5], [1, 1], ex21.layer2.out_channels, self.w*self.h, 1)
            self.layer3.threshold = 1 / self.layer3.in_channels
        self.m = ex21.layer1.repeat_column(self.layer3.in_grid).to_machine()
        self.m.push(ex21.layer2.repeat_column(self.layer3.in_grid))
        self.input_shape = np.array(self.m.in_shape)
        self.output_shape = np.array(self.layer3.out_shape)

    def run(self, sdr, learn, update_activity):
        self.m.infer(sdr)
        pre_sdr = self.m.last_output_sdr()
        fin_sdr = self.layer3.run(pre_sdr,learn=learn, update_activity=update_activity)
        return fin_sdr

    def save(self):
        self.layer3.save(self.layer3_save_file)

    def sums(self, output_sdr):
        return self.layer3.sums_for_sdr(output_sdr)


class Experiment241(Experiment):
    def __init__(self, w, h, c):
        super().__init__(1, c, "experiment241oc" + str(w * h)+"oc"+str(c))
        ex24 = Experiment24(w, h)
        self.layer4_save_file = self.save_file + " layer4.model"
        if os.path.exists(self.layer4_save_file):
            self.layer4 = ecc.CpuEccDense.load(self.layer4_save_file)
        else:
            self.layer4 = ecc.CpuEccDense([1, 1], [1, 1], [1, 1], ex24.layer3.out_channels, c, 1)
            self.layer4.threshold = 0.000001
        self.m = ex24.m
        self.m.push(ex24.layer3)
        self.input_shape = np.array(self.m.in_shape)
        self.output_shape = np.array(self.layer4.out_shape)
        self.sdr = None
        self.drift = int(self.input_shape[:2].mean()/2+0.5)
        self.plot = False
        self.num_of_snapshots = 6

    def run(self, sdr, learn, update_activity):
        self.m.infer(sdr)
        pre_sdr = self.m.last_output_sdr()
        fin_sdr = self.layer4.run(pre_sdr, learn=learn, update_activity=update_activity)
        if self.sdr is None or len(self.sdr) == 0:
            self.sdr = fin_sdr
            if update_activity:
                self.layer4.decrement_activities(fin_sdr)
        if learn:
            self.layer4.learn(pre_sdr, self.sdr)
        return self.sdr

    def take_action(self):
        self.sdr = None

    def save(self):
        self.layer4.save(self.layer4_save_file)

    def sums(self, output_sdr):
        return self.layer4.sums_for_sdr(output_sdr)


class Experiment2411(Experiment):
    def __init__(self, w, h, c):
        super().__init__(1, c, "experiment2411oc" + str(w * h)+"oc"+str(c))
        ex241 = Experiment241(w, h, c)
        self.m = ex241.m
        self.m.push(ex241.layer4)
        self.m = self.m.repeat_column([19,19])
        self.input_shape = np.array(self.m.in_shape)
        self.output_shape = np.array(self.m.out_shape)

    def run(self, sdr, learn, update_activity):
        self.m.infer(sdr)
        return self.m.last_output_sdr()


class Experiment242(Experiment):
    def __init__(self, w, h):
        super().__init__(0, 0, "experiment242oc" + str(w * h))
        ex24 = Experiment24(w, h)
        self.m = ex24.m
        self.m.push(ex24.layer3)
        self.m = self.m.repeat_column([19,19])
        self.input_shape = np.array(self.m.in_shape)
        self.output_shape = np.array(self.m.out_shape)

    def run(self, sdr, learn, update_activity):
        self.m.infer(sdr)
        return self.m.last_output_sdr()

PARALLEL = False

# Experiment24(12, 12).experiment()
# Experiment242(12, 12).eval_with_classifier_head()
# Experiment241(12, 12, 16).experiment()
# Experiment2411(12, 12, 16).eval_with_classifier_head()

# e = Experiment4(5, 8)
# e.experiment()
# cm = e.measure_cross_correlation()
# ## produces: predictive_coding_stacked6_experiment4oc40 conf_mat.png
# visualise_recursive_weights(e.w, e.h, e.layer1_to_layer1)
# ## produces: predictive_coding_stacked6_experiment4oc40 layer1_to_layer1_weights.png

