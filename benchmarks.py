import rusty_neat
from rusty_neat import ndalgebra as nd
import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import numpy as np
import os
import sys
import importlib, inspect


def genGabor(sz, omega, theta, func=np.cos, K=np.pi):
    radius = (int(sz[0] / 2.0), int(sz[1] / 2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))
    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)
    gauss = omega ** 2 / (4 * np.pi * K ** 2) * np.exp(- omega ** 2 / (8 * K ** 2) * (4 * x1 ** 2 + y1 ** 2))
    sinusoid = func(omega * x1) * np.exp(K ** 2 / 2)
    gabor = gauss * sinusoid
    return gabor


def encode_img_gabor(img, target, filters, encoders, visualize=False):
    img = img.type(torch.float) / 255
    if visualize:
        fig, axs = plt.subplots(len(filters) + 1)
        axs[0].imshow(img)
    if len(filters) > 0:
        for k, (kernel, encoder) in enumerate(zip(filters, encoders)):
            i = ndimage.convolve(img, kernel, mode='constant')
            i = i > 5
            if visualize:
                axs[1 + k].imshow(i)
            i = i.reshape(28 * 28)
            i = i.tolist()
            encoder.encode(target, i)
    else:
        i = img > 0.8
        if visualize:
            axs[1].imshow(i)
        i = i.reshape(28 * 28)
        i = i.tolist()
        encoders.encode(target, i)
    if visualize:
        plt.show()


# plt.imshow(genGabor((28, 28), 2, np.pi * 0.1, func=np.cos))
def gen_gabor_filters():
    return [genGabor((8, 8), 2, np.pi * x / 8, func=np.cos) for x in range(0, 8)]


class Model:

    def __init__(self, gabor_filters, cat, htm_class, syn, neg=0., update_method='update'):
        self.gabor_filters = gabor_filters
        self.inp_enc = rusty_neat.htm.EncoderBuilder()
        if len(gabor_filters) > 0:
            self.img_enc = [self.inp_enc.add_bits(28 * 28) for _ in gabor_filters]
        else:
            self.img_enc = self.inp_enc.add_bits(28 * 28)
        self.syn = syn
        self.neg = neg
        self.htm_class = htm_class
        self.out_enc = rusty_neat.htm.EncoderBuilder()
        self.lbl_enc = self.out_enc.add_categorical(10, cat)
        self.sdr = rusty_neat.htm.CpuSDR()
        self.bitset = rusty_neat.htm.CpuBitset(self.inp_enc.input_size)
        self.active_columns_bitset = rusty_neat.htm.CpuBitset(self.out_enc.input_size)
        if update_method == 'update':
            self.infer = self.update_permanence
        elif update_method == 'penalize':
            self.infer = self.update_permanence_and_penalize

    def generate_htm(self):
        if self.htm_class == 2:
            htm = rusty_neat.htm.CpuHTM2(self.inp_enc.input_size, self.out_enc.input_size,
                                         self.lbl_enc.sdr_cardinality,
                                         int(self.inp_enc.input_size * self.syn))
        elif self.htm_class == 4:
            htm = rusty_neat.htm.CpuHTM4(self.inp_enc.input_size, self.out_enc.input_size,
                                         self.lbl_enc.sdr_cardinality,
                                         int(self.inp_enc.input_size * self.syn),
                                         self.neg)
        elif self.htm_class == '4 rand neg':
            htm = rusty_neat.htm.cpu_htm4_new_globally_uniform_prob(self.inp_enc.input_size,
                                                                    self.out_enc.input_size,
                                                                    self.lbl_enc.sdr_cardinality,
                                                                    int(self.inp_enc.input_size * self.syn),
                                                                    self.neg)
        return htm

    def update_permanence_and_penalize(self, htm, img, lbl=None):
        self.bitset.clear()
        encode_img_gabor(img, self.bitset, self.gabor_filters, self.img_enc)
        if lbl is not None:
            self.lbl_enc.encode(self.active_columns_bitset, lbl)
            htm.update_permanence_and_penalize(self.active_columns_bitset, self.bitset)
            self.sdr.clear()
        else:
            predicted_columns = htm.compute(self.bitset)
            return self.lbl_enc.find_category_with_highest_overlap(predicted_columns)

    def update_permanence(self, htm, img, lbl=None):
        self.bitset.clear()
        encode_img_gabor(img, self.bitset, self.gabor_filters, self.img_enc)
        if lbl is not None:
            self.lbl_enc.encode(self.sdr, lbl)
            htm.update_permanence(self.sdr, self.bitset)
            self.sdr.clear()
        else:
            predicted_columns = htm.compute(self.bitset)
            return self.lbl_enc.find_category_with_highest_overlap(predicted_columns)

    def run(self, samples, repetitions=64, population=20):
        MNIST, LABELS = torch.load('htm/data/mnist.pt')
        shuffle = torch.randperm(len(MNIST))
        MNIST = MNIST[shuffle]
        LABELS = LABELS[shuffle]

        results_file = "benchmarks/" + str(type(self).__name__) + "-" + str(samples) + ".pth"
        results = torch.load(results_file) if os.path.exists(results_file) else np.zeros((population, repetitions, 2))
        for htm_instance_idx, zero in enumerate(results.sum((1, 2))):
            if zero == 0:
                break

        def show_results(instance_idx):

            plt.clf()
            plt.title(str(type(self).__name__) + " samp=" + str(samples) + " inst=" + str(instance_idx))
            plot_results(results, 0, "train")
            plot_results(results, 1, "test")
            plt.legend()
            plt.pause(0.001)

        show_results(htm_instance_idx)

        if htm_instance_idx < population:
            eval_points = [0, 1, 2, 3, 4, 5, 6, 7]
            if htm_instance_idx == 0:
                if samples == 100:
                    eval_points.extend([15, 16, 31, 32, 62, 63])
                elif samples == 400:
                    eval_points.extend([15, 16])
                assert repetitions > max(eval_points)

            for repetition in tqdm(range(max(eval_points) + 1), desc="Instance#" + str(htm_instance_idx),
                                   total=repetitions,
                                   position=0):

                htm_instance = self.generate_htm()
                for img, lbl in zip(MNIST[0:samples], LABELS[0:samples]):
                    self.infer(htm_instance, img, lbl)

                def eval(begin, end):
                    confusion_matrix = np.zeros((10, 10))
                    for img, lbl in tqdm(zip(MNIST[begin:end], LABELS[begin:end]),
                                         desc="Eval[" + str(begin) + ":" + str(end) + "]", total=end - begin,
                                         position=1):
                        guessed = self.infer(htm_instance, img)
                        confusion_matrix[guessed, lbl] += 1
                    return confusion_matrix

                def save_accuracy(idx, label):
                    confusion_matrix = eval(idx * samples, idx * samples + samples)
                    accuracy = confusion_matrix.trace() / confusion_matrix.sum()
                    results[htm_instance_idx, repetition, idx] = accuracy
                    print(label + " accuracy(" + str(htm_instance_idx) + "," + str(repetition) + "):", accuracy)
                    print(confusion_matrix)

                if repetition in eval_points:
                    save_accuracy(0, "Training")
                    save_accuracy(1, "Testing")
                    show_results(htm_instance_idx)
                    torch.save(results, results_file)


class VoteGaborCat1024HTM2Syn08(Model):

    def __init__(self):
        super().__init__(gen_gabor_filters(), 1024, 2, 0.8)


class VoteGaborCat1024HTM4Syn08Neg08(Model):

    def __init__(self):
        super().__init__(gen_gabor_filters(), 1024, 4, 0.8, neg=0.8)


class VoteGaborCat1024HTM4Syn08Neg02(Model):

    def __init__(self):
        super().__init__(gen_gabor_filters(), 1024, 4, 0.8, neg=0.2)


class VoteGaborCat1024HTM4Syn08Neg02Penalize(Model):

    def __init__(self):
        super().__init__(gen_gabor_filters(), 1024, 4, 0.8, neg=0.2, update_method='penalize')


class VoteGaborCat1024HTM4RandNegPermSyn08Neg02(Model):

    def __init__(self):
        super().__init__(gen_gabor_filters(), 1024, '4 rand neg', 0.8, neg=0.2)


class VoteCat1024HTM2Syn08(Model):

    def __init__(self):
        super().__init__([], 1024, 2, 0.8)


class VoteCat1024HTM4Syn08Neg08(Model):

    def __init__(self):
        super().__init__([], 1024, 4, 0.8, neg=0.8)


class VoteCat1024HTM4Syn08Neg02(Model):

    def __init__(self):
        super().__init__([], 1024, 4, 0.8, neg=0.2)


class VoteCat1024HTM4Syn08Neg02Penalize(Model):

    def __init__(self):
        super().__init__([], 1024, 4, 0.8, neg=0.2, update_method='penalize')


class VoteCat1024HTM4RandNegPermSyn08Neg02(Model):

    def __init__(self):
        super().__init__([], 1024, '4 rand neg', 0.8, neg=0.2)


def plot_results(results, idx, label):
    train_results = results[:, :, idx]
    train_non_zero_results = train_results.copy()
    train_non_zero_results[train_non_zero_results != 0] = 1
    plt.plot(train_results.sum(0) / train_non_zero_results.sum(0), label=label)


def class_for_name(name):
    for c_name, c in inspect.getmembers(sys.modules[__name__]):
        if c.__module__ == '__main__' and name == c_name:
            return c()


def show_benchmarks(idx):
    for benchmarks in os.listdir('benchmarks'):
        model = benchmarks[:-len(".pth")]
        benchmarks = torch.load('benchmarks/' + benchmarks)
        plot_results(benchmarks, idx, label=model)
    plt.legend()
    plt.show()


def list_benchmarks():
    results = {}
    for f in os.listdir('benchmarks'):
        name, samples = f[:-len(".pth")].split("-")
        samples = int(samples)
        if name in results:
            results[name].append(samples)
        else:
            results[name] = [samples]

    class Candidate:
        def __init__(self, name, samples, evaluated_instances):
            self.name = name
            self.samples = samples
            self.evaluated_instances = evaluated_instances

        def __gt__(self, other):
            return self.samples > other.samples or (
                    self.samples == other.samples and self.evaluated_instances > other.evaluated_instances)

    candidates = []
    for name, c in inspect.getmembers(sys.modules[__name__]):
        if c != Model and type(c) is type and issubclass(c, Model):
            if name in results:
                results[name].sort()
                recommended_samples = [100, 400, 1000]
                for samples in results[name]:
                    if samples in recommended_samples:
                        recommended_samples.remove(samples)
                    benchmarks = torch.load('benchmarks/' + name + '-' + str(samples) + '.pth')
                    benchmarks = benchmarks[:, :, 0]
                    benchmarks = benchmarks.sum(1)
                    evaluated_instances = sum(benchmarks != 0)
                    candidates.append(Candidate(name, samples, evaluated_instances))
                for samples in recommended_samples:
                    candidates.append(Candidate(name, samples, 0))
            else:
                candidates.append(Candidate(name, 100, 0))

    candidates.sort()
    for c in candidates:
        print(c.name, c.samples, c.evaluated_instances)


if sys.argv[1] == "train":
    show_benchmarks(0)
elif sys.argv[1] == "test":
    show_benchmarks(1)
elif sys.argv[1] == "list":
    list_benchmarks()
else:
    class_for_name(sys.argv[1]).run(int(sys.argv[2]) if len(sys.argv) > 1 else 100)
