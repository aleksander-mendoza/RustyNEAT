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
import itertools
import importlib, inspect

# ["Htm", ["n", "g"], "1024", ["2", "3", "4#0.2", "4#0.8"], "0.8", ["update", "penalize", "ltd"], ["100", "400", "1000"]]

BENCHMARK_DIR = "benchmarks2"
BENCHMARK_LIST = [
    "Htm n 1024 2 0.8 update 10000",
    "Htm n 1024 2 0.8 ltd 10000",
    "Htm n 1024 4#0.2 0.8 update 10000",
    "Htm n 1024 4#0.2 0.8 ltd 10000",
]

POPULATION = 20
TEST_SIZE = 400
TRAIN_SIZE = 4000


def genGabor(sz, omega, theta, func=np.cos, K=np.pi):
    radius = (int(sz[0] / 2.0), int(sz[1] / 2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))
    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)
    gauss = omega ** 2 / (4 * np.pi * K ** 2) * np.exp(- omega ** 2 / (8 * K ** 2) * (4 * x1 ** 2 + y1 ** 2))
    sinusoid = func(omega * x1) * np.exp(K ** 2 / 2)
    gabor = gauss * sinusoid
    return gabor


def visualize_gabor(img, filters, threshold):
    img = img.type(torch.float) / 255
    fig, axs = plt.subplots(len(filters) + 1, 2)
    axs[0, 0].imshow(img)
    for k, kernel in enumerate(filters):
        i = ndimage.convolve(img, kernel, mode='constant')
        i = i > threshold
        axs[1 + k, 0].imshow(i)
        axs[1 + k, 1].imshow(kernel)
    plt.show()


def encode_img_gabor(img, targets, filters, encoders, threshold=5.):
    img = img.type(torch.float) / 255
    for k, (kernel, encoder) in enumerate(zip(filters, encoders)):
        i = img
        if kernel is not None:
            i = ndimage.convolve(i, kernel, mode='constant')
        i = i > threshold
        i = i.reshape(28 * 28)
        i = i.tolist()
        for target in targets:
            encoder.encode(target, i)


def htm_for_code(inp_size, card, out_size, syn_per_col, code: str):
    if code.startswith('2'):
        htm = rusty_neat.htm.CpuHTM2(inp_size, out_size, card, syn_per_col)
    elif code.startswith('3'):
        htm = rusty_neat.htm.CpuHTM3(inp_size, out_size, card, syn_per_col)
    elif code.startswith('4'):
        code, excitatory_perm_prob = code.split("#")
        excitatory_perm_prob = float(excitatory_perm_prob)
        if code == '4rn':
            htm = rusty_neat.htm.cpu_htm4_new_globally_uniform_prob(inp_size, out_size, card, syn_per_col,
                                                                    excitatory_perm_prob)
        else:
            htm = rusty_neat.htm.CpuHTM4(inp_size, out_size, card, syn_per_col, excitatory_perm_prob)
    return htm


# plt.imshow(genGabor((28, 28), 1.5, np.pi * 0.1, func=np.cos))

class Model:

    def __init__(self, gabor_filters, cat, htm_class, syn, update_method='update'):
        self.args = [self.__class__.__name__, gabor_filters, cat, htm_class, syn, update_method]
        if gabor_filters == 'g':
            self.gabor_filters = [genGabor((8, 8), 2, np.pi * x / 8, func=np.cos) for x in range(0, 8)]
            self.gabor_threshold = 5
        elif gabor_filters == 'b':
            self.gabor_filters = [genGabor((16, 16), 1.5, np.pi * x / 8, func=np.cos) for x in range(0, 8)]
            self.gabor_threshold = 8
        elif gabor_filters == 'n':
            self.gabor_filters = [None]
            self.gabor_threshold = 0.8
        self.syn = syn
        self.htm_class = htm_class
        if update_method == 'update':
            self.infer = self.update_permanence
        elif update_method == 'penalize':
            self.infer = self.update_permanence_and_penalize
        elif update_method == 'ltd':
            self.infer = self.update_permanence_ltd

    def generate_htm(self):
        pass

    def encode_img_gabor(self, img, targets):
        for target in targets:
            target.clear()
        encode_img_gabor(img, targets, self.gabor_filters, self.img_enc, threshold=self.gabor_threshold)

    def update_permanence_and_penalize(self, htm, img, lbl=None):
        pass

    def update_permanence(self, htm, img, lbl=None):
        pass

    def update_permanence_ltd(self, htm, img, lbl=None):
        pass

    def name(self):
        return ' '.join(map(str, self.args))

    def run(self, samples, ):
        MNIST, LABELS = torch.load('htm/data/mnist.pt')
        shuffle = torch.randperm(len(MNIST))
        MNIST = MNIST[shuffle]
        LABELS = LABELS[shuffle]

        name = self.name()
        results_file = "/" + name + " " + str(samples) + ".pth"
        results = torch.load(results_file) if os.path.exists(results_file) else np.zeros((POPULATION, repetitions, 2))
        for htm_instance_idx, zero in enumerate(results.sum((1, 2))):
            if zero == 0:
                break

        def show_results(instance_idx):

            plt.clf()
            plt.title(name + " " + str(samples) + " #" + str(instance_idx))
            plt.plot(avg_results(results, 0), label="train")
            plt.plot(avg_results(results, 1), label="test")
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

                def eval(begin, end):
                    confusion_matrix = np.zeros((10, 10))
                    for img, lbl in tqdm(zip(MNIST[begin:end], LABELS[begin:end]),
                                         desc="Eval[" + str(begin) + ":" + str(end) + "]", total=end - begin,
                                         position=1):
                        guessed = self.infer(htm_instance, img)
                        confusion_matrix[guessed, lbl] += 1
                    return confusion_matrix

                def calc_accuracy(begin, end):
                    confusion_matrix = eval(begin, end)
                    return confusion_matrix.trace() / confusion_matrix.sum()

                def save_accuracy(idx, label):
                    confusion_matrix = eval(idx * samples, idx * samples + samples)
                    accuracy = confusion_matrix.trace() / confusion_matrix.sum()
                    results[htm_instance_idx, repetition, idx] = accuracy
                    print(label + " accuracy(" + str(htm_instance_idx) + "," + str(repetition) + "):", accuracy)
                    print(confusion_matrix)

                htm_instance = self.generate_htm()
                if sample_checkpoint <= samples:
                    accuracies = [calc_accuracy(samples, samples + 1000)]
                for sample_idx, (img, lbl) in enumerate(zip(MNIST[0:samples], LABELS[0:samples])):
                    self.infer(htm_instance, img, lbl)
                    if sample_idx % sample_checkpoint == sample_checkpoint - 1:
                        accuracies.append(calc_accuracy(samples, samples + 1000))
                        plt.clf()
                        plt.plot(accuracies)
                        plt.pause(0.01)
                if sample_checkpoint <= samples:
                    plt.show()

                if repetition in eval_points:
                    save_accuracy(0, "Training")
                    save_accuracy(1, "Testing")
                    show_results(htm_instance_idx)
                    torch.save(results, results_file)


class Htm(Model):

    def __init__(self, gabor_filters, cat, htm_class, syn, update_method='update'):
        super().__init__(gabor_filters, cat, htm_class, syn, update_method)
        self.inp_enc = rusty_neat.htm.EncoderBuilder()
        self.img_enc = [self.inp_enc.add_bits(28 * 28) for _ in self.gabor_filters]
        self.out_enc = rusty_neat.htm.EncoderBuilder()
        self.lbl_enc = self.out_enc.add_categorical(10, cat)
        self.sdr = rusty_neat.htm.CpuSDR()
        self.bitset = rusty_neat.htm.CpuBitset(self.inp_enc.input_size)
        self.active_columns_bitset = rusty_neat.htm.CpuBitset(self.out_enc.input_size)

    def generate_htm(self):
        return htm_for_code(self.inp_enc.input_size,
                            self.lbl_enc.sdr_cardinality,
                            self.out_enc.input_size,
                            int(self.inp_enc.input_size * self.syn),
                            self.htm_class)

    def update_permanence_and_penalize(self, htm, img, lbl=None):
        self.encode_img_gabor(img, [self.bitset])
        if lbl is not None:
            self.lbl_enc.encode(self.active_columns_bitset, lbl)
            htm.update_permanence_and_penalize(self.active_columns_bitset, self.bitset)
            self.sdr.clear()
        else:
            predicted_columns = htm.compute(self.bitset)
            return self.lbl_enc.find_category_with_highest_overlap(predicted_columns)

    def update_permanence(self, htm, img, lbl=None):
        self.encode_img_gabor(img, [self.bitset])
        if lbl is not None:
            self.lbl_enc.encode(self.sdr, lbl)
            htm.update_permanence(self.sdr, self.bitset)
            self.sdr.clear()
        else:
            predicted_columns = htm.compute(self.bitset)
            return self.lbl_enc.find_category_with_highest_overlap(predicted_columns)

    def update_permanence_ltd(self, htm, img, lbl=None):
        self.encode_img_gabor(img, [self.bitset])
        predicted_columns = htm.compute(self.bitset)
        if lbl is not None:
            self.lbl_enc.encode(self.active_columns_bitset, lbl)
            htm.update_permanence_ltd(predicted_columns, self.active_columns_bitset, self.bitset)
            self.active_columns_bitset.clear()
        else:
            return self.lbl_enc.find_category_with_highest_overlap(predicted_columns)


class HtmHom(Model):

    def __init__(self, gabor_filters, cat, htm_class, syn, update_method='update'):
        super().__init__(gabor_filters, cat, htm_class, syn, update_method)
        self.inp_enc = rusty_neat.htm.EncoderBuilder()
        self.img_enc = [self.inp_enc.add_bits(28 * 28) for _ in self.gabor_filters]
        self.out_enc = rusty_neat.htm.EncoderBuilder()
        self.lbl_enc = self.out_enc.add_categorical(10, cat)
        self.sdr = rusty_neat.htm.CpuSDR()
        self.bitset = rusty_neat.htm.CpuBitset(self.inp_enc.input_size)
        self.active_columns_bitset = rusty_neat.htm.CpuBitset(self.out_enc.input_size)

    def generate_htm(self):
        htm1 = htm_for_code(self.inp_enc.input_size,
                            self.lbl_enc.sdr_cardinality,
                            self.out_enc.input_size,
                            int(self.inp_enc.input_size * self.syn),
                            self.htm_class)
        hom = rusty_neat.htm.CpuHOM(4, self.enc.input_size)

    def update_permanence(self, img, htm, lbl=None):
        htm1, hom = htm
        self.encode_img_gabor(img, [self.bitset, self.sdr])
        active_columns = htm1(self.bitset, lbl is not None)
        hom(self.sdr, lbl is not None)
        predicted_columns = hom(active_columns, lbl is not None)
        if lbl is not None:
            self.sdr.clear()
            self.lbl_enc.encode(self.sdr, lbl)
            predicted_columns = hom(self.sdr, True)
        hom.reset()
        return self.lbl_enc.find_category_with_highest_overlap(predicted_columns)


def avg_results(results, idx):
    train_results = results[:, :, idx]
    train_non_zero_results = train_results.copy()
    train_non_zero_results[train_non_zero_results != 0] = 1
    avg = train_results.sum(0) / train_non_zero_results.sum(0)
    return avg


def show_benchmarks(idx):
    fig = plt.figure()
    plot = fig.add_subplot(111)
    model_to_results = {}
    for benchmarks in os.listdir(BENCHMARK_DIR):
        model = benchmarks[:-len(".pth")]
        benchmarks = torch.load(BENCHMARK_DIR+'/' + benchmarks)
        avg = avg_results(benchmarks, idx)
        plot.plot(avg, label=model, gid=model)
        model_to_results[model] = avg

    annot = plot.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                          bbox=dict(boxstyle="round", fc="w"),
                          arrowprops=dict(arrowstyle="->"))

    def on_plot_hover(event):
        # Iterating over each data member plotted
        for curve in plot.get_lines():
            # Searching which data member corresponds to current mouse position
            if curve.contains(event)[0]:
                model = curve.get_gid()
                x = int(event.xdata)
                y = model_to_results[model][x]
                annot.xy = x, y
                annot.set_text(model + " " + str((x, y)))
                fig.canvas.draw_idle()
                return

    fig.canvas.mpl_connect('button_press_event', on_plot_hover)
    plt.show()


def list_benchmarks():
    results = {}
    for f in os.listdir(BENCHMARK_DIR):
        name, samples = f[:-len(".pth")].rsplit(' ', maxsplit=1)
        samples = int(samples)
        if name in results:
            results[name].append(samples)
        else:
            results[name] = [samples]

    class Candidate:
        def __init__(self, favourite, name, samples, evaluated_instances):
            self.favourite = favourite
            self.name = name
            self.samples = samples
            self.evaluated_instances = evaluated_instances

        def __gt__(self, other):
            if self.favourite and not other.favourite:
                return True
            if not self.favourite and other.favourite:
                return False
            return self.samples > other.samples or (
                    self.samples == other.samples and self.evaluated_instances > other.evaluated_instances)

    candidates = []

    def add_entry(favourite, name):
        if name in results:
            results[name].sort()
            recommended_samples = [100, 400, 1000]
            for samples in results[name]:
                if samples in recommended_samples:
                    recommended_samples.remove(samples)
                benchmarks = torch.load(BENCHMARK_DIR+'/' + name + '-' + str(samples) + '.pth')
                benchmarks = benchmarks[:, :, 0]
                benchmarks = benchmarks.sum(1)
                evaluated_instances = sum(benchmarks != 0)
                candidates.append(Candidate(favourite, name, samples, evaluated_instances))
            for samples in recommended_samples:
                candidates.append(Candidate(favourite, name, samples, 0))
        else:
            candidates.append(Candidate(favourite, name, 100, 0))

    favourite_set = set(BENCHMARK_LIST)
    for name in BENCHMARK_LIST:
        add_entry(True, name)
    for name in results.keys():
        if name not in favourite_set:
            add_entry(True, name)

    candidates.sort()
    for c in candidates:
        print(c.name, c.samples, c.evaluated_instances)


def run(model_class, gabor_filters, cat, htm_class, syn, update_method, samples, sample_checkpoint):
    for c_name, c in inspect.getmembers(sys.modules[__name__]):
        if type(c) == type and c.__module__ == '__main__' and model_class == c_name:
            c(gabor_filters, int(cat), str(htm_class), float(syn), update_method).run(int(samples),
                                                                                      int(sample_checkpoint))


list_benchmarks()
exit()
# run("Htm", "g", 1024, "2", 0.8, "update", 10000, 500)
# exit()
if sys.argv[1] == "train":
    show_benchmarks(0)
elif sys.argv[1] == "test":
    show_benchmarks(1)
elif sys.argv[1] == "list":
    list_benchmarks()
else:
    run(*sys.argv[1:])
