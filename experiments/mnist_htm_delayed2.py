import rusty_neat
from rusty_neat import ndalgebra as nd
from rusty_neat import htm
import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import numpy as np

GABOR_FILTERS = [
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
    # np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32),
    # np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32),
    # np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32),
    # np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32)
]

GABOR_FILTERS = np.dstack(GABOR_FILTERS)

GABOR_FILTERS = np.transpose(GABOR_FILTERS, (2, 0, 1))
in_channels = 1
out_channels, kernel_h, kernel_w = GABOR_FILTERS.shape
GABOR_FILTERS = GABOR_FILTERS.reshape(out_channels, in_channels, kernel_h, kernel_w)
GABOR_FILTERS = torch.from_numpy(GABOR_FILTERS)

S = 28 * 28


# intervals = [(f * S, int(S * 0.8), (f + 1) * S) for f in range(len(GABOR_FILTERS))]

def prod(l):
    i = 1
    for e in l:
        i *= e
    return i


class Net:
    def __init__(self, in_shape, kernels, strides, neurons_per_column,
                 synapses_percentage, n_percentage, pattern_sep_ratio, group_into_columns, rand_seed):
        if type(group_into_columns) is bool:
            self.group_into_columns = [group_into_columns] * len(kernels)
        else:
            assert type(group_into_columns) is list
            self.group_into_columns = group_into_columns.copy()
        if type(n_percentage) is not list:
            n_percentage = [n_percentage] * len(kernels)
        assert len(self.group_into_columns) == len(kernels)
        self.group_into_columns.insert(0, False)
        self.shapes = [in_shape]
        self.kernels = kernels.copy()
        self.strides = strides.copy()
        self.neurons_per_column = neurons_per_column.copy()
        self.neurons_per_column.insert(0, in_shape[0])
        self.synapses_percentage = synapses_percentage
        self.sizes = [prod(in_shape)]
        self.offsets = [0]
        self.encoder_builder = rusty_neat.htm.EncoderBuilder()
        self.encoders = [self.encoder_builder.add_bits(self.sizes[0])]
        self.n = [0]
        self.pop = [None]
        assert len(kernels) == len(strides) == len(neurons_per_column) == len(n_percentage)
        assert len(self.group_into_columns) == len(kernels) + 1
        max_delay = len(synapses_percentage)
        for out_layer in range(1, len(kernels) + 1):
            stride = self.strides[out_layer - 1]
            kernel = self.kernels[out_layer - 1]
            in_size = self.sizes[out_layer - 1]
            out_shape = rusty_neat.htm.conv_out_size(in_shape[1:], stride, kernel)
            out_shape[0] = self.neurons_per_column[out_layer]
            self.shapes.append(out_shape)
            out_size = prod(out_shape)
            self.sizes.append(out_size)
            pop = rusty_neat.htm.Population(out_size)
            for depth, in_layer in enumerate(reversed(range(max(0, out_layer - max_delay), out_layer))):
                subpop = rusty_neat.htm.Population(0)
                input_volume = self.neurons_per_column[in_layer] * prod(kernel)
                synapses_per_segment = int(synapses_percentage[depth] * input_volume)
                input_range = (self.offsets[in_layer], self.offsets[in_layer] + self.sizes[in_layer])
                rand_seed = subpop.push_add_2d_column_grid_with_3d_input(
                    input_range=input_range,
                    neurons_per_column=self.neurons_per_column[out_layer],
                    segments_per_neuron=1,
                    synapses_per_segment=synapses_per_segment,
                    stride=stride,
                    kernel=kernel,
                    input_size=self.shapes[in_layer],
                    rand_seed=rand_seed
                )
                stride, kernel = rusty_neat.htm.conv_compose(self.strides[in_layer], self.kernels[in_layer], stride,
                                                             kernel)
                stride, kernel = stride[1:], kernel[1:]
                pop = pop * subpop
            self.pop.append(pop)
            self.encoders.append(self.encoder_builder.add_bits(out_size))
            n_per = n_percentage[out_layer-1]
            if type(n_per) is int:
                n = n_per
            else:
                assert type(n_per) is float
                if self.group_into_columns[out_layer]:
                    n = int(self.neurons_per_column[out_layer] * n_per)
                else:
                    n = int(out_size * n_per)
            self.n.append(n)
            self.offsets.append(self.offsets[out_layer - 1] + in_size)
            in_shape = out_shape
        self.htm = [None]
        self.total_size = sum(self.sizes)
        for layer in range(1, len(kernels) + 1):
            htm = rusty_neat.htm.CpuHTM2(self.total_size, self.n[layer])
            rand_seed = htm.add_population(self.pop[layer], rand_seed=rand_seed)
            self.htm.append(htm)
        assert len(self.htm) == len(self.pop) == len(self.offsets) == len(self.sizes)
        self.pattern_sep_layer = len(self.sizes)
        # Final layer for pattern separation
        in_size = self.sizes[self.pattern_sep_layer - 1]
        out_size = int(in_size * pattern_sep_ratio)

        self.sizes.append(out_size)
        self.offsets.append(self.offsets[self.pattern_sep_layer - 1] + in_size)
        out_shape = [pattern_sep_ratio, in_size, 1]
        self.shapes.append(out_shape)
        self.column_count = [prod(shape[1:]) for shape in self.shapes]
        synapses_per_segment = self.n[self.pattern_sep_layer - 1]
        if self.group_into_columns[self.pattern_sep_layer - 1]:
            synapses_per_segment = synapses_per_segment * self.column_count[self.pattern_sep_layer - 1]
        n = synapses_per_segment
        pop = rusty_neat.htm.Population(out_size)
        input_range = (self.offsets[self.pattern_sep_layer - 1],
                       self.offsets[self.pattern_sep_layer - 1] + self.sizes[self.pattern_sep_layer - 1])
        pop.add_uniform_rand_inputs_from_range(input_range, synapses_per_segment)
        self.pop.append(pop)
        self.n.append(n)
        htm = rusty_neat.htm.CpuHTM2(self.total_size, n, pop)
        htm.set_all_permanences(1.)
        self.htm.append(htm)
        self.encoders.append(self.encoder_builder.add_bits(out_size))
        self.bitset = rusty_neat.htm.CpuBitset(self.total_size)

        self.group_into_columns.append(False)
        assert len(self.group_into_columns) == len(self.shapes) == len(self.column_count) == len(self.sizes)

    def run(self, x, train):
        assert self.shapes[0] == list(x.shape), str(self.shapes[0]) + " " + str(x.shape)
        self.bitset.clear()
        for i in range(1, len(self.sizes)):
            self.encoders[i - 1].encode(self.bitset, x)
            learn = train and i != self.pattern_sep_layer
            if self.group_into_columns[i]:
                x = self.htm[i].infer_and_group_into_columns(self.neurons_per_column[i], self.column_count[i],
                                                             self.bitset, learn)
            else:
                x = self.htm[i](self.bitset, learn)
        return x

    def visualise(self, layer):
        in_shapes = self.shapes.copy()
        in_shapes.pop()
        out_shapes = [[0, 0, 0]] * len(in_shapes)
        out_shapes[layer - 1] = self.shapes[layer]
        self.htm[layer].visualise(in_shapes, out_shapes, input=self.bitset.to_sdr())


number_of_samples = 500, 500
EXPERIMENT_PROFILE = 3
if EXPERIMENT_PROFILE == 0:
    net = Net([out_channels, 28, 28], [[3, 3], [3, 3]], [[1, 1], [1, 1]], [4, 4], [0.5, 0.5], 0.05, 4, False, 3463657)
    categories_cardinality = 28
    # training
    # 100 / 100 = 1.0
    # 374 / 500 = 0.748
    # validation
    # (100) 45 / 100 = 0.45
    # (500) 191 / 500 = 0.382
elif EXPERIMENT_PROFILE == 1:
    net = Net([out_channels, 28, 28], [[3, 3], [3, 3], [3, 3], [3, 3]], [[1, 1], [1, 1], [1, 1], [1, 1]], [8, 8, 8, 4],
              [0.5, 0.5],
              0.05, 4, False, 3463657)
    categories_cardinality = 28
    # training
    # 92 / 100 = 0.92, 0.91, 0.91
    # 245 / 500 = 0.49
    # validation
    # (100) 30 / 100 = 0.3, 0.31, 0.4
    # (500) 174 / 500 = 0.348
elif EXPERIMENT_PROFILE == 2:
    net = Net([out_channels, 28, 28], [[28, 28]], [[1, 1]], [28 * 28 * 4], [0.5, 0.5],
              0.05, 4, False, 3463657)
    categories_cardinality = 28
    # training
    # 456 / 500 = 0.912
    # validation
    # (500) 261 / 500 = 0.522
elif EXPERIMENT_PROFILE == 3:
    net = Net([out_channels, 28, 28],
              kernels=[[5, 5], [5, 5]],
              strides=[[1, 1], [1, 1]],
              neurons_per_column=[16, 4],
              synapses_percentage=[0.8, 0.5, 0.3],
              n_percentage=[1, 1],
              pattern_sep_ratio=4,
              group_into_columns=True,
              rand_seed=3463657)
    categories_cardinality = 28
    # training
    # 439 / 500 = 0.878
    # validation
    # (500) 207 / 500 = 0.414
else:
    print("Unknown experiment profile")
    exit()

# net.visualise(3)

map_encoder_builder = htm.EncoderBuilder()
map_enc = map_encoder_builder.add_categorical(10, categories_cardinality)
map_shape = [map_enc.num_of_categories, map_enc.sdr_cardinality]
map_size = prod(map_shape)
categories = []
for category_idx in range(map_enc.num_of_categories):
    category_sdr = htm.CpuInput(map_size)
    map_enc.encode(category_sdr, category_idx)
    categories.append(category_sdr)

map_htm = htm.CpuBigHTM(input_size=net.sizes[net.pattern_sep_layer], minicolumns=map_size, n=4)

MNIST, LABELS = torch.load('htm/data/mnist.pt')


def train(samples, repetitions, train_cortex, train_map):
    correct_inferences = 0
    for _ in range(repetitions):
        for img, lbl in tqdm(zip(MNIST[samples], LABELS[samples]), desc="training", total=len(samples)):
            img = img.type(torch.float) / 255
            img = img.reshape(1, 1, 28, 28)
            img = torch.conv2d(img, GABOR_FILTERS, padding=1)
            img = img.squeeze(0)
            img = img > 0.8
            pattern_sep = net.run(img.numpy(), train_cortex)
            # pattern_sep.shift(-net.total_size)
            pattern_sep = pattern_sep.to_input(net.sizes[net.pattern_sep_layer])

            if train_map:
                # net.visualise(1)
                map_activity = map_htm.infer_from_whitelist(pattern_sep, categories[lbl], learn=True)
            else:
                map_activity = map_htm.infer(pattern_sep, learn=False)
                predicted_lbl = map_enc.find_category_with_highest_overlap(map_activity)
                correct_inferences += int(predicted_lbl == lbl)
    return correct_inferences


if type(number_of_samples) is tuple:
    training_samples, validation_samples = number_of_samples
else:
    training_samples, validation_samples = number_of_samples, 0
validation_samples = range(training_samples, training_samples + validation_samples)
training_samples = range(training_samples)
train(training_samples, 1, train_cortex=True, train_map=False)
train(training_samples, 1, train_cortex=False, train_map=True)
result_score = train(training_samples, 1, train_cortex=False, train_map=False)
number_of_samples = len(training_samples)
print(EXPERIMENT_PROFILE, "training", result_score, "/", number_of_samples, "=", result_score / number_of_samples)
number_of_samples = len(validation_samples)
if number_of_samples > 0:
    result_score = train(validation_samples, 1, train_cortex=False, train_map=False)
    print(EXPERIMENT_PROFILE, "validation", "(" + str(len(training_samples)) + ")", result_score, "/",
          number_of_samples,
          "=", result_score / number_of_samples)
