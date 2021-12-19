import rusty_neat
from rusty_neat import ndalgebra as nd
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


rand_seed = 25436466
out_kernel = [5, 5]
out_stride = [1, 1]
top_kernel = [2, 2]
top_stride = [1, 1]
in_shape = [out_channels, 28, 28]
out_shape = rusty_neat.htm.conv_out_size(in_shape[1:], out_stride, out_kernel)
column_count = prod(out_shape)
neurons_per_column = 16
out_shape[0] = neurons_per_column
top_shape = rusty_neat.htm.conv_out_size(out_shape[1:], top_stride, top_kernel)
top_shape[0] = neurons_per_column
shapes = [in_shape, out_shape, top_shape]
sizes = [prod(shape) for shape in shapes]
in_size, out_size, top_size = sizes
encoder_builder = rusty_neat.htm.EncoderBuilder()
in_encoder = encoder_builder.add_bits(in_size)
out_encoder = encoder_builder.add_bits(out_size)
top_encoder = encoder_builder.add_bits(top_size)
n = 4
synapses_percentage = 1.0
input_volume = neurons_per_column * prod(out_kernel)
synapses_per_segment = int(synapses_percentage * input_volume)
out_pop = rusty_neat.htm.Population(0)
rand_seed = out_pop.push_add_2d_column_grid_with_3d_input(
    input_range=(0, in_size),
    neurons_per_column=neurons_per_column,
    segments_per_neuron=1,
    synapses_per_segment=synapses_per_segment,
    stride=out_stride,
    kernel=out_kernel,
    input_size=in_shape,
    rand_seed=rand_seed
)
out_htm = rusty_neat.htm.CpuHTM2(in_size, n)
rand_seed = out_htm.add_population(out_pop, rand_seed=rand_seed)
top_pop = rusty_neat.htm.Population(0)
rand_seed = top_pop.push_add_2d_column_grid_with_3d_input(
    input_range=(in_size, in_size+out_size),
    neurons_per_column=neurons_per_column,
    segments_per_neuron=1,
    synapses_per_segment=synapses_per_segment,
    stride=top_stride,
    kernel=top_kernel,
    input_size=in_shape,
    rand_seed=rand_seed
)
top_htm = rusty_neat.htm.CpuHTM2(out_size, n)
rand_seed = out_htm.add_population(top_pop, rand_seed=rand_seed)
bitset = rusty_neat.htm.CpuBitset(sum(sizes))


def visualise(x):
    htm.visualise([in_shape], [out_shape], input=bitset.to_sdr(), output=x)


MNIST, LABELS = torch.load('htm/data/mnist.pt')


def train(samples, repetitions, learn=True):
    for _ in range(repetitions):
        for img, lbl in tqdm(zip(MNIST[samples], LABELS[samples]), desc="training", total=len(samples)):
            img = img.type(torch.float) / 255
            img = img.reshape(1, 1, 28, 28)
            img = torch.conv2d(img, GABOR_FILTERS, padding=1)
            img = img.squeeze(0)
            img = img > 0.8
            bitset.clear()
            in_encoder.encode(bitset, img)
            x = htm.infer_and_group_into_columns(neurons_per_column, column_count, bitset, learn)
            cast_votes = rusty_neat.htm.vote_conv2d(x, n, vote_threshold, [1,1], [2,2])

train(range(100), 1)
