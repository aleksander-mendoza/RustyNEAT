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
    # np.array([[0,0,0], [0,1,0], [0,0,0]], dtype=np.float),
    np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float),
    np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float),
    np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float),
    np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float)
]
enc = rusty_neat.htm.EncoderBuilder()
S = 28 * 28


# intervals = [(f * S, int(S * 0.8), (f + 1) * S) for f in range(len(GABOR_FILTERS))]

def prod(l):
    i = 1
    for e in l:
        i *= e
    return i


layer1_shape = [len(GABOR_FILTERS), 28, 28]
layer1_size = prod(layer1_shape)
layer2_neurons_per_column = 4
layer1_to_2_stride = [2, 2]
layer1_to_2_kernel = [4, 4]
layer1_to_2_pop = htm.Population()
layer1_to_2_pop.push_add_2d_column_grid_with_3d_input(
    input_range=(0, layer1_size),
    neurons_per_column=layer2_neurons_per_column,
    segments_per_neuron=1,
    synapses_per_segment=30,
    stride=layer1_to_2_stride,
    kernel=layer1_to_2_kernel,
    input_size=layer1_shape,
    rand_seed=53463
)
layer2_shape = htm.conv_out_size(layer1_shape[1:], layer1_to_2_stride, layer1_to_2_kernel)
layer2_shape[0] = layer2_neurons_per_column
layer2_size = prod(layer2_shape)
layer3_neurons_per_column = 4
layer2_to_3_stride = [1, 1]
layer2_to_3_kernel = [3, 3]
layer2_to_3_pop = htm.Population()
layer2_to_3_pop.push_add_2d_column_grid_with_3d_input(
    input_range=(layer1_size, layer1_size + layer2_size),
    neurons_per_column=layer3_neurons_per_column,
    segments_per_neuron=1,
    synapses_per_segment=30,
    stride=layer2_to_3_stride,
    kernel=layer2_to_3_kernel,
    input_size=layer2_shape,
    rand_seed=43643908
)
layer1_to_3_stride, layer1_to_3_kernel = htm.conv_compose(layer1_to_2_stride, layer1_to_2_kernel,
                                                          layer2_to_3_stride, layer2_to_3_kernel)
layer1_to_3_stride, layer1_to_3_kernel = layer1_to_3_stride[1:], layer1_to_3_kernel[1:]
layer3_shape = htm.conv_out_size(layer2_shape[1:], layer2_to_3_stride, layer2_to_3_kernel)
layer3_shape[0] = layer3_neurons_per_column
layer3_size = prod(layer3_shape)
layer1_to_3_pop = htm.Population()
layer1_to_3_pop.push_add_2d_column_grid_with_3d_input(
    input_range=(0, layer1_size),
    neurons_per_column=layer3_neurons_per_column,
    segments_per_neuron=1,
    synapses_per_segment=30,
    stride=layer1_to_3_stride,
    kernel=layer1_to_3_kernel,
    input_size=layer1_shape,
    rand_seed=43643908
)

layer4_shape = [4, layer3_size, 1]
layer4_size = prod(layer4_shape)
layer3_to_4_pop = htm.Population(layer4_size)
layer3_to_4_pop.add_uniform_rand_inputs_from_range((layer1_size + layer2_size, layer1_size + layer2_size + layer3_size),
                                                   30)

input_shapes = [layer1_shape, layer2_shape, layer3_shape, layer4_shape]
output_shapes = [layer2_shape, layer3_shape, layer4_shape, [0, 0, 0]]

layer1_enc = [enc.add_bits(S) for _ in GABOR_FILTERS]
layer2_enc = enc.add_bits(layer2_size)
layer3_enc = enc.add_bits(layer3_size)
# layer4_enc = enc.add_bits(layer4_size)

htm1 = htm.CpuHTM2(enc.input_size, 30, layer1_to_2_pop)
# htm1.visualise(input_shapes, [layer2_shape, [0, 0, 0], [0, 0, 0]])

htm2 = htm.CpuHTM2(enc.input_size, 30, layer2_to_3_pop * layer1_to_3_pop)
# htm2.visualise(input_shapes, [[0, 0, 0], layer3_shape, [0, 0, 0]])

htm3 = htm.CpuHTM2(enc.input_size, 30, layer3_to_4_pop)
htm3.set_all_permanences(1.)

htm4 = htm.CpuBigHTM(input_size=layer4_size, minicolumns=10 * 28, n=4)

bitset = htm.CpuBitset(enc.input_size)
MNIST, LABELS = torch.load('htm/data/mnist.pt')


def train(samples, repetitions, train_map):
    for _ in range(repetitions):
        for img, lbl in tqdm(zip(MNIST[:samples], LABELS[:samples]), desc="training", total=samples):
            img = img.type(torch.float) / 255
            bitset.clear()
            for kernel, enc in zip(GABOR_FILTERS, layer1_enc):
                i = ndimage.convolve(img, kernel, mode='constant')
                i = i.clip(0, 1)
                i = i > 0.8
                enc.encode(bitset, i)
            # htm1.visualise(input_shapes, [layer2_shape, [0, 0, 0], [0, 0, 0]], input=bitset.to_sdr(),
            # input_cell_margin=0.4)
            layer2_activity = htm1(bitset, True)
            layer2_enc.encode(bitset, layer2_activity)
            # htm2.visualise(input_shapes, [[0, 0, 0], layer3_shape, [0, 0, 0]], input=bitset.to_sdr(),
            #                input_cell_margin=0.4)
            layer3_activity = htm2(bitset, True)
            layer3_enc.encode(bitset, layer3_activity)
            # htm3.visualise(input_shapes, [[0, 0, 0], layer3_shape, [0, 0, 0]], input=bitset.to_sdr(),
            #                input_cell_margin=0.4)
            if train_map:
                layer4_activity = htm3(bitset, False)
                layer4_activity = layer4_activity.to_input(layer4_size)
                # htm3.visualise(input_shapes, [[0, 0, 0], [0, 0, 0], layer4_shape, [0, 0, 0]], input=bitset.to_sdr(),
                #                input_cell_margin=0.4)
                layer5_activity = htm4.infer_sticky(layer4_activity,)


train(100, 1, False)
train(100, 1, True)
