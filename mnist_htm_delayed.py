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

S = 28 * 28


# intervals = [(f * S, int(S * 0.8), (f + 1) * S) for f in range(len(GABOR_FILTERS))]

def prod(l):
    i = 1
    for e in l:
        i *= e
    return i


#################
# Configuration #
#################

EXPERIMENT_PROFILE = 1
if EXPERIMENT_PROFILE == 0:
    number_of_samples = 800
    layer2_neurons_per_column = 4
    layer1_to_2_synapses_per_segment = 30
    layer3_neurons_per_column = 4
    layer2_to_3_synapses_per_segment = 30
    layer1_to_3_synapses_per_segment = 30
    layer_4_depth = 4
    categories_cardinality = 28
    # training
    # 49 / 50 = 0.98
    # 99 / 100 = 0.99
    # 185 / 200 = 0.925
    # 328 / 400 = 0.82
    # 589 / 800 = 0.73625
elif EXPERIMENT_PROFILE == 1:
    number_of_samples = 200, 800
    layer2_neurons_per_column = 8
    layer1_to_2_synapses_per_segment = 30
    layer3_neurons_per_column = 8
    layer2_to_3_synapses_per_segment = 30
    layer1_to_3_synapses_per_segment = 30
    layer_4_depth = 4
    categories_cardinality = 28
    # training
    # 186 / 200 = 0.93, 0.89, 0.895
    # 334 / 400 = 0.835
    # 616 / 800 = 0.77
    # validation
    # (200) 255 / 800 = 0.31875
elif EXPERIMENT_PROFILE == 2:
    number_of_samples = 800, 800
    layer2_neurons_per_column = 4
    layer1_to_2_synapses_per_segment = 30
    layer3_neurons_per_column = 4
    layer2_to_3_synapses_per_segment = 30
    layer1_to_3_synapses_per_segment = 30
    layer_4_depth = 4
    categories_cardinality = 50
    # training
    # 190 / 200 = 0.95, 0.955, 0.955, 0.94
    # 357 / 400 = 0.8925, 0.885
    # 593 / 800 = 0.74125
    # validation
    # (200) 281 / 800 = 0.35125
    # (400) 320 / 800 = 0.4
    # (800) 335 / 800 = 0.41875
elif EXPERIMENT_PROFILE == 3:
    number_of_samples = 800, 800
    layer2_neurons_per_column = 4
    layer1_to_2_synapses_per_segment = 30
    layer3_neurons_per_column = 4
    layer2_to_3_synapses_per_segment = 30
    layer1_to_3_synapses_per_segment = 30
    layer_4_depth = 4
    categories_cardinality = 100
    # training
    # 197 / 200 = 0.985
    # 377 / 400 = 0.9425
    # 708 / 800 = 0.885
    # validation
    # 351 / 800 = 0.43875
else:
    print("Unknown experiment profile")
    exit()

layer1_shape = [len(GABOR_FILTERS), 28, 28]
layer1_size = prod(layer1_shape)
layer1_to_2_stride = [2, 2]
layer1_to_2_kernel = [4, 4]
layer1_to_2_pop = htm.Population()
layer1_to_2_pop.push_add_2d_column_grid_with_3d_input(
    input_range=(0, layer1_size),
    neurons_per_column=layer2_neurons_per_column,
    segments_per_neuron=1,
    synapses_per_segment=layer1_to_2_synapses_per_segment,
    stride=layer1_to_2_stride,
    kernel=layer1_to_2_kernel,
    input_size=layer1_shape,
    rand_seed=53463
)
layer2_shape = htm.conv_out_size(layer1_shape[1:], layer1_to_2_stride, layer1_to_2_kernel)
layer2_shape[0] = layer2_neurons_per_column
layer2_size = prod(layer2_shape)
layer2_to_3_stride = [1, 1]
layer2_to_3_kernel = [3, 3]
layer2_to_3_pop = htm.Population()
layer2_to_3_pop.push_add_2d_column_grid_with_3d_input(
    input_range=(layer1_size, layer1_size + layer2_size),
    neurons_per_column=layer3_neurons_per_column,
    segments_per_neuron=1,
    synapses_per_segment=layer2_to_3_synapses_per_segment,
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
    synapses_per_segment=layer1_to_3_synapses_per_segment,
    stride=layer1_to_3_stride,
    kernel=layer1_to_3_kernel,
    input_size=layer1_shape,
    rand_seed=43643908
)

layer4_shape = [layer_4_depth, layer3_size, 1]
layer4_size = prod(layer4_shape)
layer3_to_4_pop = htm.Population(layer4_size)
layer3_to_4_pop.add_uniform_rand_inputs_from_range((layer1_size + layer2_size, layer1_size + layer2_size + layer3_size),
                                                   30)

input_shapes = [layer1_shape, layer2_shape, layer3_shape, layer4_shape]
output_shapes = [layer2_shape, layer3_shape, layer4_shape, [0, 0, 0]]

enc1 = htm.EncoderBuilder()
layer1_enc = [enc1.add_bits(S) for _ in GABOR_FILTERS]
layer2_enc = enc1.add_bits(layer2_size)
layer3_enc = enc1.add_bits(layer3_size)
# layer4_enc = enc.add_bits(layer4_size)
enc5 = htm.EncoderBuilder()
layer5_enc = enc5.add_categorical(10, categories_cardinality)
layer5_shape = [layer5_enc.num_of_categories, layer5_enc.sdr_cardinality]
layer5_size = prod(layer5_shape)
layer5 = []
for category_idx in range(layer5_enc.num_of_categories):
    category_sdr = htm.CpuInput(layer5_size)
    layer5_enc.encode(category_sdr, category_idx)
    layer5.append(category_sdr)

htm1 = htm.CpuHTM2(enc1.input_size, 30, layer1_to_2_pop)
# htm1.visualise(input_shapes, [layer2_shape, [0, 0, 0], [0, 0, 0]])

htm2 = htm.CpuHTM2(enc1.input_size, 30, layer2_to_3_pop * layer1_to_3_pop)
# htm2.visualise(input_shapes, [[0, 0, 0], layer3_shape, [0, 0, 0]])

htm3 = htm.CpuHTM2(enc1.input_size, 30, layer3_to_4_pop)
htm3.set_all_permanences(1.)

htm4 = htm.CpuBigHTM(input_size=layer4_size, minicolumns=layer5_size, n=4)

bitset = htm.CpuBitset(enc1.input_size)
MNIST, LABELS = torch.load('htm/data/mnist.pt')


def train(samples, repetitions, train_cortex, train_map):
    correct_inferences = 0
    for _ in range(repetitions):
        for img, lbl in tqdm(zip(MNIST[samples], LABELS[samples]), desc="training", total=len(samples)):
            img = img.type(torch.float) / 255
            bitset.clear()
            for kernel, enc in zip(GABOR_FILTERS, layer1_enc):
                i = ndimage.convolve(img, kernel, mode='constant')
                i = i.clip(0, 1)
                i = i > 0.8
                enc.encode(bitset, i)
            # htm1.visualise(input_shapes, [layer2_shape, [0, 0, 0], [0, 0, 0]], input=bitset.to_sdr(),
            # input_cell_margin=0.4)
            layer2_activity = htm1(bitset, train_cortex)
            layer2_enc.encode(bitset, layer2_activity)
            # htm2.visualise(input_shapes, [[0, 0, 0], layer3_shape, [0, 0, 0]], input=bitset.to_sdr(),
            #                input_cell_margin=0.4)
            layer3_activity = htm2(bitset, train_cortex)
            layer3_enc.encode(bitset, layer3_activity)
            # htm3.visualise(input_shapes, [[0, 0, 0], layer3_shape, [0, 0, 0]], input=bitset.to_sdr(),
            #                input_cell_margin=0.4)
            layer4_activity = htm3(bitset, False)
            layer4_activity = layer4_activity.to_input(layer4_size)
            # htm3.visualise(input_shapes, [[0, 0, 0], [0, 0, 0], layer4_shape, [0, 0, 0]], input=bitset.to_sdr(),
            #                input_cell_margin=0.4)
            if train_map:
                layer5_activity = htm4.infer_from_whitelist(layer4_activity, layer5[lbl], learn=True)
            else:
                layer5_activity = htm4.infer(layer4_activity, learn=False)
                predicted_lbl = layer5_enc.find_category_with_highest_overlap(layer5_activity)
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
    print(EXPERIMENT_PROFILE, "validation", "("+str(len(training_samples))+")", result_score, "/", number_of_samples,
          "=", result_score / number_of_samples)
