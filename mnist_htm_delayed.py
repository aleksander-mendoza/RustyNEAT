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
        i*=e
    return i

layer1 = [enc.add_bits(S) for _ in GABOR_FILTERS]
layer2 = enc.add_bits(13 * 13 * 8)
layer1_shape = [len(GABOR_FILTERS), 28, 28]
layer1_size = prod(layer1_shape)
layer2_shape = [8, 13, 13]
layer2_size = prod(layer2_shape)
layer3_shape = [8, 13, 13]
layer3_size = prod(layer3_shape)
input_shapes = [layer1_shape, layer2_shape]
htm1 = htm.CpuHTM2(enc.input_size, 30)

htm1.add_2d_column_grid_with_3d_input(minicolumns_per_column=8,
                                      inputs_per_minicolumn=30,
                                      input_stride=[2, 2],
                                      input_kernel=[4, 4],
                                      input_size=input_shapes[0],
                                      input_range=(0, layer1_size),
                                      rand_seed=53463)

htm2 = htm.CpuHTM2(enc.input_size, 30)
htm2.add_2d_column_grid_with_3d_input(minicolumns_per_column=16,
                                      inputs_per_minicolumn=30,
                                      input_stride=[2, 2],
                                      input_kernel=[4, 4],
                                      input_size=input_shapes[0],
                                      input_range=(layer1_size, layer1_size+layer2_size),
                                      rand_seed=53463)

output_shapes = [[13, 13, 8], [0, 0, 0]]

htm1.visualise(input_shapes, output_shapes)

htm2 = htm.CpuHTM2(enc.input_size * 2, 28 * 8)
MNIST, LABELS = torch.load('htm/data/mnist.pt')


def encode_img(img):
    img = img.type(torch.float) / 255
    for kernel, enc in zip(GABOR_FILTERS, img_enc):
        i = ndimage.convolve(img, kernel, mode='constant')
        i = i.clip(0, 1)
        i = i > 0.8
        # plt.imshow(i)
        # plt.show()
        i = i.reshape(S)
        i = i.tolist()
        enc.encode(bitset, i)


def clear_img():
    for enc in img_enc:
        enc.clear(bitset)


def encode(img, lbl):
    encode_img(img)
    lbl_enc.encode(bitset, lbl)


def train(samples, repetitions):
    for _ in range(repetitions):
        for img, lbl in tqdm(zip(MNIST[:samples], LABELS[:samples]), desc="training", total=samples):
            bitset.clear()
            encode(img, lbl)
            active_columns = htm1(bitset, True)
            # active_columns = active_columns.to_bitset(enc.input_size)
            # active_columns = htm2(active_columns, True)


def test(img):
    bitset.clear()
    encode_img(img)
    active_columns_no_lbl = htm1(bitset)
    # active_columns_no_lbl = active_columns_no_lbl.to_bitset(enc.input_size)
    # active_columns_no_lbl = htm2(active_columns_no_lbl)
    overlap = [0] * 10
    for lbl in range(0, 10):
        lbl_enc.clear(bitset)
        lbl_enc.encode(bitset, lbl)
        active_columns = htm1(bitset)
        # active_columns = active_columns.to_bitset(enc.input_size)
        # active_columns = htm2(active_columns)
        overlap[lbl] = active_columns_no_lbl.overlap(active_columns)
    # print(lbl, overlap[lbl])
    return np.argmax(overlap)


def eval(samples):
    correct = 0
    for img, lbl in tqdm(zip(MNIST[:samples], LABELS[:samples]), desc="evaluation", total=samples):
        guessed = test(img)
        if guessed == lbl:
            correct += 1
    return correct


def random_trials(samples, repetitions, trials):
    avg = 0
    for _ in tqdm(range(trials), desc="trial", total=trials):
        generate_htm()
        train(samples, repetitions)
        correct = eval(samples)
        print(correct, "/", samples, "=", correct / samples)
        avg += correct / samples
    return avg / trials


def run(samples, repetitions, trials):
    acc = random_trials(samples, repetitions, trials)
    print("Ensemble accuracy(" + str(samples) + "," + str(repetitions) + "," + str(trials) + "):", acc)


run(100, 2, 20)
