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
    np.array([[0,0,0], [0,1,0], [0,0,0]], dtype=np.float),
    # np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float),
    # np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float),
    # np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float),
    # np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float)
]

enc = rusty_neat.htm.EncoderBuilder()
S = 28 * 28
img_enc = [enc.add_bits(S) for _ in GABOR_FILTERS]
lbl_enc = enc.add_categorical(10, 28 * 4)
bitset = rusty_neat.htm.CpuBitset(enc.input_size)
htm1 = None
htm2 = None
MNIST, LABELS = torch.load('htm/data/mnist.pt')


def generate_htm():
    global htm1
    global htm2
    htm1 = rusty_neat.htm.CpuHTM2(enc.input_size, 28 * (28 + 10 * 4), 30, 28 * 4)
    # htm2 = rusty_neat.htm.CpuHTM2(enc.input_size * 2, 28 * (28 + 10 * 4), 30, 28 * 8)


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
    print("Ensemble accuracy("+str(samples)+","+str(repetitions)+","+str(trials)+"):", acc)


run(100, 2, 20)

# Encoding:
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   lbl_enc = enc.add_categorical(10, 28 * 8)
#   bitset = rusty_neat.htm.CpuBitset(enc.input_size)
#   htm = rusty_neat.htm.CpuHTM2(enc.input_size, 28 * (28 + 10 * 4), 30, 28 * 4)
# Ensemble accuracy(100,2,20):


# Encoding:
#   GABOR_FILTERS = [np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float)]
# Configuration:
#   lbl_enc = enc.add_categorical(10, 28 * 8)
#   bitset = rusty_neat.htm.CpuBitset(enc.input_size)
#   htm = rusty_neat.htm.CpuHTM2(enc.input_size, 28 * (28 + 10 * 4), 30, 28 * 4)
# Ensemble accuracy(100,2,20):


# Encoding:
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   lbl_enc = enc.add_categorical(10, 28 * 8)
#   bitset = rusty_neat.htm.CpuBitset(enc.input_size)
#   htm1 = rusty_neat.htm.CpuHTM2(enc.input_size, enc.input_size, 30, 28 * 4)
#   htm2 = rusty_neat.htm.CpuHTM2(enc.input_size, 28 * (28 + 10 * 4), 30, 28 * 4)
# Network topology:
#   active_columns = htm1(bitset)
#   active_columns = active_columns.to_bitset(enc.input_size)
#   active_columns = htm2(active_columns)
# Ensemble accuracy(100,2,20):


# Encoding:
#   GABOR_FILTERS = [
#       np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float),
#       np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float),
#       np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float),
#       np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float)
#   ]
# Configuration:
#   lbl_enc = enc.add_categorical(10, 28 * 8)
#   bitset = rusty_neat.htm.CpuBitset(enc.input_size)
#   htm = rusty_neat.htm.CpuHTM2(enc.input_size, 28 * (28 + 10 * 4), 30, 28 * 8)
# Ensemble accuracy(100,2,20):


# Encoding:
#   GABOR_FILTERS = [
#       np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float),
#       np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float),
#       np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float),
#       np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float)
#   ]
# Configuration:
#   lbl_enc = enc.add_categorical(10, 28 * 4)
#   bitset = rusty_neat.htm.CpuBitset(enc.input_size)
#   htm = rusty_neat.htm.CpuHTM2(enc.input_size, 28 * (28 + 10 * 4), 30, 28 * 8)
# Ensemble accuracy(100,2,20):


# Encoding:
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   lbl_enc = enc.add_categorical(10, 28 * 8)
#   bitset = rusty_neat.htm.CpuBitset(enc.input_size)
#   htm = rusty_neat.htm.CpuHTM4(enc.input_size, 28 * (28 + 10 * 4), 30, 28 * 4, 0.5)
# Ensemble accuracy(100,2,20):
