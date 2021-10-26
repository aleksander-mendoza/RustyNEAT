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

out_columns = 28 * (28 + 10 * 4)
htm_enc = rusty_neat.htm.EncoderBuilder()
S = 28 * 28
img_enc = [htm_enc.add_bits(S) for _ in GABOR_FILTERS]
hom_enc = rusty_neat.htm.EncoderBuilder()
out_enc = hom_enc.add_bits(out_columns)
lbl_enc = hom_enc.add_categorical(10, 28 * 4)
bitset = rusty_neat.htm.CpuBitset(htm_enc.input_size)
sdr = rusty_neat.htm.CpuSDR()
htm1 = None
hom = None
MNIST, LABELS = torch.load('htm/data/mnist.pt')


def generate_htm():
    global htm1
    global hom
    htm1 = rusty_neat.htm.CpuHTM2(htm_enc.input_size, out_columns, 30, 28 * 4)
    hom = rusty_neat.htm.CpuHOM(1, hom_enc.input_size)


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


def infer(img, lbl=None):
    bitset.clear()
    encode_img(img)
    active_columns = htm1(bitset, lbl is not None)
    predicted_columns = hom(active_columns, lbl is not None)
    if lbl is not None:
        lbl_enc.encode(sdr, lbl)
        predicted_columns = hom(sdr, True)
        sdr.clear()
    hom.reset()
    assert predicted_columns.is_normalized()
    return predicted_columns


def train(samples, repetitions):
    for _ in range(repetitions):
        for img, lbl in tqdm(zip(MNIST[:samples], LABELS[:samples]), desc="training", total=samples):
            infer(img, lbl)


def test(img):
    predicted = infer(img)
    overlap = [0] * 10
    for lbl in range(0, 9):
        lbl_enc.encode(sdr, lbl)
        overlap[lbl] = predicted.overlap(active_columns)
        sdr.clear()
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
# Evaluation: 302 / 600 = 0.5, 663 / 1600 = 0.414375
# Ensemble accuracy(100,2,20): 0.422, 0.401


# Encoding:
#   GABOR_FILTERS = [np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float)]
# Configuration:
#   lbl_enc = enc.add_categorical(10, 28 * 8)
#   bitset = rusty_neat.htm.CpuBitset(enc.input_size)
#   htm = rusty_neat.htm.CpuHTM2(enc.input_size, 28 * (28 + 10 * 4), 30, 28 * 4)
# Ensemble accuracy(100,2,20): 0.205

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
# Ensemble accuracy(100,2,20): 0.304


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
# Ensemble accuracy(100,2,20): 0.363


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
# Ensemble accuracy(100,2,20): 0.239

# Encoding:
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   lbl_enc = enc.add_categorical(10, 28 * 8)
#   bitset = rusty_neat.htm.CpuBitset(enc.input_size)
#   htm = rusty_neat.htm.CpuHTM4(enc.input_size, 28 * (28 + 10 * 4), 30, 28 * 4, 0.5)
# Evaluation: 306 / 600 = 0.51, 645 / 1600 = 0.403125
# Ensemble accuracy(100,2,20): 0.352
