import rusty_neat
from rusty_neat import ndalgebra as nd
import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

enc = rusty_neat.htm.EncoderBuilder()
S = 28 * 28
img_enc = enc.add_bits(S)
lbl_enc = enc.add_categorical(10, 28 * 8)
bitset = rusty_neat.htm.CpuBitset(enc.input_size)
htm = None
MNIST, LABELS = torch.load('htm/data/mnist.pt')


def generate_htm():
    global htm
    global hom
    htm = rusty_neat.htm.CpuHTM2(enc.input_size, 28 * (28 + 10 * 4), 30, 28 * 4)
    htm = rusty_neat.htm.CpuHTM2(enc.input_size, 28 * (28 + 10 * 4), 30, 28 * 4)


def encode(img, lbl):
    img = img.reshape(S)
    img = img.clamp(0, 1).type(torch.bool).tolist()
    img_enc.encode(bitset, img)
    lbl_enc.encode(bitset, lbl)


def train(samples, repetitions):
    for _ in range(repetitions):
        for img, lbl in tqdm(zip(MNIST[:samples], LABELS[:samples]), desc="training", total=samples):
            bitset.clear()
            encode(img, lbl)
            active_columns = htm(bitset, True)


def test(img):
    bitset.clear()
    img = img.reshape(S)
    img = img.clamp(0, 1).type(torch.bool).tolist()
    img_enc.encode(bitset, img)
    # lbl_enc.encode(bitset, 1)
    active_columns_no_lbl = htm(bitset)
    overlap = [0] * 10
    for lbl in range(0, 9):
        lbl_enc.clear(bitset)
        lbl_enc.encode(bitset, lbl)
        active_columns = htm(bitset)
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
    return avg/trials


print("Ensemble accuracy:", random_trials(100, 2, 20))

# Configuration:
# lbl_enc = enc.add_categorical(10, 28 * 8)
# bitset = rusty_neat.htm.CpuBitset(enc.input_size)
# htm = rusty_neat.htm.CpuHTM2(enc.input_size, 28 * (28 + 10 * 4), 30, 28 * 4)
# Evaluation: 35/100 = 0.35, 39/100 = 0.39, 40/100 = 0.4, 48/100 = 0.48, 302 / 600 = 0.5, 663 / 1600 = 0.414375
# Ensemble accuracy: 0.422

# Configuration:
# lbl_enc = enc.add_categorical(10, 28 * 8)
# bitset = rusty_neat.htm.CpuBitset(enc.input_size)
# htm = rusty_neat.htm.CpuHTM4(enc.input_size, 28 * (28 + 10 * 4), 30, 28 * 4, 0.5)
# Evaluation: 25/100 = 0.25, 28/100 = 0.28, 35/100 = 0.35, 43/100 = 0.43, 306 / 600 = 0.51, 645 / 1600 = 0.403125
# Ensemble accuracy: 0.352


