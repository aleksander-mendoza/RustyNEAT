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
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float),
    # np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float),
    # np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float),
    # np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float),
    # np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float)
]

out_columns = 28 * (28 + 10 * 4)
htm_enc = rusty_neat.htm.EncoderBuilder()
S = 28 * 28
img_enc = [htm_enc.add_bits(S) for _ in GABOR_FILTERS]
htm2_enc = rusty_neat.htm.EncoderBuilder()
lbl_enc = htm2_enc.add_categorical(10, 28 * 8)
bitset = rusty_neat.htm.CpuBitset(htm_enc.input_size)
sdr = rusty_neat.htm.CpuSDR()
htm1 = None
htm2 = None
MNIST, LABELS = torch.load('htm/data/mnist.pt')


def generate_htm():
    global htm1
    global htm2
    htm1 = rusty_neat.htm.CpuHTM2(htm_enc.input_size, out_columns, 30, 28 * 4)
    htm2 = rusty_neat.htm.CpuHTM2(out_columns, lbl_enc.len, 28*4, 28 * 4)


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
    htm2_input = active_columns.to_bitset(htm2.input_size)
    if lbl is not None:
        lbl_enc.encode(sdr, lbl)
        htm2.update_permanence(htm2_input, sdr)
        sdr.clear()
    else:
        predicted_columns = htm2.compute(htm2_input)
        assert predicted_columns.is_normalized()
        return predicted_columns


def train(repetitions, begin, end):
    for _ in range(repetitions):
        for img, lbl in tqdm(zip(MNIST[begin:end], LABELS[begin:end]), desc="training", total=end-begin):
            infer(img, lbl)


def test(img):
    predicted = infer(img)
    overlap = [0] * 10
    for lbl in range(0, 9):
        lbl_enc.encode(sdr, lbl)
        overlap[lbl] = predicted.overlap(sdr)
        sdr.clear()
    # print(lbl, overlap[lbl])
    return np.argmax(overlap)


def eval(begin, end):
    confusion_matrix = np.zeros((10, 10))
    for img, lbl in tqdm(zip(MNIST[begin:end], LABELS[begin:end]), desc="evaluation", total=end-begin):
        guessed = test(img)
        confusion_matrix[guessed, lbl] += 1
    return confusion_matrix


def random_trials(repetitions, trials, samples, test_samples):
    confusion_matrix_avg = np.zeros((10, 10))
    for _ in tqdm(range(trials), desc="trial", total=trials):
        generate_htm()
        train(repetitions, 0, samples)
        correct = eval(0, samples) if test_samples is None else eval(samples, samples+test_samples)
        if test_samples is not None:
            samples = test_samples
        print(sum(correct.diagonal()), "/", samples, "=", sum(correct.diagonal()) / samples)
        confusion_matrix_avg += correct
    return confusion_matrix_avg / (trials * samples)


def run(repetitions, trials, samples, test_samples=None):
    acc = random_trials(repetitions, trials, samples, test_samples)
    s = "," + str(test_samples) if test_samples is not None else ""
    print("Ensemble accuracy(" + str(repetitions) + "," + str(trials) + "," + str(samples) + s + "):",
          sum(acc.diagonal()))
    print(acc)


run(1,1,10000,1000)

# Encoding:
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   out_columns = 28 * (28 + 10 * 4)
#   lbl_enc = htm2_enc.add_categorical(10, 28 * 8)
#   htm1 = rusty_neat.htm.CpuHTM2(htm_enc.input_size, out_columns, 30, 28 * 4)
#   htm2 = rusty_neat.htm.CpuHTM2(out_columns, lbl_enc.len, 28*4, 28 * 4)
# Ensemble accuracy(2,20,100): 0.2235
# Ensemble accuracy(2,20,200): 0.3405
# Ensemble accuracy(4,20,100): 0.2680
# Ensemble accuracy(2,20,400): 0.3223
# Ensemble accuracy(8,20,100): 0.3680


