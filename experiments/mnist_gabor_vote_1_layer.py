import rusty_neat
from rusty_neat import ndalgebra as nd
import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import numpy as np


def genGabor(sz, omega, theta, func=np.cos, K=np.pi):
    radius = (int(sz[0] / 2.0), int(sz[1] / 2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))
    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)
    gauss = omega ** 2 / (4 * np.pi * K ** 2) * np.exp(- omega ** 2 / (8 * K ** 2) * (4 * x1 ** 2 + y1 ** 2))
    sinusoid = func(omega * x1) * np.exp(K ** 2 / 2)
    gabor = gauss * sinusoid
    return gabor


# plt.imshow(genGabor((28, 28), 2, np.pi * 0.1, func=np.cos))

GABOR_FILTERS = [genGabor((8, 8), 2, np.pi * x/8, func=np.cos) for x in range(0,8)]

inp_enc = rusty_neat.htm.EncoderBuilder()
S = 28 * 28
img_enc = [inp_enc.add_bits(S) for _ in GABOR_FILTERS]
out_enc = rusty_neat.htm.EncoderBuilder()
lbl_enc = out_enc.add_categorical(10, 1024)
sdr = rusty_neat.htm.CpuSDR()
bitset = rusty_neat.htm.CpuBitset(inp_enc.input_size)
active_columns_bitset = rusty_neat.htm.CpuBitset(out_enc.input_size)
htm = None
MNIST, LABELS = torch.load('htm/data/mnist.pt')
shuffle = torch.randperm(len(MNIST))
MNIST = MNIST[shuffle]
LABELS = LABELS[shuffle]


def generate_htm():
    global htm
    htm = rusty_neat.htm.CpuHTM2(inp_enc.input_size, out_enc.input_size, lbl_enc.sdr_cardinality, int(inp_enc.input_size * 0.8))


def encode_img(img, visualize=False):
    img = img.type(torch.float) / 255
    if visualize:
        fig, axs = plt.subplots(len(GABOR_FILTERS) + 1)
        axs[0].imshow(img)
    for k, (kernel, enc) in enumerate(zip(GABOR_FILTERS, img_enc)):
        i = ndimage.convolve(img, kernel, mode='constant')
        i = i > 5
        if visualize:
            axs[1+k].imshow(i)
        i = i.reshape(S)
        i = i.tolist()
        enc.encode(bitset, i)
    if visualize:
        plt.show()


def infer(img, lbl=None):
    bitset.clear()
    encode_img(img)
    if lbl is not None:
        lbl_enc.encode(sdr, lbl)
        htm.update_permanence(sdr, bitset)
        sdr.clear()
    else:
        predicted_columns = htm.compute(bitset)
        return predicted_columns


def train(repetitions, begin, end):
    for _ in range(repetitions):
        for img, lbl in tqdm(zip(MNIST[begin:end], LABELS[begin:end]), desc="training", total=end - begin):
            infer(img, lbl)


def test(img):
    predicted = infer(img)
    return lbl_enc.find_category_with_highest_overlap(predicted)


def eval(begin, end):
    confusion_matrix = np.zeros((10, 10))
    for img, lbl in tqdm(zip(MNIST[begin:end], LABELS[begin:end]), desc="evaluation", total=end - begin):
        guessed = test(img)
        confusion_matrix[guessed, lbl] += 1
    return confusion_matrix


def random_trials(repetitions, trials, samples, test_samples):
    confusion_matrix_avg = np.zeros((10, 10))
    for _ in tqdm(range(trials), desc="trial", total=trials):
        generate_htm()
        train(repetitions, 0, samples)
        correct = eval(0, samples) if test_samples is None else eval(samples, samples + test_samples)
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


# Encoding:
#   GABOR_FILTERS = [genGabor((8, 8), 2, np.pi * x/8, func=np.cos) for x in range(0,8)]
# Configuration:
#     lbl_enc = out_enc.add_categorical(10, 1024)
#     htm = rusty_neat.htm.CpuHTM4(inp_enc.input_size, out_enc.input_size, lbl_enc.sdr_cardinality,
#                                  int(inp_enc.input_size * 0.8), 0.2)
#     htm.permanence_decrement = -0.002
#     htm.permanence_increment = 0.01
# Voting mechanism:
#     if lbl is not None:
#         lbl_enc.encode(sdr, lbl)
#         htm.update_permanence(sdr, bitset)
#         sdr.clear()
#     else:
#         predicted_columns = htm.compute(bitset)
#         return predicted_columns
# Ensemble accuracy(2,20,100): 0.5524
# Ensemble accuracy(2,20,200):
# Ensemble accuracy(4,20,100): 0.6255
# Ensemble accuracy(2,20,400):
# Ensemble accuracy(8,20,100):
# Ensemble accuracy(32,20,100):
# Ensemble accuracy(64,1,100): 0.1500
# Ensemble accuracy(128,1,100):
# Ensemble accuracy(512,1,100):


# Encoding:
#   GABOR_FILTERS = [genGabor((8, 8), 2, np.pi * x/8, func=np.cos) for x in range(0,8)]
# Configuration:
#     lbl_enc = out_enc.add_categorical(10, 1024)
#     htm = rusty_neat.htm.CpuHTM2(inp_enc.input_size, out_enc.input_size, lbl_enc.sdr_cardinality, int(inp_enc.input_size * 0.8))
# Voting mechanism: same as above
# Ensemble accuracy(2,20,100):
# Ensemble accuracy(2,20,200):
# Ensemble accuracy(4,20,100): 0.704
# Ensemble accuracy(4,20,100,100): 0.5994
# Ensemble accuracy(2,20,400):
# Ensemble accuracy(8,20,100): 0.7585
# Ensemble accuracy(32,1,100): 0.74
# Ensemble accuracy(64,1,100): 0.73
# Ensemble accuracy(128,1,100):
# Ensemble accuracy(512,1,100):



run(4,20,100,100)
