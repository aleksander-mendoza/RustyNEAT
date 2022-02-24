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
    None
    # np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float),
    # np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float),
    # np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float),
    # np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float)
]

inp_enc = rusty_neat.htm.EncoderBuilder()
S = 28 * 28
img_enc = [inp_enc.add_bits(S) for _ in GABOR_FILTERS]
out_enc = rusty_neat.htm.EncoderBuilder()
lbl_enc = out_enc.add_categorical(10, 1024)
bitset = rusty_neat.htm.CpuBitset(inp_enc.input_size)
active_columns_bitset = rusty_neat.htm.CpuBitset(out_enc.input_size)
sdr = rusty_neat.htm.CpuSDR()
htm = None
MNIST, LABELS = torch.load('htm/data/mnist.pt')
shuffle = torch.randperm(len(MNIST))
MNIST = MNIST[shuffle]
LABELS = LABELS[shuffle]
# M5 = torch.zeros((28,28))
# for m,l in zip(MNIST,LABELS):
#     if l == 7:
#         M5 += m
#
# plt.imshow(M5/sum(sum(M5)))
# plt.show()

def generate_htm():
    global htm
    htm = rusty_neat.htm.CpuHTM4(inp_enc.input_size,out_enc.input_size,lbl_enc.sdr_cardinality,int(inp_enc.input_size*0.8), 0.2)


def encode_img(img):
    img = img.type(torch.float) / 255
    for kernel, enc in zip(GABOR_FILTERS, img_enc):
        i = img
        if kernel is not None:
            i = ndimage.convolve(i, kernel, mode='constant')
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
    if lbl is not None:
        lbl_enc.encode(sdr, lbl)
        htm.update_permanence(sdr, bitset)
        # htm.update_permanence_ltd(predicted_columns, active_columns_bitset, bitset)
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
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   lbl_enc = out_enc.add_categorical(10, 1024)
#   htm = rusty_neat.htm.cpu_htm4_new_globally_uniform_prob(inp_enc.input_size, out_enc.input_size,
#        out_enc.sdr_cardinality, int(inp_enc.input_size*0.3), 0.2)
# Voting mechanism:
#     if lbl is not None:
#         lbl_enc.encode(sdr, lbl)
#         htm.update_permanence(sdr, bitset)
#         sdr.clear()
#     else:
#         predicted_columns = htm.compute(bitset)
#         return predicted_columns
# Ensemble accuracy(2,20,100): 0.5585
# Ensemble accuracy(2,20,200):
# Ensemble accuracy(4,20,100): 0.5409
# Ensemble accuracy(2,20,400):
# Ensemble accuracy(8,20,100): 0.49900
# Ensemble accuracy(32,20,100): 0.484
# Ensemble accuracy(64,1,100):
# Ensemble accuracy(128,1,100):
# Ensemble accuracy(512,1,100):


# Encoding:
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   lbl_enc = out_enc.add_categorical(10, 1024)
#   htm = rusty_neat.htm.cpu_htm4_new_globally_uniform_prob(inp_enc.input_size, out_enc.input_size,
#        out_enc.sdr_cardinality, int(inp_enc.input_size*0.3), 0.2)
# Voting mechanism:
#     predicted_columns = htm.compute(bitset)
#     if lbl is not None:
#         lbl_enc.encode(sdr, lbl)
#         htm.update_permanence_ltd(predicted_columns, sdr, bitset)
#         sdr.clear()
#     return predicted_columns
# Ensemble accuracy(2,20,100): 0.3575
# Ensemble accuracy(2,20,200):
# Ensemble accuracy(4,20,100):
# Ensemble accuracy(2,20,400):
# Ensemble accuracy(8,20,100): 0.5775
# Ensemble accuracy(16,20,100): 0.607
# Ensemble accuracy(32,20,100): 0.506
# Ensemble accuracy(64,1,100): 0.41
# Ensemble accuracy(128,1,100):
# Ensemble accuracy(512,1,100):
# Ensemble accuracy(1,1,1000): 0.3490000000000001


# Encoding:
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   lbl_enc = out_enc.add_categorical(10, 1024)
#   htm = rusty_neat.htm.CpuHTM4(inp_enc.input_size, out_enc.input_size,
#        out_enc.sdr_cardinality, int(inp_enc.input_size*0.3), 0.2)
# Voting mechanism:
#     predicted_columns = htm.compute(bitset)
#     if lbl is not None:
#         lbl_enc.encode(sdr, lbl)
#         htm.update_permanence_ltd(predicted_columns, sdr, bitset)
#         sdr.clear()
#     return predicted_columns
# Ensemble accuracy(2,20,100): 0.1885
# Ensemble accuracy(2,20,200):
# Ensemble accuracy(4,20,100): 0.292
# Ensemble accuracy(2,20,400):
# Ensemble accuracy(8,20,100): 0.5145
# Ensemble accuracy(16,20,100): 0.7755
# Ensemble accuracy(32,20,100): 0.72449
# Ensemble accuracy(64,1,100): 0.65
# Ensemble accuracy(128,1,100):
# Ensemble accuracy(512,1,100):
# Ensemble accuracy(1,1,1000):


# Encoding:
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   lbl_enc = out_enc.add_categorical(10, 1024)
#   htm = rusty_neat.htm.CpuHTM4(inp_enc.input_size, out_enc.input_size,
#        out_enc.sdr_cardinality, int(inp_enc.input_size*0.3), 0.2)
# Voting mechanism:
#     if lbl is not None:
#         lbl_enc.encode(active_columns_bitset, lbl)
#         htm.update_permanence_and_penalize(active_columns_bitset, bitset, -0.2)
#         active_columns_bitset.clear()
#     else:
#         predicted_columns = htm.compute(bitset)
#         return predicted_columns
# Ensemble accuracy(2,20,100): 0.3315
# Ensemble accuracy(2,20,200):
# Ensemble accuracy(4,20,100): 0.454
# Ensemble accuracy(2,20,400):
# Ensemble accuracy(8,20,100): 0.659
# Ensemble accuracy(16,20,100): 0.60
# Ensemble accuracy(32,20,100):
# Ensemble accuracy(64,1,100):
# Ensemble accuracy(128,1,100):
# Ensemble accuracy(512,1,100):
# Ensemble accuracy(1,1,1000):


# Encoding:
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   lbl_enc = out_enc.add_categorical(10, 1024)
#   htm = rusty_neat.htm.CpuHTM4(inp_enc.input_size, out_enc.input_size,
#        out_enc.sdr_cardinality, int(inp_enc.input_size*0.3), 0.2)
# Voting mechanism:
#     if lbl is not None:
#         lbl_enc.encode(active_columns_bitset, lbl)
#         htm.update_permanence_and_penalize_thresholded(active_columns_bitset, bitset, 28*2, -0.2)
#         active_columns_bitset.clear()
#     else:
#         predicted_columns = htm.compute(bitset)
#         return predicted_columns
# Ensemble accuracy(2,20,100):
# Ensemble accuracy(2,20,200):
# Ensemble accuracy(4,20,100):
# Ensemble accuracy(2,20,400):
# Ensemble accuracy(8,20,100):
# Ensemble accuracy(16,20,100):
# Ensemble accuracy(32,20,100):
# Ensemble accuracy(64,1,100):
# Ensemble accuracy(128,1,100):
# Ensemble accuracy(512,1,100):
# Ensemble accuracy(1,1,1000):

run(2,1,400,1000)


# Encoding:
#   GABOR_FILTERS = [None]
# Configuration:
#   lbl_enc = out_enc.add_categorical(10, 1024)
#   htm = rusty_neat.htm.CpuHTM4(inp_enc.input_size, out_enc.input_size,
#        out_enc.sdr_cardinality, int(inp_enc.input_size*0.8), 0.2)
# Voting mechanism:
#     predicted_columns = htm.compute(bitset)
#     if lbl is not None:
#         lbl_enc.encode(sdr, lbl)
#         htm.update_permanence_ltd(predicted_columns, sdr, bitset)
#         sdr.clear()
#     return predicted_columns
# Ensemble accuracy(2,1,1000, 1000):  0.55899


# Encoding: same as above
# Configuration:
#   lbl_enc = out_enc.add_categorical(10, 1024)
#   htm = rusty_neat.htm.CpuHTM2(inp_enc.input_size, out_enc.input_size,
#        out_enc.sdr_cardinality, int(inp_enc.input_size*0.8))
# Voting mechanism: same as above
# Ensemble accuracy(2,1,1000, 1000):  0.101


# Encoding: same as above
# Configuration:
#   lbl_enc = out_enc.add_categorical(10, 1024)
#   htm = rusty_neat.htm.CpuHTM4(inp_enc.input_size, out_enc.input_size,
#        out_enc.sdr_cardinality, int(inp_enc.input_size*0.8), 0.2)
# Voting mechanism:
# def infer(img, lbl=None):
#     bitset.clear()
#     encode_img(img)
#     if lbl is not None:
#         lbl_enc.encode(sdr, lbl)
#         htm.update_permanence(sdr, bitset)
#         sdr.clear()
#     else:
#         predicted_columns = htm.compute(bitset)
#         return predicted_columns
# Ensemble accuracy(2,1,1000, 1000):  0.126