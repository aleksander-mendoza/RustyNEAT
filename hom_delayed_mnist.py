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

out_columns = 4096
enc = rusty_neat.htm.EncoderBuilder()
S = 28 * 28
img_enc = [enc.add_bits(S) for _ in GABOR_FILTERS]
out_enc = enc.add_bits(out_columns)
lbl_enc = enc.add_categorical(10, 28 * 8)
bitset = rusty_neat.htm.CpuBitset(enc.input_size)
sdr = rusty_neat.htm.CpuSDR()
htm1 = None
hom = None
MNIST, LABELS = torch.load('htm/data/mnist.pt')


def generate_htm():
    global htm1
    global hom
    htm1 = rusty_neat.htm.CpuHTM4(enc.input_size, out_columns, 28*3, 28 * 4, 0.2)
    hom = rusty_neat.htm.CpuHOM(4, enc.input_size)


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
        enc.encode(sdr, i)


def infer(img, lbl=None):
    bitset.clear()
    sdr.clear()
    encode_img(img)
    active_columns = htm1(bitset, lbl is not None)
    hom(sdr, lbl is not None)
    predicted_columns = hom(active_columns, lbl is not None)
    if lbl is not None:
        sdr.clear()
        lbl_enc.encode(sdr, lbl)
        predicted_columns = hom(sdr, True)
    hom.reset()
    assert predicted_columns.is_normalized()
    return predicted_columns


def train(repetitions, begin, end):
    for _ in range(repetitions):
        for i, (img, lbl) in tqdm(enumerate(zip(MNIST[begin:end], LABELS[begin:end])), desc="training", total=end - begin):
            infer(img, lbl)


def test(img):
    predicted = infer(img)
    return lbl_enc.find_category_with_highest_overlap(predicted)


def eval(begin, end):
    confusion_matrix = np.zeros((10, 10))
    for i, (img, lbl) in tqdm(enumerate(zip(MNIST[begin:end], LABELS[begin:end])), desc="evaluation", total=end - begin):
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

# Encoding:
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   out_columns = 28 * (28 + 10 * 4)
#   lbl_enc = enc.add_categorical(10, 28 * 8)
#   htm1 = rusty_neat.htm.CpuHTM2(enc.input_size, out_columns, 28*3, 28 * 4)
#   hom = rusty_neat.htm.CpuHOM(4, enc.input_size)
# Ensemble accuracy(2,20,100): 0.9924
# Ensemble accuracy(2,20,400): 0.9218
# Ensemble accuracy(1,1,200): 0.99999
# Ensemble accuracy(1,1,200,200): 0.52
# Ensemble accuracy(1,1,100,100): 0.58
# Ensemble accuracy(1,1,1000,1000): 0.522
# Ensemble accuracy(8,20,100,100): 0.516
# Ensemble accuracy(8,20,200,200): 0.53125


# Encoding:
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   out_columns = 4096
#   lbl_enc = enc.add_categorical(10, 28 * 8)
#   htm1 = rusty_neat.htm.CpuHTM2(enc.input_size, out_columns, 28*3, 28 * 4)
#   hom = rusty_neat.htm.CpuHOM(4, enc.input_size)
# Ensemble accuracy(2,20,100):
# Ensemble accuracy(8,20,100):
# Ensemble accuracy(8,20,200,200): 0.52025


# Encoding:
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   out_columns = 28 * (28 + 10 * 4)
#   lbl_enc = hom_enc.add_categorical(10, 28 * 8)
#   htm1 = rusty_neat.htm.CpuHTM4(enc.input_size, out_columns, 28*3, 28 * 4, 0.5)
#   hom = rusty_neat.htm.CpuHOM(4, enc.input_size)
# Ensemble accuracy(2,20,100): 0.9845
# Ensemble accuracy(8,20,100): 0.9925
# Ensemble accuracy(8,20,100,100): 0.5660


# Encoding:
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   out_columns = 4096
#   lbl_enc = enc.add_categorical(10, 28 * 8)
#   htm1 = rusty_neat.htm.CpuHTM4(enc.input_size, out_columns, 28*3, 28 * 4, 0.2)
#   hom = rusty_neat.htm.CpuHOM(4, enc.input_size)
# Ensemble accuracy(2,20,100): 0.9720
# Ensemble accuracy(8,20,100): 0.998
# Ensemble accuracy(8,1,200): 0.994
# Ensemble accuracy(8,20,100,100): 0.5975
# Ensemble accuracy(8,20,200,200): 0.52875
# Ensemble accuracy(1,1,5000,1000): 0.67499
# Ensemble accuracy(1,1,10000,1000): 0.419

run(1,1,400)


