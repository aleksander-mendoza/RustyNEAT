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

enc = rusty_neat.htm.EncoderBuilder()
img_enc = [enc.add_bits(28 * 28) for _ in GABOR_FILTERS]
lbl_enc = enc.add_categorical(10, 28 * 8)
sdr = rusty_neat.htm.CpuSDR()
hom = None
MNIST, LABELS = torch.load('htm/data/mnist.pt')


def generate_htm():
    global hom
    hom = rusty_neat.htm.CpuHOM(1, enc.input_size)


def encode_img(img):
    img = img.type(torch.float) / 255
    for kernel, enc in zip(GABOR_FILTERS, img_enc):
        i = ndimage.convolve(img, kernel, mode='constant')
        i = i.clip(0, 1)
        i = i > 0.8
        # plt.imshow(i)
        # plt.show()
        i = i.reshape(28 * 28)
        i = i.tolist()
        enc.encode(sdr, i)


def infer(img, lbl=None):
    sdr.clear()
    encode_img(img)
    predicted_columns = hom(sdr, lbl is not None)
    if lbl is not None:
        sdr.clear()
        lbl_enc.encode(sdr, lbl)
        predicted_columns = hom(sdr, True)
    hom.reset()
    assert predicted_columns.is_normalized()
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


run(10,1,100,100)

# Encoding:
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   lbl_enc = enc.add_categorical(10, 28 * 8)
#   hom = rusty_neat.htm.CpuHOM(1, enc.input_size)
# Ensemble accuracy(2,20,100): 0.54
# Ensemble accuracy(2,1 ,100): 0.54
# Ensemble accuracy(10,1,100): 0.53
# Ensemble accuracy(10,1,100,100): 0.36
# Ensemble accuracy(10,1,100,500):
# Ensemble accuracy(1,1,100,500):
# Ensemble accuracy(10,1,500,500):
# Ensemble accuracy(1,1,500,500):
# Ensemble accuracy(1,1,1000,1000):
# Ensemble accuracy(1,1,5000,1000):
