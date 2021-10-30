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
hom_enc = rusty_neat.htm.EncoderBuilder()
out_enc = hom_enc.add_bits(out_columns)
lbl_enc = hom_enc.add_categorical(10, 28 * 8)
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


def train(repetitions, begin, end):
    for _ in range(repetitions):
        for img, lbl in tqdm(zip(MNIST[begin:end], LABELS[begin:end]), desc="training", total=end - begin):
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


run(10, 20, 500, 500)

# Encoding:
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   out_columns = 28 * (28 + 10 * 4)
#   lbl_enc = hom_enc.add_categorical(10, 28 * 4)
#   htm1 = rusty_neat.htm.CpuHTM2(htm_enc.input_size, out_columns, 30, 28 * 4)
#   hom = rusty_neat.htm.CpuHOM(1, hom_enc.input_size)
# Ensemble accuracy(2,20,100): 0.4629
# Ensemble accuracy(2,20,200): 0.59675
# Ensemble accuracy(4,20,100): 0.6155
# Ensemble accuracy(2,20,400): 0.6712
# Ensemble accuracy(8,20,100): 0.7105


# Encoding:
#   GABOR_FILTERS = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)]
# Configuration:
#   out_columns = 28 * (28 + 10 * 4)
#   lbl_enc = hom_enc.add_categorical(10, 28 * 8)
#   htm1 = rusty_neat.htm.CpuHTM2(htm_enc.input_size, out_columns, 30, 28 * 4)
#   hom = rusty_neat.htm.CpuHOM(1, hom_enc.input_size)
# Ensemble accuracy(2,20,100): 0.493
# Ensemble accuracy(1,1,200): 0.8699999999999999
# [[0.105 0.015 0.    0.    0.    0.    0.    0.    0.    0.115]
#  [0.    0.115 0.    0.    0.    0.    0.    0.    0.    0.   ]
#  [0.    0.    0.1   0.    0.    0.    0.    0.    0.    0.   ]
#  [0.    0.    0.    0.105 0.    0.    0.    0.    0.    0.   ]
#  [0.    0.    0.    0.    0.105 0.    0.    0.    0.    0.   ]
#  [0.    0.    0.    0.    0.    0.065 0.    0.    0.    0.   ]
#  [0.    0.    0.    0.    0.    0.    0.095 0.    0.    0.   ]
#  [0.    0.    0.    0.    0.    0.    0.    0.105 0.    0.   ]
#  [0.    0.    0.    0.    0.    0.    0.    0.    0.075 0.   ]
#  [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]]
# Ensemble accuracy(1,1,100,100): 0.22
# [[0.08 0.07 0.14 0.08 0.1  0.08 0.05 0.08 0.06 0.12]
#  [0.   0.05 0.   0.   0.   0.   0.   0.   0.   0.  ]
#  [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
#  [0.   0.   0.   0.02 0.   0.   0.   0.   0.   0.  ]
#  [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
#  [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
#  [0.   0.   0.   0.   0.   0.   0.03 0.   0.   0.  ]
#  [0.   0.   0.   0.   0.   0.   0.   0.03 0.   0.  ]
#  [0.   0.   0.   0.   0.   0.   0.   0.   0.01 0.  ]
#  [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]]
# Ensemble accuracy(1,1,1000,1000): 0.34900000000000003
# [[0.088 0.014 0.085 0.064 0.078 0.069 0.057 0.041 0.051 0.067]
#  [0.    0.069 0.    0.    0.    0.    0.    0.    0.    0.   ]
#  [0.    0.    0.007 0.001 0.    0.    0.    0.    0.002 0.   ]
#  [0.    0.    0.    0.022 0.    0.003 0.    0.    0.    0.002]
#  [0.003 0.    0.001 0.    0.027 0.004 0.008 0.002 0.001 0.012]
#  [0.    0.    0.    0.004 0.    0.005 0.    0.    0.001 0.   ]
#  [0.002 0.    0.001 0.001 0.001 0.002 0.041 0.    0.001 0.   ]
#  [0.001 0.019 0.002 0.    0.003 0.    0.    0.062 0.001 0.018]
#  [0.    0.002 0.003 0.006 0.    0.005 0.    0.002 0.028 0.011]
#  [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]]
# Ensemble accuracy(8,20,100,100): 0.3535
# [[0.08   0.027  0.1315 0.0575 0.068  0.072  0.033  0.0505 0.0435 0.0955]
#  [0.     0.093  0.0005 0.     0.001  0.005  0.     0.0045 0.0035 0.    ]
#  [0.     0.     0.0065 0.     0.0005 0.0005 0.0005 0.     0.     0.    ]
#  [0.     0.     0.0005 0.024  0.     0.002  0.     0.     0.0005 0.    ]
#  [0.     0.     0.     0.     0.0265 0.     0.     0.     0.     0.005 ]
#  [0.     0.     0.     0.     0.     0.0005 0.     0.     0.     0.    ]
#  [0.     0.     0.     0.     0.     0.     0.0465 0.     0.     0.    ]
#  [0.     0.     0.     0.     0.0035 0.     0.     0.054  0.     0.018 ]
#  [0.     0.     0.001  0.0185 0.0005 0.     0.     0.001  0.0225 0.0015]
#  [0.     0.     0.     0.     0.     0.     0.     0.     0.     0.    ]]
# Ensemble accuracy(8,20,200,200): 0.4235
# [[0.0865  0.0115  0.10075 0.026   0.06975 0.07175 0.04175 0.053   0.031   0.074  ]
#  [0.      0.11475 0.00125 0.      0.      0.      0.00125 0.00425 0.003   0.     ]
#  [0.00025 0.0015  0.00525 0.0065  0.00025 0.0005  0.00075 0.0025  0.004   0.00075]
#  [0.      0.00075 0.0005  0.058   0.      0.00375 0.      0.00025 0.0105  0.003  ]
#  [0.      0.00025 0.00075 0.      0.0445  0.      0.      0.0015  0.      0.02325]
#  [0.0025  0.00025 0.0005  0.0075  0.      0.00725 0.      0.00025 0.0025  0.     ]
#  [0.00075 0.      0.00025 0.      0.      0.0015  0.04075 0.      0.0005  0.00025]
#  [0.      0.00025 0.0005  0.      0.0005  0.      0.00025 0.04325 0.00025 0.00225]
#  [0.      0.00075 0.00025 0.002   0.      0.00025 0.00025 0.      0.02325 0.0015 ]
#  [0.      0.      0.      0.      0.      0.      0.      0.      0.      0.     ]]
# Ensemble accuracy(10,20,500,500): 0.4300
# Ensemble accuracy(1,1,5000,1000): 0.0956
# [[0.019  0.0056 0.0118 0.0092 0.0086 0.0104 0.0076 0.0052 0.0066 0.0086]
#  [0.     0.0158 0.     0.     0.0002 0.     0.     0.     0.     0.    ]
#  [0.     0.     0.0046 0.0008 0.0006 0.     0.     0.0002 0.     0.0004]
#  [0.     0.     0.0012 0.0098 0.     0.0008 0.0002 0.     0.0014 0.0002]
#  [0.     0.     0.0002 0.     0.0068 0.     0.     0.0002 0.     0.0044]
#  [0.0032 0.     0.0002 0.0022 0.     0.004  0.0004 0.     0.0016 0.    ]
#  [0.0002 0.0002 0.0002 0.     0.0002 0.0002 0.0132 0.     0.     0.    ]
#  [0.     0.     0.     0.     0.001  0.     0.     0.0146 0.0004 0.0074]
#  [0.0002 0.     0.0004 0.001  0.0002 0.0006 0.     0.     0.0078 0.0002]
#  [0.     0.     0.     0.     0.     0.     0.     0.     0.     0.    ]]
