import rusty_neat
from rusty_neat import ndalgebra as nd
import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import numpy as np

MNIST, LABELS = torch.load('htm/data/mnist.pt')

PATCH_SIZE = np.array([5, 5])
POP_SIZE2 = 5
POP_SIZE1 = 20
sep_sparsity = 5
wExc = np.random.rand(PATCH_SIZE.prod(), POP_SIZE1)
wExc = wExc / wExc.sum(0)
wInInh = np.random.rand(PATCH_SIZE.prod(), POP_SIZE2)
wInInh = wInInh / wInInh.sum(0)
wInh = np.random.rand(POP_SIZE2, POP_SIZE1)
wInh = wInh / wInh.sum(0)


def rand_patch():
    r = np.random.rand(2)
    img = MNIST[int(np.random.rand() * len(MNIST))]
    left_bottom = (img.shape - PATCH_SIZE) * r
    left_bottom = left_bottom.astype(int)
    top_right = left_bottom + PATCH_SIZE
    img = img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]]
    return np.float32(img / 255 > 0.8)


sep_threshold = 0.4
epsilon = 0.0001
activity_epsilon = 0.0001
activity_regen_epsilon = activity_epsilon / POP_SIZE2
max_k = 6
threshold = 0.30  # max_k/PATCH_SIZE.prod()
inh_threshold = 0.2
indices = np.array(list(range(PATCH_SIZE.prod())))
indices_out = np.array(list(range(POP_SIZE1)))


def preprocess_x(x):
    x = x.reshape(-1)
    to_drop = int(max(x.sum() - max_k, 0))
    if to_drop > 0:
        i = indices[x > 0]
        np.random.shuffle(i)
        i = i[:to_drop]
        x = x.copy()
        x[i] = 0
    return x


def ltp(x, W, y):
    column = W[:, y > 0].T
    if column.size > 0:
        column += epsilon * x
        W[:, y > 0] = column.T / column.sum(1)


def iltp(x, W, s):
    s = s.copy()
    s[s>threshold] = 0
    column = W[:, x > 0].T
    if column.size > 0:
        column += epsilon * s
        W[:, x > 0] = column.T / column.sum(1)


def spike(x, W):
    s = x @ W
    y = np.float32(s > threshold)
    return y


def run(x, learn):
    x = preprocess_x(x)
    sInInh = x @ wInInh
    inh_y = np.float32(sInInh > inh_threshold)
    sExc = x @ wExc
    sInh = inh_y @ wInh
    s = sExc - sInh
    exc_y = np.float32(s > threshold)
    if learn:
        ltp(x, wInInh, inh_y)
        ltp(x, wExc, exc_y)
        iltp(inh_y, wInh, s)
    return exc_y


w, h = 5, 4
fig, axs = plt.subplots(w, h)
for a in axs:
    for b in a:
        b.set_axis_off()
test_patches = [rand_patch() for _ in range(20000)]
test_patches = [img for img in test_patches if img.sum() > 1]
for idx in range(2000000):
    img = rand_patch()
    if img.sum() > 1:
        out = run(img, True)
    if idx % 2000 == 0:
        stats = np.zeros((POP_SIZE2, PATCH_SIZE[0], PATCH_SIZE[1]))
        probability = np.zeros(POP_SIZE2)
        for img in test_patches:
            out = run(img, False)
            for i in indices_out[out > 0]:
                probability[i] += 1
                stats[i] += img
        print(probability)
        print(probability / probability.sum())
        # print(stats)
        for i in range(w):
            for j in range(h):
                axs[i, j].imshow(stats[i + j * w])
        plt.pause(0.01)

plt.show()
