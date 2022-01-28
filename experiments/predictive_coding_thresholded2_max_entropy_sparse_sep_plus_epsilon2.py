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
POP_SIZE2 = 20
POP_SIZE1 = 80
sep_sparsity = 5
w1 = np.zeros((PATCH_SIZE.prod(), POP_SIZE1))
for i in range(POP_SIZE1):
    w1[np.random.permutation(PATCH_SIZE.prod())[:sep_sparsity], i] = 1
w2 = np.random.rand(POP_SIZE1, POP_SIZE2)
w1 = w1 / w1.sum(0)
w2n = w2 / w2.sum(0)
del w2
p2 = np.ones(POP_SIZE2)
k1 = 15


def rand_patch():
    r = np.random.rand(2)
    img = MNIST[int(np.random.rand() * len(MNIST))]
    left_bottom = (img.shape - PATCH_SIZE) * r
    left_bottom = left_bottom.astype(int)
    top_right = left_bottom + PATCH_SIZE
    img = img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]]
    return np.float32(img / 255 > 0.8)


sep_threshold = 0.4
threshold = 0.2
epsilon = 0.0001
activity_epsilon = 0.0001
activity_regen_epsilon = activity_epsilon / POP_SIZE2


def run(x):
    global p2
    s1 = x.reshape(-1) @ w1
    top_k = np.argpartition(s1, -k1)[-k1:]
    top_k = top_k[s1[top_k] >= sep_threshold]
    x2 = np.zeros(POP_SIZE1)
    x2[top_k] = 1
    s2 = x2 @ w2n
    r2 = s2 + p2
    top_1 = r2.argmax()
    if s2[top_1] >= threshold:
        p2[top_1] -= activity_epsilon
    else:
        top_1 = None
    return x2, top_1


def learn(x2, top_1):
    global w2n
    # !!! This approach converges a hell lot faster !!!
    # !!! I thought it would lead to instability but it turns out its very stable after just a few iterations !!!
    w2n[:, top_1] += epsilon * x2
    w2n[:, top_1] = w2n[:, top_1] / w2n[:, top_1].sum(0)


w, h = 5, 4
fig, axs = plt.subplots(w, h)
test_patches = [rand_patch() for _ in range(20000)]
for idx in range(2000000):
    img = rand_patch()
    middle, top = run(img)
    if top is not None:
        learn(middle, top)
    if idx % 2000 == 0:
        stats = np.zeros((POP_SIZE2, PATCH_SIZE[0], PATCH_SIZE[1]))
        probability = np.zeros(POP_SIZE2)
        for img in test_patches:
            _, top = run(img)
            if top is not None:
                probability[top] += 1
                stats[top] += img
        print(probability)
        print(probability/probability.sum())
        # print(stats)
        for i in range(w):
            for j in range(h):
                axs[i, j].imshow(stats[i + j * w])
        plt.pause(0.01)

plt.show()
