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
n = PATCH_SIZE.prod()
w, h = 7, 7
m = h * w
m_neg = 20
epsilon = 0.01


def rand_patch():
    r = np.random.rand(2)
    img = MNIST[int(np.random.rand() * len(MNIST))]
    left_bottom = (img.shape - PATCH_SIZE) * r
    left_bottom = left_bottom.astype(int)
    top_right = left_bottom + PATCH_SIZE
    img = img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]]
    return np.float32(img / 255 > 0.8)


X = [rand_patch() for _ in range(20000)]
X = np.array([x for x in X if x.sum() > 1])


class Weights:

    def __init__(self, n, m, norm=np.linalg.norm):
        self.norm = norm
        W = np.random.rand(n, m)
        self.W = W / self.norm(W, axis=0)

    def __rmatmul__(self, x):
        return x @ self.W

    def learn(self, x, k):
        self.W[:, k] += x * epsilon
        self.W[:, k] /= self.norm(self.W[:, k])


class L2:

    def __init__(self):
        self.norm = np.linalg.norm
        W = np.random.rand(n, m)
        self.W = W / self.norm(W, axis=0)
        self.neg_sign = np.random.permutation(m)[:m_neg]

    def run(self, x, learn):
        x = x.reshape(n)
        s = x @ self.W
        k = s > 0.5
        if learn:
            self.W[:, k] += x * epsilon
            self.W[:, k] /= self.norm(self.W[:, k])
        return k


def calculate_Q(ecc):
    K = np.array([ecc.run(x, False) for x in X])
    p_y = np.bincount(K, minlength=m)
    Q = np.zeros((m, PATCH_SIZE[0], PATCH_SIZE[1]))
    for x, j in zip(X, K):
        Q[j] += x
    for j in range(m):
        if p_y[j] == 0:
            assert Q[j].sum() == 0
        else:
            Q[j] /= p_y[j]

    expected_xQ = np.zeros(m)
    for x, j in zip(X, K):
        expected_xQ[j] = x.reshape(n) @ Q[j].reshape(n)
    for j in range(m):
        if p_y[j] == 0:
            assert expected_xQ[j].sum() == 0
        else:
            expected_xQ[j] /= p_y[j]
    return expected_xQ.sum(), Q


def train(ecc, steps):
    for _ in range(steps):
        x = rand_patch()
        if x.sum() > 1:
            ecc.run(x, True)


def experiment(steps=2000, epochs=20000):
    l1 = L2()

    fig, axs = plt.subplots(h, w)
    for a in axs:
        for b in a:
            b.set_axis_off()

    for _ in range(epochs):
        train(l1, steps)
        ExQ_l1, Q1 = calculate_Q(l1)
        print("expected_xQ_sum=", ExQ_l1)
        for i in range(w):
            for j in range(h):
                axs[j, i].imshow(Q1[i + j * w])
        plt.pause(0.01)


experiment()
