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

w1 = np.random.rand(PATCH_SIZE.prod(), POP_SIZE1)
w2 = np.random.rand(POP_SIZE1, POP_SIZE2)
w1 = w1 / w1.sum(0)
w2n = w2 / w2.sum(0)
k1 = 15


def rand_patch():
    r = np.random.rand(2)
    img = MNIST[int(np.random.rand() * len(MNIST))]
    left_bottom = (img.shape - PATCH_SIZE) * r
    left_bottom = left_bottom.astype(int)
    top_right = left_bottom + PATCH_SIZE
    return img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]]


threshold = 0.2
epsilon = 0.0001


def run(x):
    x1 = np.float32(x / 255 > 0.8)
    s1 = x1.reshape(-1) @ w1
    top_k = np.argpartition(s1, -k1)[-k1:]
    x2 = np.zeros(POP_SIZE1)
    x2[top_k] = 1
    s2 = x2 @ w2n
    top_1 = s2.argmax()
    return x1, x2, top_1


def learn(x2, top_1):
    global w2n
    w2[:, top_1] += epsilon * (x2 - 0.5) * 2
    w2n = w2 / w2.sum(0)


for _ in tqdm(range(1000), desc="training"):
    img = rand_patch()
    if img.sum() == 0:
        continue
    _, middle, top = run(img)
    learn(middle, top)

stats = np.zeros((POP_SIZE2, PATCH_SIZE[0], PATCH_SIZE[1]))
for _ in tqdm(range(500), desc="eval"):
    img = rand_patch()
    if img.sum() == 0:
        continue
    bottom, _, top = run(img)
    stats[top] += bottom

w, h = 5, 4
fig, axs = plt.subplots(w, h)
for i in range(w):
    for j in range(h):
        axs[i, j].imshow(stats[i + j * w])
plt.show()
