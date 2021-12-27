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
epsilon = 0.0001
threshold = 0.2


class Project:
    def __init__(self, input_size, output_size):
        self.input_size, self.output_size = input_size, output_size
        self.w = np.random.rand(input_size, output_size)
        self.wn = None
        self.top = None
        self.y = None
        self.normalize()

    def normalize(self):
        self.wn = self.w / self.w.sum(0)

    def run_top_k(self, x, k):
        x = x.reshape(-1)
        s = x @ self.wn
        top_k = np.argpartition(s, -k)[-k:]
        top_k = top_k[s[top_k] >= threshold]
        self.top = top_k if len(top_k) > 0 else None

    def run_top_1(self, x):
        x = x.reshape(-1)
        s = x @ self.wn
        top = s.argmax()
        self.top = top if s[top] >= threshold else None

    def compute_y(self):
        y = np.zeros(self.output_size)
        y[self.top] = 1
        self.y = y
        return y

    def learn(self):
        if self.top is not None:
            self.w[:, self.top] += epsilon * (self.x - 0.5) * 2

            self.normalize()


class Separate(Project):
    def __init__(self, input_size, output_size, k):
        super().__init__(input_size, output_size)
        self.k = k

    def run(self, x):
        self.run_top_k(x, self.k)


class Converge(Project):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.x = None

    def run(self, x):
        self.x = x
        self.run_top_1(x)


class Convolve:
    def __init__(self, input_size, kernel, stride, in_channels, out_channels, column_constructor):
        assert (input_size - kernel) % stride == 0, "kernel and stride do not divide input size evenly"
        self.column_grid_dim = (input_size - kernel) / stride + 1
        self.in_column_shape = np.array([in_channels, kernel[0], kernel[1]])
        self.out_column_shape = np.array([out_channels, 1, 1])
        self.columns = [[column_constructor(self.in_column_shape, self.out_column_shape)
                         for _ in range(self.column_grid_dim[1])]
                        for _ in range(self.column_grid_dim[0])]
        self.kernel = kernel
        self.stride = stride
        self.input_shape = np.array([in_channels, input_size[0], input_size[1]])
        self.output_shape = np.array([out_channels, input_size[0], input_size[1]])

    def run(self, x):
        assert x.shape == self.input_shape
        for i0 in self.column_grid_dim[0]:
            for i1 in self.column_grid_dim[1]:
                i = np.array([i0, i1])
                begin = i * self.stride
                end = begin + self.kernel
                patch = x[begin[0]:end[0], begin[1]:end[1]]
                column = self.columns[i0][i1]
                column.run(patch)

    def compute_y(self):
        y = np.zeros(self.output_shape)
        for i0 in self.column_grid_dim[0]:
            for i1 in self.column_grid_dim[1]:
                column = self.columns[i0][i1]
                y[column.y, i0, i1] = 1
        return y

    def learn(self):
        for i0 in self.column_grid_dim[0]:
            for i1 in self.column_grid_dim[1]:
                self.columns[i0][i1].learn()


class ConvolveSeparate(Convolve):
    def __init__(self, input_size, kernel, stride, in_channels, out_channels, k):
        super().__init__(input_size, kernel, stride, in_channels, out_channels,
                         lambda i, o: Separate(i.prod(), o.prod(), k))


class ConvolveConverge(Convolve):
    def __init__(self, input_size, kernel, stride, in_channels, out_channels):
        super().__init__(input_size, kernel, stride, in_channels, out_channels,
                         lambda i, o: Converge(i.prod(), o.prod()))


class ExclusiveCoincidenceMachine:
    def __init__(self, input_size, separation_size, output_size, k):
        self.sep = Separate(input_size, separation_size, k)
        self.conv = Converge(separation_size, output_size)

    def run(self, x):
        self.sep.run(x)
        self.conv.run(self.sep.compute_y())

    def learn(self):
        self.conv.learn()

    def compute_y(self):
        return self.conv.compute_y()

    @property
    def y(self):
        return self.conv.y

    @property
    def top(self):
        return self.conv.top


def rand_patch():
    r = np.random.rand(2)
    img = MNIST[int(np.random.rand() * len(MNIST))]
    left_bottom = (img.shape - PATCH_SIZE) * r
    left_bottom = left_bottom.astype(int)
    top_right = left_bottom + PATCH_SIZE
    return img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]]


def normalise_img(x):
    x = np.float32(x / 255 > 0.8)
    return x


def experiment1():
    w, h = 5, 4
    fig, axs = plt.subplots(w, h)
    test_patches = [normalise_img(rand_patch()) for _ in range(20000)]
    m = ExclusiveCoincidenceMachine(PATCH_SIZE.prod(), 80, 20, 15)
    for s in tqdm(range(100000), desc="training"):
        img = normalise_img(rand_patch())
        if img.sum() == 0:
            continue
        m.run(img)
        m.learn()
        if s % 2000 == 0:
            stats = np.zeros((POP_SIZE2, PATCH_SIZE[0], PATCH_SIZE[1]))
            for img in tqdm(test_patches, desc="eval"):
                # img = normalise_img(rand_patch())
                if img.sum() == 0:
                    continue
                m.run(img)
                stats[m.top] += img


            for i in range(w):
                for j in range(h):
                    axs[i, j].imshow(stats[i + j * w])
            plt.pause(0.01)


experiment1()
