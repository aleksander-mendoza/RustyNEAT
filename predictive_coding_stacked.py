import rusty_neat
from rusty_neat import ndalgebra as nd
import pandas as pd
import copy
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import numpy as np

MNIST, LABELS = torch.load('htm/data/mnist.pt')

PATCH_SIZE = np.array([5, 5])
activity_epsilon = 0.0001


def sparse_weights(input_size, output_size, sparsity):
    w = np.zeros((input_size, output_size))
    for i in range(output_size):
        w[np.random.permutation(input_size)[:sparsity], i] = 1
    return w


def dense_weights(input_size, output_size):
    return np.random.rand(input_size, output_size)


class Project:
    def __init__(self, input_size, output_size, sparsity=None, threshold=0.2, plasticity=0.0001):
        self.input_size, self.output_size = input_size, output_size
        input_size, output_size = np.prod(input_size), np.prod(output_size)
        self.w = dense_weights(input_size, output_size) if sparsity is None \
            else sparse_weights(input_size, output_size, sparsity)
        self.threshold = threshold
        self.plasticity = plasticity
        self.p = np.ones(output_size)
        self.wn = None
        self.top = None
        self.y = None
        self.x = None
        self.normalize()

    def normalize(self):
        self.wn = self.w / self.w.sum(0)

    def run_top_k(self, x, k):
        x = x.reshape(-1)
        s = x @ self.wn
        r = s + self.p
        top_k = np.argpartition(r, -k)[-k:]
        top_k = top_k[s[top_k] >= self.threshold]
        self.top = top_k if len(top_k) > 0 else None
        self.x = x

    def run_top_1(self, x):
        x = x.reshape(-1)
        s = x @ self.wn
        r = s + self.p
        top = r.argmax()
        self.top = top if s[top] >= self.threshold else None
        self.x = x

    def compute_y(self):
        y = np.zeros(self.output_size)
        y[self.top] = 1
        self.y = y
        return y

    def learn(self):
        if self.top is not None:
            self.p[self.top] -= activity_epsilon
            self.w[:, self.top] += self.plasticity * self.x
            self.normalize()


class Separate(Project):
    def __init__(self, input_size, output_size, connection_sparsity, k):
        assert np.prod(input_size) < np.prod(output_size)
        super().__init__(input_size, output_size, sparsity=connection_sparsity)
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
        assert ((input_size - kernel) % stride == 0).all(), "kernel and stride do not divide input size evenly"
        self.column_grid_dim = np.int64((input_size - kernel) / stride + 1)
        self.in_column_shape = np.array([in_channels, kernel[0], kernel[1]])
        self.out_column_shape = np.array([out_channels, 1, 1])
        self.columns: [Project] = [[column_constructor(self.in_column_shape, self.out_column_shape)
                                    for _ in range(self.column_grid_dim[1])]
                                   for _ in range(self.column_grid_dim[0])]
        for c0 in self.columns:
            for c1 in c0:
                assert (c1.input_size == self.in_column_shape).all(), str(c1.input_size)+" "+str(self.in_column_shape)
                assert (c1.output_size == self.out_column_shape).all()
        self.kernel = kernel
        self.stride = stride
        self.y = None
        self.input_shape = np.array([in_channels, input_size[0], input_size[1]])
        self.output_shape = np.array([out_channels, self.column_grid_dim[0], self.column_grid_dim[1]])

    def run(self, x):
        assert np.all(x.shape == self.input_shape), str(x.shape) + " " + str(self.input_shape)
        for i0 in range(self.column_grid_dim[0]):
            for i1 in range(self.column_grid_dim[1]):
                i = np.array([i0, i1])
                begin = i * self.stride
                end = begin + self.kernel
                patch = x[:, begin[0]:end[0], begin[1]:end[1]]
                column = self.columns[i0][i1]
                column.run(patch)

    def compute_y(self):
        y = np.zeros(self.output_shape)
        for i0 in range(self.column_grid_dim[0]):
            for i1 in range(self.column_grid_dim[1]):
                column = self.columns[i0][i1]
                y[column.top, i0, i1] = 1
        self.y = y
        return y

    def learn(self):
        for i0 in range(self.column_grid_dim[0]):
            for i1 in range(self.column_grid_dim[1]):
                self.columns[i0][i1].learn()


class ConvolveSeparate(Convolve):
    def __init__(self, input_size, kernel, stride, in_channels, out_channels, connection_sparsity, k):
        super().__init__(input_size, kernel, stride, in_channels, out_channels,
                         lambda i, o: Separate(i, o, connection_sparsity=connection_sparsity, k=k))


class ConvolveConverge(Convolve):
    def __init__(self, input_size, kernel, stride, in_channels, out_channels):
        super().__init__(input_size, kernel, stride, in_channels, out_channels,
                         lambda i, o: Converge(i, o))


class ConvolveCopy(Convolve):
    def __init__(self, input_size, kernel, stride, in_channels, out_channels, column):
        super().__init__(input_size, kernel, stride, in_channels, out_channels,
                         lambda i, o: copy.deepcopy(column))


class ExclusiveCoincidenceMachine:
    def __init__(self, input_size, separation_size, output_size, connection_sparsity, k):
        self.sep = Separate(input_size, separation_size, connection_sparsity=connection_sparsity, k=k)
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


class ExclusiveCoincidenceMachineConv:
    def __init__(self, input_size, kernels, channels, strides, connection_sparsity, k):
        assert len(kernels) == len(strides)
        assert len(kernels) + 2 == len(channels)
        assert len(kernels) > 0
        self.sep = ConvolveSeparate(input_size=input_size,
                                    kernel=kernels[0],
                                    in_channels=channels[0],
                                    out_channels=channels[1],
                                    stride=strides[0],
                                    connection_sparsity=connection_sparsity,
                                    k=k)

        self.conv = ConvolveConverge(input_size=self.sep.column_grid_dim,
                                     kernel=np.array([1, 1]),
                                     stride=np.array([1, 1]),
                                     in_channels=channels[1],
                                     out_channels=channels[2])
        prev_grid_dim = self.conv.column_grid_dim
        self.extra = []
        for kernel, stride, in_channels, out_channels in zip(kernels[1:], strides[1:], channels[2:], channels[3:]):
            extra_layer = ConvolveConverge(input_size=prev_grid_dim,
                                           kernel=kernel,
                                           stride=stride,
                                           in_channels=in_channels,
                                           out_channels=out_channels)
            self.extra.append(extra_layer)
            prev_grid_dim = extra_layer.column_grid_dim

    def run(self, x):
        self.sep.run(x)
        self.conv.run(self.sep.compute_y())
        prev = self.conv
        for extra in self.extra:
            extra.run(prev.compute_y())
            prev = extra

    def learn(self):
        self.conv.learn()
        for extra in self.extra:
            extra.learn()

    def last(self):
        return self.extra[-1] if len(self.extra) > 0 else self.conv

    def compute_y(self):
        return self.last().compute_y()

    @property
    def y(self):
        return self.last().y

    @property
    def top(self):
        return [[c.top for c in r] for r in self.last().columns]


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
    POP_SIZE2 = 20
    POP_SIZE1 = 80
    sep_sparsity = 5
    m = ExclusiveCoincidenceMachine(input_size=PATCH_SIZE.prod(),
                                    separation_size=POP_SIZE1,
                                    output_size=POP_SIZE2,
                                    connection_sparsity=sep_sparsity,
                                    k=15)
    for s in tqdm(range(100000), desc="training"):
        img = normalise_img(rand_patch())
        m.run(img)
        m.learn()
        if s % 2000 == 0:
            stats = np.zeros((POP_SIZE2, PATCH_SIZE[0], PATCH_SIZE[1]))
            for img in tqdm(test_patches, desc="eval"):
                # img = normalise_img(rand_patch())
                m.run(img)
                if m.top is not None:
                    stats[m.top] += img

            for i in range(w):
                for j in range(h):
                    axs[i, j].imshow(stats[i + j * w])
            plt.pause(0.01)
    plt.show()


def experiment2():
    w, h = 5, 4
    fig, axs = plt.subplots(w, h)
    test_patches = [normalise_img(rand_patch()) for _ in range(20000)]
    POP_SIZE2 = 20
    POP_SIZE1 = 80
    sep_sparsity = 5
    m = ExclusiveCoincidenceMachineConv(input_size=PATCH_SIZE,
                                        connection_sparsity=sep_sparsity,
                                        channels=[1, POP_SIZE1, POP_SIZE2],
                                        kernels=[PATCH_SIZE],
                                        strides=[1],
                                        k=15)
    for s in tqdm(range(100000), desc="training"):
        img = normalise_img(rand_patch())
        m.run(np.expand_dims(img,0))
        m.learn()
        if s % 2000 == 0:
            stats = np.zeros((POP_SIZE2, PATCH_SIZE[0], PATCH_SIZE[1]))
            for img in tqdm(test_patches, desc="eval"):
                # img = normalise_img(rand_patch())
                m.run(np.expand_dims(img,0))
                top = m.top[0][0]
                if top is not None:
                    stats[top] += img

            for i in range(w):
                for j in range(h):
                    axs[i, j].imshow(stats[i + j * w])
            plt.pause(0.01)
    plt.show()


experiment2()
