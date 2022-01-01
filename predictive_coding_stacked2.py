import rusty_neat
from rusty_neat import ndalgebra as nd
from rusty_neat import htm
import pandas as pd
import copy
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import numpy as np
from scipy.special import entr

MNIST, LABELS = torch.load('htm/data/mnist.pt')

activity_epsilon = 0.0001


def input_size(kernels, strides, output_size):
    input_size = output_size
    for stride, kernel in zip(reversed(strides), reversed(kernels)):
        input_size = htm.conv_in_size(input_size, stride, kernel)
    return input_size


def sparse_weights(input_size, output_size, sparsity):
    w = np.zeros((input_size, output_size))
    for i in range(output_size):
        w[np.random.permutation(input_size)[:sparsity], i] = 1
    return w


def dense_weights(input_size, output_size):
    w = np.random.rand(input_size, output_size)
    w = w / w.sum(0)
    return w


class Project:
    def __init__(self, input_size, output_size, sparsity=None, threshold=0.2, plasticity=0.0001):
        self.input_size, self.output_size = input_size, output_size
        input_size, output_size = np.prod(input_size), np.prod(output_size)
        self.w = dense_weights(input_size, output_size) if sparsity is None \
            else sparse_weights(input_size, output_size, sparsity)
        self.threshold = threshold
        self.plasticity = plasticity
        self.p = np.ones(output_size)
        self.top = None
        self.y = None
        self.x = None

    def run_top_k(self, x, k):
        x = x.reshape(-1)
        s = x @ self.w
        r = s + self.p
        top_k = np.argpartition(r, -k)[-k:]
        top_k = top_k[s[top_k] >= self.threshold]
        # self.p[top_k] -= activity_epsilon
        self.top = top_k if len(top_k) > 0 else None
        self.x = x

    def run_top_1(self, x):
        x = x.reshape(-1)
        s = x @ self.w
        r = s + self.p
        top = r.argmax()
        if s[top] >= self.threshold:
            self.top = top
            self.p[top] -= activity_epsilon
        else:
            self.top = None
        self.x = x

    def compute_y(self):
        y = np.zeros(self.output_size)
        if self.top is not None:
            y[self.top] = 1
        self.y = y
        return y

    def learn(self):
        if self.top is not None:
            w = self.w[:, self.top]
            w += self.plasticity * self.x
            w /= w.sum(0)


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

    def run(self, x):
        self.run_top_1(x)


class Convolve:
    def __init__(self, input_size, kernel, stride, in_channels, out_channels, column_constructor):
        assert ((input_size - kernel) % stride == 0).all(), "kernel and stride do not divide input size evenly (" + \
                                                            str(input_size) + " - " + str(kernel) + ") / " + str(stride)
        self.column_grid_dim = np.int64((input_size - kernel) / stride + 1)
        self.in_column_shape = np.array([in_channels, kernel[0], kernel[1]])
        self.out_column_shape = np.array([out_channels, 1, 1])
        self.columns: [Project] = [[column_constructor(self.in_column_shape, self.out_column_shape)
                                    for _ in range(self.column_grid_dim[1])]
                                   for _ in range(self.column_grid_dim[0])]
        for c0 in self.columns:
            for c1 in c0:
                assert (c1.input_size == self.in_column_shape).all(), str(c1.input_size) + " " + str(
                    self.in_column_shape)
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
                if column.top is not None:
                    y[column.top, i0, i1] = 1
        self.y = y
        return y

    def learn(self):
        for i0 in range(self.column_grid_dim[0]):
            for i1 in range(self.column_grid_dim[1]):
                self.columns[i0][i1].learn()

    @property
    def top(self):
        return [[c.top for c in r] for r in self.columns]

    def set_threshold(self, threshold):
        for i0 in range(self.column_grid_dim[0]):
            for i1 in range(self.column_grid_dim[1]):
                self.columns[i0][i1].threshold = threshold


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
        self.channels = channels
        self.extra = []
        self.strides = strides.copy()
        self.kernels = kernels.copy()
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

    def set_threshold(self, layer, threshold):
        if layer == 0:
            self.sep.set_threshold(threshold)
        elif layer == 1:
            self.conv.set_threshold(threshold)
        else:
            self.extra[layer - 2].set_threshold(threshold)

    def last(self):
        return self.extra[-1] if len(self.extra) > 0 else self.conv

    def compute_y(self):
        return self.last().compute_y()

    @property
    def y(self):
        return self.last().y

    @property
    def top(self):
        return self.last().top


def rand_patch(patch_size):
    r = np.random.rand(2)
    img = MNIST[int(np.random.rand() * len(MNIST))]
    left_bottom = (img.shape - patch_size) * r
    left_bottom = left_bottom.astype(int)
    top_right = left_bottom + patch_size
    return img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]]


def normalise_img(x):
    x = np.float32(x / 255 > 0.8)
    return x


def experiment1():
    w, h = 5, 4
    fig, axs = plt.subplots(w, h)
    PATCH_SIZE = np.array([5, 5])
    test_patches = [normalise_img(rand_patch(PATCH_SIZE)) for _ in range(20000)]
    POP_SIZE2 = 20
    POP_SIZE1 = 80
    sep_sparsity = 5
    m = ExclusiveCoincidenceMachine(input_size=PATCH_SIZE.prod(),
                                    separation_size=POP_SIZE1,
                                    output_size=POP_SIZE2,
                                    connection_sparsity=sep_sparsity,
                                    k=15)
    entropy = []
    for s in range(1000000):
        img = normalise_img(rand_patch(PATCH_SIZE))
        m.run(img)
        m.learn()
        if s % 2000 == 0:
            stats = np.zeros((POP_SIZE2, PATCH_SIZE[0], PATCH_SIZE[1]))
            freq = np.zeros(POP_SIZE2)
            for img in test_patches:
                # img = normalise_img(rand_patch())
                m.run(img)
                if m.top is not None:
                    stats[m.top] += img
                    freq[m.top] += 1

            print(freq)
            freq = freq / freq.sum()
            print(freq)
            # entropy.append(entr(freq).sum())
            for i in range(w):
                for j in range(h):
                    axs[i, j].imshow(stats[i + j * w])
            plt.pause(0.01)
    plt.show()


def experiment2():
    w, h = 5, 4
    fig, axs = plt.subplots(w, h)
    PATCH_SIZE = np.array([5, 5])
    test_patches = [normalise_img(rand_patch(PATCH_SIZE)) for _ in range(20000)]
    POP_SIZE2 = 20
    POP_SIZE1 = 80
    sep_sparsity = 5
    m = ExclusiveCoincidenceMachineConv(input_size=PATCH_SIZE,
                                        connection_sparsity=sep_sparsity,
                                        channels=[1, POP_SIZE1, POP_SIZE2],
                                        kernels=[PATCH_SIZE],
                                        strides=[np.array([1, 1])],
                                        k=15)
    for s in tqdm(range(100000), desc="training"):
        img = normalise_img(rand_patch(PATCH_SIZE))
        m.run(np.expand_dims(img, 0))
        m.learn()
        if s % 2000 == 0:
            stats = np.zeros((POP_SIZE2, PATCH_SIZE[0], PATCH_SIZE[1]))
            for img in tqdm(test_patches, desc="eval"):
                # img = normalise_img(rand_patch())
                m.run(np.expand_dims(img, 0))
                top = m.top[0][0]
                if top is not None:
                    stats[top] += img

            for i in range(w):
                for j in range(h):
                    axs[i, j].imshow(stats[i + j * w])
            plt.pause(0.01)
    plt.show()


def experiment3():
    w, h = 5, 8
    fig, axs = plt.subplots(w, h)
    for a in axs:
        for b in a:
            b.set_axis_off()
    PATCH_SIZE = np.array([7, 7])
    test_patches = [normalise_img(rand_patch(PATCH_SIZE)) for _ in range(20000)]
    POP_SIZE2 = 20
    POP_SIZE1 = 80
    sep_sparsity = 5
    m = ExclusiveCoincidenceMachineConv(input_size=PATCH_SIZE,
                                        connection_sparsity=sep_sparsity,
                                        channels=[1, POP_SIZE1, POP_SIZE2, 40],
                                        kernels=[np.array([5, 5]), np.array([2, 2])],
                                        strides=[np.array([2, 2]), np.array([1, 1])],
                                        k=15)
    m.set_threshold(2, 0.02)
    for s in range(100000):
        img = normalise_img(rand_patch(PATCH_SIZE))
        m.run(np.expand_dims(img, 0))
        m.learn()
        if s % 2000 == 0:
            stats = np.zeros((m.channels[-1], PATCH_SIZE[0], PATCH_SIZE[1]))
            freq = np.zeros(m.channels[-1])
            for img in test_patches:
                # img = normalise_img(rand_patch())
                m.run(np.expand_dims(img, 0))
                top = m.top[0][0]
                if top is not None:
                    stats[top] += img
                    freq[top] += 1
            print(freq)
            print(freq/freq.sum())
            for i in range(w):
                for j in range(h):
                    axs[i, j].imshow(stats[i + j * w])
            plt.pause(0.01)
    plt.show()


def experiment4():
    w, h = 8, 8
    fig, axs = plt.subplots(w, h)
    for a in axs:
        for b in a:
            b.set_axis_off()
    PATCH_SIZE = np.array([11, 11])
    test_patches = [normalise_img(rand_patch(PATCH_SIZE)) for _ in range(20000)]
    POP_SIZE2 = 20
    POP_SIZE1 = 64
    sep_sparsity = 5
    m = ExclusiveCoincidenceMachineConv(input_size=PATCH_SIZE,
                                        connection_sparsity=sep_sparsity,
                                        channels=[1, POP_SIZE1, POP_SIZE2, 40, 64],
                                        kernels=[np.array([5, 5]), np.array([2, 2]), np.array([3, 3])],
                                        strides=[np.array([2, 2]), np.array([1, 1]), np.array([1, 1])],
                                        k=15)
    assert (m.last().output_shape == np.array([m.channels[-1],1,1])).all()
    m.set_threshold(2, 0.02)
    m.set_threshold(3, 0.02)
    for s in tqdm(range(1000000), desc="training"):
        img = normalise_img(rand_patch(PATCH_SIZE))
        m.run(np.expand_dims(img, 0))
        m.learn()
        if s % 10000 == 0:
            stats = np.zeros((m.channels[-1], PATCH_SIZE[0], PATCH_SIZE[1]))
            for img in tqdm(test_patches, desc="eval"):
                # img = normalise_img(rand_patch())
                m.run(np.expand_dims(img, 0))
                top = m.top[0][0]
                if top is not None:
                    stats[top] += img

            for i in range(w):
                for j in range(h):
                    axs[i, j].imshow(stats[i + j * w])
            plt.pause(0.01)
    plt.show()


def experiment5():
    w, h = 10, 10
    fig, axs = plt.subplots(w, h)
    for a in axs:
        for b in a:
            b.set_axis_off()

    kernels = [np.array([5, 5]), np.array([3, 3]), np.array([3, 3]), np.array([3, 3])]
    strides = [np.array([2, 2]), np.array([2, 2]), np.array([1, 1]), np.array([1, 1])]
    PATCH_SIZE = input_size(kernels, strides, np.array([1, 1]))
    PATCH_SIZE = np.array(PATCH_SIZE[1:])
    print("PATCH_SIZE=",PATCH_SIZE)
    test_patches = [normalise_img(rand_patch(PATCH_SIZE)) for _ in range(20000)]
    POP_SIZE2 = 20
    POP_SIZE1 = 64
    sep_sparsity = 5
    m = ExclusiveCoincidenceMachineConv(input_size=PATCH_SIZE,
                                        connection_sparsity=sep_sparsity,
                                        channels=[1, POP_SIZE1, POP_SIZE2, 40, 64, 100],
                                        kernels=kernels,
                                        strides=strides,
                                        k=15)

    assert (m.last().output_shape == np.array([m.channels[-1],1,1])).all()
    m.set_threshold(2, 0.02)
    m.set_threshold(3, 0.02)
    m.set_threshold(4, 0.01)
    for s in tqdm(range(1000000), desc="training"):
        img = normalise_img(rand_patch(PATCH_SIZE))
        m.run(np.expand_dims(img, 0))
        m.learn()
        if s % 100000 == 0:
            stats = np.zeros((m.channels[-1], PATCH_SIZE[0], PATCH_SIZE[1]))
            for img in tqdm(test_patches, desc="eval"):
                # img = normalise_img(rand_patch())
                m.run(np.expand_dims(img, 0))
                top = m.top[0][0]
                if top is not None:
                    stats[top] += img

            for i in range(w):
                for j in range(h):
                    axs[i, j].imshow(stats[i + j * w])
            plt.pause(0.01)
    plt.show()

experiment5()
