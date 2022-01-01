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
import pickle

MNIST, LABELS = torch.load('htm/data/mnist.pt')

activity_epsilon = 0.0001


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


def conv_input_size(kernels, strides, output_size):
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

    def __getstate__(self):
        """Return state values to be pickled."""
        d = self.__dict__.copy()
        del d["x"]
        del d["y"]
        del d["top"]
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)


class Separate(Project):
    def __init__(self, input_size, output_size, connection_sparsity, k):
        # assert np.prod(input_size) < np.prod(output_size), str(input_size) + " < " + str(output_size) + " k=" + k + " s=" + connection_sparsity
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

    def __getstate__(self):
        """Return state values to be pickled."""
        d = self.__dict__.copy()
        del d["y"]
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)

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
    def __init__(self, output_size, kernels, channels, strides, k_and_conn_sparsity, thresholds):
        assert len(kernels) == len(strides) == len(k_and_conn_sparsity) == len(thresholds)
        assert len(kernels) + 1 == len(channels)
        input_size = conv_input_size(kernels, strides, output_size)
        input_size = np.array(input_size[1:])
        self.input_size = input_size
        self.channels = channels.copy()
        self.k_and_conn_sparsity = k_and_conn_sparsity.copy()
        self.layers = []
        self.strides = strides.copy()
        self.kernels = kernels.copy()
        for kernel, stride, in_channels, out_channels, k_and_c, threshold in zip(kernels, strides, channels,
                                                                                 channels[1:], k_and_conn_sparsity,
                                                                                 thresholds):
            if k_and_c is None:
                layer = ConvolveConverge(input_size=input_size,
                                         kernel=kernel,
                                         stride=stride,
                                         in_channels=in_channels,
                                         out_channels=out_channels)
            else:
                k, connection_sparsity = k_and_c
                layer = ConvolveSeparate(input_size=input_size,
                                         kernel=kernel,
                                         in_channels=in_channels,
                                         out_channels=out_channels,
                                         stride=stride,
                                         connection_sparsity=connection_sparsity,
                                         k=k)
            layer.set_threshold(threshold)
            self.layers.append(layer)
            input_size = layer.column_grid_dim

    def run(self, x):
        self.layers[0].run(x)
        for prev, curr in zip(self.layers, self.layers[1:]):
            curr.run(prev.compute_y())

    def learn(self):
        for layer, is_sep in zip(self.layers, self.k_and_conn_sparsity):
            if is_sep is None:
                layer.learn()

    def set_threshold(self, layer, threshold):
        self.layers[layer].set_threshold(threshold)

    def last(self):
        return self.layers[-1]

    def compute_y(self):
        return self.last().compute_y()

    @property
    def y(self):
        return self.last().y

    @property
    def top(self):
        return self.last().top

    def experiment(self, w, h, save_file, iterations=1000000, interval=100000, test_patches=20000):
        print("PATCH_SIZE=", self.input_size)
        fig, axs = plt.subplots(w, h)
        for a in axs:
            for b in a:
                b.set_axis_off()
        test_patches = [normalise_img(rand_patch(self.input_size)) for _ in range(test_patches)]

        assert (self.last().output_shape == np.array([self.channels[-1], 1, 1])).all()
        for s in tqdm(range(iterations), desc="training"):
            img = normalise_img(rand_patch(self.input_size))
            self.run(np.expand_dims(img, 0))
            self.learn()
            if s % interval == 0:
                with open(save_file+".model", "wb+") as f:
                    pickle.dump(self, f)
                stats = np.zeros((self.channels[-1], self.input_size[0], self.input_size[1]))
                for img in tqdm(test_patches, desc="eval"):
                    # img = normalise_img(rand_patch())
                    self.run(np.expand_dims(img, 0))
                    top = self.top[0][0]
                    if top is not None:
                        stats[top] += img

                for i in range(w):
                    for j in range(h):
                        axs[i, j].imshow(stats[i + j * w])
                plt.pause(0.01)
                img_file_name = save_file+" before.png" if s == 0 else save_file+" after.png"
                plt.savefig(img_file_name)
        plt.show()


EXPERIMENT = 2
n = "predictive_coding_stacked3_experiment"+str(EXPERIMENT)
if EXPERIMENT == 1:
    ExclusiveCoincidenceMachineConv(
        output_size=np.array([1, 1]),
        kernels=[np.array([5, 5]), np.array([3, 3]), np.array([3, 3])],
        strides=[np.array([2, 2]), np.array([1, 1]), np.array([1, 1])],
        channels=[1, 50, 20, 20],
        thresholds=[1 / 50, 1 / 20, 1 / 20],
        k_and_conn_sparsity=[(10, 4), None, None]
    ).experiment(4, 5, n)
elif EXPERIMENT == 2:
    ExclusiveCoincidenceMachineConv(
        output_size=np.array([1, 1]),
        kernels=[np.array([5, 5]), np.array([3, 3]), np.array([3, 3]), np.array([4, 4]), np.array([3, 3])],
        strides=[np.array([2, 2]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1])],
        channels=[1, 50, 20, 20, 80, 40],
        thresholds=[1 / 50, 1 / 20, 1 / 20, 1/80, 1/40],
        k_and_conn_sparsity=[(10, 4), None, None, (15, 7), None]
    ).experiment(8, 5, n)

