import rusty_neat
from rusty_neat import ndalgebra as nd
from rusty_neat import htm
from rusty_neat import ecc
import pandas as pd
import copy
from matplotlib import pyplot as plt
import torch
import numpy as np

from tqdm import tqdm
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


def conv_input_size(kernels, strides, output_size):
    input_size = output_size
    for stride, kernel in zip(reversed(strides), reversed(kernels)):
        input_size = htm.conv_in_size(input_size, stride, kernel)
    return input_size


class ExclusiveCoincidenceMachineConv:
    def __init__(self, output_size, kernels, channels, strides, k, conn_sparsity):
        assert len(kernels) == len(strides) == len(k) == len(conn_sparsity)
        assert len(kernels) + 1 == len(channels)
        self.channels = channels.copy()
        self.k = k.copy()
        self.conn_sparsity = conn_sparsity.copy()
        self.layers = []
        self.strides = strides.copy()
        self.kernels = kernels.copy()
        for e in reversed(list(zip(kernels, strides, channels, channels[1:], k, conn_sparsity))):
            kernel, stride, in_channels, out_channels, k, connection_sparsity = e
            if connection_sparsity is None:
                layer = ecc.CpuEccDense(output=output_size,
                                        kernel=kernel,
                                        stride=stride,
                                        in_channels=in_channels,
                                        out_channels=out_channels,
                                        k=k)
            else:
                layer = ecc.CpuEccSparse(output=output_size,
                                         kernel=kernel,
                                         in_channels=in_channels,
                                         out_channels=out_channels,
                                         stride=stride,
                                         connections_per_output=connection_sparsity,
                                         k=k)
            self.layers.append(layer)
            output_size = layer.in_shape[:2]
        self.layers.reverse()
        self.top = [None] * (len(self.layers) + 1)

    def run(self, x):
        for i, layer in enumerate(self.layers):
            self.top[i] = x
            x = layer.run(x)
        self.top[-1] = x
        return x

    def learn(self):
        for layer, is_sep, inp, out in zip(self.layers, self.conn_sparsity, self.top, self.top[1:]):
            if is_sep is None:
                layer.learn(inp, out)

    def set_threshold(self, layer, threshold):
        self.layers[layer].set_threshold(threshold)

    def last(self):
        return self.layers[-1]

    def top_input(self, layer):
        return self.top[layer]

    def top_output(self, layer):
        return self.top[layer + 1]

    @property
    def input_size(self):
        return np.array(self.layers[0].in_shape)

    def experiment(self, w, h, save_file, iterations=1000000, interval=100000, test_patches=20000):
        patch_size = self.input_size
        print("PATCH_SIZE=", patch_size)
        fig, axs = plt.subplots(w, h)
        for a in axs:
            for b in a:
                b.set_axis_off()
        enc = htm.EncoderBuilder()
        img_w, img_h, img_c = patch_size
        img_encoder = enc.add_image(img_w, img_h, img_c, 0.8)

        def normalise_img(img):
            sdr = htm.CpuSDR()
            img_encoder.encode(sdr, img.unsqueeze(2).numpy())
            return img, sdr

        test_patches = [normalise_img(rand_patch(patch_size[:2])) for _ in range(test_patches)]

        assert (self.last().out_shape == np.array([1, 1, self.channels[-1]])).all()
        for s in tqdm(range(iterations), desc="training"):
            img, sdr = normalise_img(rand_patch(patch_size[:2]))
            self.run(sdr)
            self.learn()
            if s % interval == 0:
                # with open(save_file + ".model", "wb+") as f:
                #     self.
                stats = np.zeros(patch_size)
                for img, sdr in tqdm(test_patches, desc="eval"):
                    # img = normalise_img(rand_patch())
                    top = self.run(sdr)
                    if top:
                        stats[top] += img

                for i in range(w):
                    for j in range(h):
                        axs[i, j].imshow(stats[:, :, i + j * w])
                plt.pause(0.01)
                img_file_name = save_file + " before.png" if s == 0 else save_file + " after.png"
                plt.savefig(img_file_name)
        plt.show()


EXPERIMENT = 1
n = "predictive_coding_stacked3_experiment" + str(EXPERIMENT)
if EXPERIMENT == 1:
    ExclusiveCoincidenceMachineConv(
        output_size=np.array([1, 1]),
        kernels=[np.array([5, 5]), np.array([3, 3]), np.array([3, 3])],
        strides=[np.array([2, 2]), np.array([1, 1]), np.array([1, 1])],
        channels=[1, 50, 20, 20],
        k=[10, 1, 1],
        conn_sparsity=[4, None, None]
    ).experiment(4, 5, n)
elif EXPERIMENT == 2:
    pass
