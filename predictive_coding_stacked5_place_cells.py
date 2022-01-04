import rusty_neat
import os
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
    idx = int(np.random.rand() * len(MNIST))
    img = MNIST[idx]
    left_bottom = (img.shape - patch_size) * r
    left_bottom = left_bottom.astype(int)
    top_right = left_bottom + patch_size
    return img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]], LABELS[idx]


def experiment(clazz, k, channels, iterations=1000000, interval=100000, test_patches=20000):
    model_file = "predictive_coding_stacked5_experiment3.model"
    save_file = "predictive_coding_stacked5_place_cells_experiment" + str(EXPERIMENT)
    m = clazz.load(model_file)
    m.set_k(m.len-1, k)
    patch_size = np.array(m.get_in_shape(0))
    print("PATCH_SIZE=", patch_size, "Params=", m.learnable_paramemters())
    pc = [ecc.CpuEccMachine(
        output=[1, 1],
        kernels=[[1, 1]],
        strides=[[1, 1]],
        channels=[m.last_output_channels(), channels],
        k=[1],
        connections_per_output=[None],
    ) for _ in range(10)]
    fig, axs = plt.subplots(channels, 10)
    for a in axs:
        for b in a:
            b.set_axis_off()
    enc = htm.EncoderBuilder()
    img_w, img_h, img_c = patch_size
    img_encoder = enc.add_image(img_w, img_h, img_c, 0.8)

    def normalise_img(img_lbl):
        img, lbl = img_lbl
        sdr = htm.CpuSDR()
        img_encoder.encode(sdr, img.unsqueeze(2).numpy())
        return img, sdr, lbl

    test_patches = [normalise_img(rand_patch(patch_size[:2])) for _ in range(test_patches)]

    for s in tqdm(range(iterations), desc="training"):
        img, sdr, lbl = normalise_img(rand_patch(patch_size[:2]))
        m.run(sdr)
        m.learn()
        if s % interval == 0:
            m.save(model_file)
            stats = torch.zeros([patch_size[0], patch_size[1], channels * 10])
            for img, sdr in tqdm(test_patches, desc="eval"):
                # img = normalise_img(rand_patch())
                m.run(sdr)
                top = m.item()
                if top is not None:
                    stats[:, :, top] += img

            for i in range(channels):
                for j in range(10):
                    axs[i, j].imshow(stats[:, :, i * 10 + j])
            plt.pause(0.01)
            img_file_name = save_file + " before.png" if s == 0 else save_file + " after.png"
            plt.savefig(img_file_name)
    plt.show()


EXPERIMENT = 1

experiment(clazz=ecc.CpuEccMachine,
           channels=4,
           k=3,
           interval=20000)
