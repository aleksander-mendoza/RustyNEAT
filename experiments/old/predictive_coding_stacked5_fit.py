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


def rand_patch(patch_size, superimpose=(1, 1)):
    patch = torch.zeros((patch_size[0], patch_size[1]))
    for i in range(superimpose[0] + int((superimpose[1] - superimpose[0]) * np.random.rand())):
        r = np.random.rand(2)
        img = MNIST[int(np.random.rand() * len(MNIST))]
        left_bottom = (img.shape - patch_size) * r
        left_bottom = left_bottom.astype(int)
        top_right = left_bottom + patch_size
        patch = patch.max(img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]])
    return patch


def experiment(clazz, output, kernels, strides, channels, k, connections_per_output, w, h, test_patches=20000, superimpose=(1, 1)):
    m = clazz(
        output=output,
        kernels=kernels,
        strides=strides,
        channels=channels,
        k=k,
        connections_per_output=connections_per_output,
    )
    patch_size = np.array(m.get_in_shape(0))
    print("PATCH_SIZE=", patch_size, "Params=", m.learnable_paramemters())
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

    test_patches = [normalise_img(rand_patch(patch_size[:2], superimpose)) for _ in range(test_patches)]
    all_processed = []
    total_fits = []
    while True:
        stats = torch.zeros([patch_size[0], patch_size[1], m.last_output_channels()])
        processed = 0
        total_fit = 0
        for img, sdr in tqdm(test_patches, desc="eval"):
            # img = normalise_img(rand_patch())
            m.run(sdr)
            for top in m.last_output_sdr():
                stats[:, :, top] += img
                processed += 1
                total_fit += m.get_sums(m.len-1)[top]
        all_processed.append(processed)
        total_fits.append(total_fit)
        print("processed=", all_processed)
        print("total_fit=", total_fits)
        for i in range(w):
            for j in range(h):
                axs[i, j].imshow(stats[:, :, i + j * w])
        plt.pause(0.01)
        for img, sdr in tqdm(test_patches, desc="training"):
            m.run(sdr)
            m.learn()


EXPERIMENT = 2
if EXPERIMENT == 1:
    experiment(clazz=ecc.CpuEccMachine,
               output=np.array([1, 1]),
               kernels=[np.array([5, 5])],
               strides=[np.array([1, 1])],
               channels=[1, 50],
               k=[1],
               connections_per_output=[None],
               w=5, h=10)
elif EXPERIMENT == 2:
    experiment(clazz=ecc.CpuEccMachine,
               output=np.array([1, 1]),
               kernels=[np.array([5, 5]), np.array([1, 1])],
               strides=[np.array([2, 2]), np.array([1, 1])],
               channels=[1, 80, 50],
               k=[10, 1],
               connections_per_output=[4, None],
               w=5, h=10)
