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
    img = MNIST[int(np.random.rand() * len(MNIST))]
    left_bottom = (img.shape - patch_size) * r
    left_bottom = left_bottom.astype(int)
    top_right = left_bottom + patch_size
    return img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]]


def experiment(clazz, w, h, save_file, k=None, test_patches=20000):
    model_file = save_file + ".model"
    m = clazz.load(model_file)
    assert w * h == m.get_out_volume(m.len - 1)
    if k is not None:
        m.set_k(m.len-1, k)
    patch_size = np.array(m.get_in_shape(0))
    print("PATCH_SIZE=", patch_size, "Params=", m.learnable_parameters())
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

    all_processed = []
    stats = torch.zeros([patch_size[0], patch_size[1], w*h])
    processed = 0
    for _ in tqdm(range(test_patches), desc="eval"):
        img, sdr = normalise_img(rand_patch(patch_size[:2]))
        m.run(sdr)
        for top in m.last_output_sdr():
            stats[:, :, top] += img
            processed += 1
    all_processed.append(processed)
    print("processed=", all_processed)

    for i in range(w):
        for j in range(h):
            axs[i, j].imshow(stats[:, :, i + j * w])
    plt.pause(0.01)
    while True:
        for a in axs:
            for b in a:
                b.set_axis_off()
        img, sdr = normalise_img(rand_patch(patch_size[:2]))
        m.run(sdr)
        axs[0, 0].imshow(img)
        print(m.last_output_sdr())
        for top in m.last_output_sdr():
            i = top % w
            j = top // w
            axs[i, j].set_axis_on()
        plt.pause(15)


EXPERIMENT = 12
n = "predictive_coding_stacked5_experiment" + str(EXPERIMENT)
experiment(clazz=ecc.CpuEccMachine, w=5*3, h=5*2, save_file=n)
