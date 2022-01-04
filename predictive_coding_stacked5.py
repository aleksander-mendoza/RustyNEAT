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


def experiment(output, kernels, strides, channels, k, connections_per_output, w, h, save_file,
               iterations=1000000, interval=100000, test_patches=20000):
    model_file = save_file + ".model"
    if os.path.exists(model_file):
        m = ecc.CpuEccMachine.load(model_file)
        unpickled = True
    else:
        unpickled = False
        m = ecc.CpuEccMachine(
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

    test_patches = [normalise_img(rand_patch(patch_size[:2])) for _ in range(test_patches)]

    for s in tqdm(range(iterations), desc="training"):
        img, sdr = normalise_img(rand_patch(patch_size[:2]))
        m.run(sdr)
        m.learn()
        if s % interval == 0:
            m.save(model_file)
            stats = torch.zeros([patch_size[0], patch_size[1], m.last_output_channels()])
            for img, sdr in tqdm(test_patches, desc="eval"):
                # img = normalise_img(rand_patch())
                m.run(sdr)
                top = m.item()
                if top is not None:
                    stats[:, :, top] += img

            for i in range(w):
                for j in range(h):
                    axs[i, j].imshow(stats[:, :, i + j * w])
            plt.pause(0.01)
            img_file_name = save_file + " before.png" if s == 0 and not unpickled else save_file + " after.png"
            plt.savefig(img_file_name)
    plt.show()


EXPERIMENT = 3
n = "predictive_coding_stacked5_experiment" + str(EXPERIMENT)
if EXPERIMENT == 1:
    experiment(output=np.array([1, 1]),
               kernels=[np.array([5, 5]), np.array([3, 3]), np.array([3, 3])],
               strides=[np.array([2, 2]), np.array([1, 1]), np.array([1, 1])],
               channels=[1, 50, 20, 20],
               k=[10, 1, 1],
               connections_per_output=[4, None, None],
               w=4, h=5, save_file=n, interval=20000)
elif EXPERIMENT == 2:
    experiment(output=np.array([1, 1]),
               kernels=[np.array([5, 5]), np.array([3, 3]), np.array([3, 3]), np.array([4, 4]), np.array([3, 3])],
               strides=[np.array([2, 2]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1])],
               channels=[1, 50, 20, 20, 80, 40],
               k=[10, 1, 1, 15, 1],
               connections_per_output=[4, None, None, 7, None],
               w=8, h=5, save_file=n,
               interval=20000)

