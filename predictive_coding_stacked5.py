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


def experiment(ecc, w, h, save_file, iterations=1000000, interval=100000, test_patches=20000):
    patch_size = np.array(ecc.get_in_shape(0))
    print("PATCH_SIZE=", patch_size, "Params=", ecc.learnable_paramemters())
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
        ecc.run(sdr)
        ecc.learn()
        if s % interval == 0:
            # with open(save_file + ".model", "wb+") as f:
            #     self.
            stats = torch.zeros([patch_size[0], patch_size[1], ecc.last_output_channels()])
            for img, sdr in tqdm(test_patches, desc="eval"):
                # img = normalise_img(rand_patch())
                ecc.run(sdr)
                # assert 0 not in ecc.output_sdr(2)
                top = ecc.item()
                if top:
                    stats[:, :, top] += img

            for i in range(w):
                for j in range(h):
                    axs[i, j].imshow(stats[:, :, i + j * w])
            plt.pause(0.01)
            img_file_name = save_file + " before.png" if s == 0 else save_file + " after.png"
            plt.savefig(img_file_name)
    plt.show()


EXPERIMENT = 2
n = "predictive_coding_stacked5_experiment" + str(EXPERIMENT)
if EXPERIMENT == 1:
    experiment(ecc.CpuEccMachine(
        output=np.array([1, 1]),
        kernels=[np.array([5, 5]), np.array([3, 3]), np.array([3, 3])],
        strides=[np.array([2, 2]), np.array([1, 1]), np.array([1, 1])],
        channels=[1, 50, 20, 20],
        k=[10, 1, 1],
        connections_per_output=[4, None, None]
    ), 4, 5, n, interval=20000)
elif EXPERIMENT == 2:
    experiment(ecc.CpuEccMachine(
        output=np.array([1, 1]),
        kernels=[np.array([5, 5]), np.array([3, 3]), np.array([3, 3]), np.array([4, 4]), np.array([3, 3])],
        strides=[np.array([2, 2]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1])],
        channels=[1, 50, 20, 20, 80, 40],
        k=[10, 1, 1, 15, 1],
        connections_per_output=[4, None, None, 7, None]
    ), 8, 5, n, interval=20000)
