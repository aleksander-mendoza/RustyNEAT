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

SAVE = True
MNIST, LABELS = torch.load('htm/data/mnist.pt')


def rand_patch(patch_size, img_idx=None):
    if img_idx is None:
        img_idx = int(np.random.rand() * len(MNIST))
    img = MNIST[img_idx]
    r = np.random.rand(2)
    left_bottom = (img.shape - patch_size) * r
    left_bottom = left_bottom.astype(int)
    top_right = left_bottom + patch_size
    return img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]]


class Experiment:

    def __init__(self, experiment_id):
        self.w = None
        self.h = None
        self.input_shape = None
        self.iterations = 1000000
        self.interval = 100000
        self.test_patches = 20000
        self.save_file = "predictive_coding_stacked6_" + str(experiment_id)

    def save(self):
        pass

    def run(self, sdr, learn, update_activity):
        pass

    def experiment(self):
        print("PATCH_SIZE=", self.input_shape)
        fig, axs = plt.subplots(self.w, self.h)
        for a in axs:
            for b in a:
                b.set_axis_off()
        enc = htm.EncoderBuilder()
        img_w, img_h, img_c = self.input_shape
        img_encoder = enc.add_image(img_w, img_h, img_c, 0.8)

        def normalise_img(img):
            sdr = htm.CpuSDR()
            img_encoder.encode(sdr, img.unsqueeze(2).numpy())
            return img, sdr

        test_patches = [normalise_img(rand_patch(self.input_shape[:2])) for _ in range(self.test_patches)]
        all_processed = []
        for s in tqdm(range(self.iterations), desc="training"):
            img, sdr = normalise_img(rand_patch(self.input_shape[:2]))
            self.run(sdr, learn=True, update_activity=True)
            if s % self.interval == 0:
                if SAVE:
                    self.save()
                stats = torch.zeros([self.input_shape[0], self.input_shape[1], self.w * self.h])
                processed = 0
                for img, sdr in tqdm(test_patches, desc="eval"):
                    output_sdr = self.run(sdr, learn=False, update_activity=False)
                    for top in output_sdr:
                        stats[:, :, top] += img
                        processed += 1
                all_processed.append(processed)
                print("processed=", all_processed)

                for i in range(self.w):
                    for j in range(self.h):
                        axs[i, j].imshow(stats[:, :, i + j * self.w])
                plt.pause(0.01)
                if SAVE:
                    img_file_name = self.save_file + " before.png"
                    if s == 0 or os.path.exists(img_file_name):
                        img_file_name = self.save_file + " after.png"
                    plt.savefig(img_file_name)
        plt.show()


class Experiment1(Experiment):
    def __init__(self):
        super().__init__("experiment1")
        self.layer1_save_file = self.save_file + " layer1.model"
        self.input_to_layer1_save_file = self.save_file + " input_to_layer1.model"
        if os.path.exists(self.input_to_layer1_save_file):
            self.input_to_layer1 = ecc.ConvWeights.load(self.input_to_layer1_save_file)
        else:
            self.input_to_layer1 = ecc.ConvWeights([1, 1, 20], [5, 5], [1, 1], 1)

        if os.path.exists(self.layer1_save_file):
            self.layer1 = ecc.CpuEccPopulation.load(self.layer1_save_file)
        else:
            self.layer1 = ecc.CpuEccPopulation(self.input_to_layer1.out_shape, 1)
        self.w = 4
        self.h = 5
        self.input_shape = np.array(self.input_to_layer1.in_shape)

    def run(self, sdr, learn, update_activity):
        layer1_sdr = self.input_to_layer1.run(sdr, self.layer1, update_activity=update_activity, learn=learn)
        return layer1_sdr

    def save(self):
        self.layer1.save(self.layer1_save_file)
        self.input_to_layer1.save(self.input_to_layer1_save_file)


Experiment1().experiment()
