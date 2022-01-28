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
    index = int(np.random.rand() * len(MNIST))
    img = MNIST[index]
    lbl = LABELS[index]
    left_bottom = (img.shape - patch_size) * r
    left_bottom = left_bottom.astype(int)
    top_right = left_bottom + patch_size
    patch = img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]]
    return patch, lbl


def experiment(clazz, save_file, train=1000, eval=2000, batch=32):
    model_file = save_file + ".model"
    m = clazz.load(model_file)
    patch_size = np.array(m.get_in_shape(0))
    print("PATCH_SIZE=", patch_size, "Params=", m.learnable_parameters())
    enc = htm.EncoderBuilder()
    img_w, img_h, img_c = patch_size
    img_encoder = enc.add_image(img_w, img_h, img_c, 0.8)
    ov = m.get_out_volume(m.len-1)
    linear = torch.nn.Linear(ov, 10)
    optim = torch.optim.Adam(linear.parameters())
    loss = torch.nn.NLLLoss()

    def normalise_img(img):
        sdr = htm.CpuSDR()
        img_encoder.encode(sdr, img.unsqueeze(2).numpy())
        return sdr
    all_correct = []
    while True:
        for _ in tqdm(range(train // batch), desc="training"):
            x = torch.zeros((batch, ov))
            y = torch.empty(batch,dtype=torch.long)
            for i in range(batch):
                img, lbl = rand_patch(patch_size[:2])
                in_sdr = normalise_img(img)
                m.infer(in_sdr)
                out_sdr = m.last_output_sdr()
                if len(out_sdr) > 0:
                    x[i, list(out_sdr)] = 1
                y[i] = lbl

            optim.zero_grad()
            x = linear(x)
            x = torch.log_softmax(x, dim=1)
            x = loss(x, y)
            x.backward()
            optim.step()
        correct = 0
        for _ in tqdm(range(eval), desc="training"):
            img, lbl = rand_patch(patch_size[:2])
            in_sdr = normalise_img(img)
            m.infer(in_sdr)
            out_sdr = m.last_output_sdr()
            if len(out_sdr) > 0:
                x = torch.zeros(ov)
                x[list(out_sdr)] = 1
                x = linear(x)
                x = torch.argmax(x)
                if x == lbl:
                    correct += 1
        all_correct.append(correct / eval)
        plt.plot(all_correct)
        plt.pause(0.01)
        print(all_correct)


EXPERIMENT = 6
n = "predictive_coding_stacked5_experiment" + str(EXPERIMENT)
experiment(clazz=ecc.CpuEccMachine, save_file=n)
