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


def rand_patch(patch_size, superimpose=(1,1)):
    patch = torch.zeros((patch_size[0], patch_size[1]))
    for i in range(superimpose[0]+int((superimpose[1]-superimpose[0])*np.random.rand())):
        r = np.random.rand(2)
        img = MNIST[int(np.random.rand() * len(MNIST))]
        left_bottom = (img.shape - patch_size) * r
        left_bottom = left_bottom.astype(int)
        top_right = left_bottom + patch_size
        patch = patch.max(img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]])
    return patch


def experiment(clazz, output, kernels, strides, channels, k, connections_per_output, w, h, save_file,
               iterations=1000000, interval=100000, test_patches=20000,superimpose=(1,1)):
    model_file = save_file + ".model"
    if os.path.exists(model_file):
        m = clazz.load(model_file)
        unpickled = True
    else:
        unpickled = False
        m = clazz(
            output=output,
            kernels=kernels,
            strides=strides,
            channels=channels,
            k=k,
            connections_per_output=connections_per_output,
        )
    assert w * h == m.get_out_volume(m.len - 1)
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

    test_patches = [normalise_img(rand_patch(patch_size[:2],superimpose)) for _ in range(test_patches)]
    all_processed = []
    for s in tqdm(range(iterations), desc="training"):
        img, sdr = normalise_img(rand_patch(patch_size[:2],superimpose))
        m.run(sdr)
        m.learn()
        if s % interval == 0:
            m.save(model_file)
            stats = torch.zeros([patch_size[0], patch_size[1], w*h])
            processed = 0
            for img, sdr in tqdm(test_patches, desc="eval"):
                # img = normalise_img(rand_patch())
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
            img_file_name = save_file + " before.png" if s == 0 and not unpickled else save_file + " after.png"
            plt.savefig(img_file_name)
    plt.show()


EXPERIMENT = 10
n = "predictive_coding_stacked5_experiment" + str(EXPERIMENT)
if EXPERIMENT == 1:
    experiment(clazz=ecc.CpuEccMachineUint,
               output=np.array([1, 1]),
               kernels=[np.array([5, 5]), np.array([3, 3]), np.array([3, 3])],
               strides=[np.array([2, 2]), np.array([1, 1]), np.array([1, 1])],
               channels=[1, 50, 20, 20],
               k=[10, 1, 1],
               connections_per_output=[4, None, None],
               w=4, h=5, save_file=n, interval=20000)
elif EXPERIMENT == 2:
    experiment(clazz=ecc.CpuEccMachineUInt,
               output=np.array([1, 1]),
               kernels=[np.array([5, 5]), np.array([3, 3]), np.array([3, 3]), np.array([4, 4]), np.array([3, 3])],
               strides=[np.array([2, 2]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1])],
               channels=[1, 50, 20, 20, 80, 40],
               k=[10, 1, 1, 15, 1],
               connections_per_output=[4, None, None, 7, None],
               w=8, h=5, save_file=n,
               interval=20000)
elif EXPERIMENT == 3:
    experiment(clazz=ecc.CpuEccMachine,
               output=np.array([1, 1]),
               kernels=[np.array([5, 5]), np.array([3, 3]), np.array([3, 3]), np.array([4, 4]), np.array([3, 3])],
               strides=[np.array([2, 2]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1])],
               channels=[1, 50, 20, 20, 80, 40],
               k=[10, 1, 1, 15, 1],
               connections_per_output=[4, None, None, 7, None],
               w=8, h=5, save_file=n,
               interval=20000)
elif EXPERIMENT == 4:
    experiment(clazz=ecc.CpuEccMachine,
               output=np.array([1, 1]),
               kernels=[np.array([5, 5]), np.array([3, 3]), np.array([3, 3]), np.array([4, 4]), np.array([3, 3])],
               strides=[np.array([2, 2]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1])],
               channels=[1, 50, 20, 20, 80, 80],
               k=[10, 1, 1, 15, 4],
               connections_per_output=[4, None, None, 7, None],
               w=8, h=10, save_file=n,
               interval=20000)
elif EXPERIMENT == 5:
    experiment(clazz=ecc.CpuEccMachine,
               output=np.array([1, 1]),
               kernels=[np.array([5, 5]), np.array([3, 3]), np.array([3, 3]), np.array([4, 4]), np.array([3, 3])],
               strides=[np.array([2, 2]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1])],
               channels=[1, 50, 20, 20, 80, 80],
               k=[10, 1, 1, 15, 1],
               connections_per_output=[4, None, None, 7, None],
               w=8, h=10, save_file=n,
               interval=20000)
elif EXPERIMENT == 6:
    experiment(clazz=ecc.CpuEccMachine,
               output=np.array([1, 1]),
               kernels=[np.array([5, 5]), np.array([3, 3]), np.array([3, 3]), np.array([4, 4]), np.array([3, 3])],
               strides=[np.array([2, 2]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1])],
               channels=[1, 80, 50, 50, 100, 100],
               k=[10, 1, 1, 15, 1],
               connections_per_output=[4, None, None, 7, None],
               w=10, h=10, save_file=n,
               interval=20000)
elif EXPERIMENT == 7:
    experiment(clazz=ecc.CpuEccMachine,
               output=np.array([1, 1]),
               kernels=[np.array([5, 5]), np.array([1, 1])],
               strides=[np.array([2, 2]), np.array([1, 1])],
               channels=[1, 80, 50],
               k=[10, 1],
               connections_per_output=[4, None],
               w=5, h=10, save_file=n,
               interval=20000)
elif EXPERIMENT == 8:
    experiment(clazz=ecc.CpuEccMachine,
               output=np.array([1, 1]),
               kernels=[np.array([5, 5]), np.array([1, 1]), np.array([2, 2]), np.array([1, 1]), np.array([3, 3]), np.array([1, 1])],
               strides=[np.array([2, 2]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1])],
               channels=[1, 80, 20, 100, 20, 100, 100],
               k=[10, 1, 13, 1, 13, 1],
               connections_per_output=[4, None, 4, None, 4, None],
               w=10, h=10, save_file=n,
               interval=20000,
               superimpose=(2, 2))
elif EXPERIMENT == 9:
    experiment(clazz=ecc.CpuEccMachine,
               output=np.array([1, 1]),
               kernels=[np.array([5, 5]), np.array([1, 1]), np.array([3, 3]), np.array([1, 1]), np.array([3, 3]), np.array([1, 1])],
               strides=[np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1])],
               channels=[1, 80, 20, 200, 20, 200, 200],
               k=[10, 1, 20, 1, 20, 1],
               connections_per_output=[4, None, 4, None, 4, None],
               w=20, h=10, save_file=n,
               interval=20000,
               superimpose=(1, 1))
elif EXPERIMENT == 10:
    experiment(clazz=ecc.CpuEccMachine,
               output=np.array([5, 5]),
               kernels=[np.array([5, 5]), np.array([3, 3]), np.array([3, 3]), np.array([3, 3]), np.array([3, 3]), np.array([3, 3])],
               strides=[np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1])],
               channels=[1, 80, 40, 40, 40, 40, 40],
               k=[10, 1, 1, 1, 1, 1],
               connections_per_output=[4, None, None, None, None, None],
               w=5*5, h=8*5, save_file=n,
               interval=20000,
               superimpose=(1, 1))

