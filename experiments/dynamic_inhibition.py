import rusty_neat
from rusty_neat import ecc, htm
from rusty_neat import ndalgebra as nd
import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import numpy as np
import pickle

MNIST, LABELS = torch.load('htm/data/mnist.pt')


def preprocess(img):
    return np.float32(img / 255 > 0.8)


class PatchDataset:

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def rand_patch(self):
        r = np.random.rand(2)
        img = MNIST[int(np.random.rand() * len(MNIST))]
        left_bottom = (img.shape - self.patch_size) * r
        left_bottom = left_bottom.astype(int)
        top_right = left_bottom + self.patch_size
        img = img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]]
        return preprocess(img)


class SdrDataset:

    def __init__(self):
        enc = htm.EncoderBuilder()
        img_enc = enc.add_image(28,28,1,0.8)
        sdrs = [htm.CpuSDR() for _ in range(len(MNIST))]
        for sdr, img in zip(sdrs, MNIST.unsqueeze(3).numpy()):
            img_enc.encode(sdr, img)
        self.X = htm.CpuSdrDataset([28,28,1],sdrs)

    def rand_patch(self):
        return self.X[int(np.random.rand() * len(MNIST))]


class L2:

    def __init__(self, n, m, epsilon):
        self.epsilon = epsilon
        self.m = m
        self.n = n
        self.W = ecc.Tensor([m, n])
        self.W.rand_assign()
        self.W.mat_norm_assign_columnwise(2)
        self.a = ecc.Tensor([m], 1)

    def run(self, x, learn):
        s = self.W.mat_sparse_dot_lhs_new_vec(x)
        s.add_assign(self.a)
        k = s.argmax()
        if learn:
            self.a.sub(k, self.epsilon * 0.1)
            self.W.mat_sparse_add_assign_scalar_to_column(k, x, self.epsilon)
            self.W.mat_norm_assign_column(k,2)
        return k


class L1:

    def __init__(self, n, m, epsilon):
        self.epsilon = epsilon
        self.m = m
        self.n = n
        self.norm = np.sum
        W = np.random.rand(n, m)
        self.W = W / self.norm(W, axis=0)
        self.a = np.ones(m)

    def run(self, x, learn):
        x = x.reshape(self.n)
        s = x @ self.W + self.a
        k = np.argmax(s)
        if learn:
            self.a[k] -= self.epsilon * 0.1
            self.W[:, k] += x * self.epsilon
            self.W[:, k] /= self.norm(self.W[:, k])
        return k


def calculate_Q(layer, X, patch_size):
    K = np.array([layer.run(x, False) for x in X])
    p_y = np.bincount(K, minlength=layer.m)
    Q = ecc.Tensor([patch_size[0], patch_size[1], layer.m], 0)
    for x, j in zip(X, K):
        Q.sparse_add_assign_scalar_to_area(x,j,1)
    Q = Q.numpy()
    for j in range(layer.m):
        if p_y[j] > 0:
            Q[:,:,j] /= p_y[j]

    # expected_xQ = np.zeros(layer.m)
    # for x, j in zip(X, K):
    #     expected_xQ[j] = x.reshape(layer.n) @ Q[j].reshape(layer.n)
    # for j in range(layer.m):
    #     if p_y[j] == 0:
    #         assert expected_xQ[j].sum() == 0
    #     else:
    #         expected_xQ[j] /= p_y[j]
    # return expected_xQ.sum(), Q
    return Q


def train(ecc, dataset, steps):
    for _ in range(steps):
        x = dataset.rand_patch()
        if x.sum() > 1:
            ecc.run(x, True)


def experiment(w, h, patch_size, epsilon=0.01, steps=2000, epochs=20000):
    n = patch_size.prod()
    m = h * w
    d = PatchDataset(patch_size)
    X = [d.rand_patch() for _ in range(20000)]
    X = np.array([x for x in X if x.sum() > 1])

    l1 = L1(n, m, epsilon)
    l2 = L2(n, m, epsilon)

    fig, axs = plt.subplots(h * 2, w)
    for a in axs:
        for b in a:
            b.set_axis_off()

    for _ in range(epochs):
        train(l1, d, steps)
        train(l2, d, steps)
        ExQ_l1, Q1 = calculate_Q(l1, X)
        ExQ_l2, Q2 = calculate_Q(l2, X)
        print("expected_xQ_sum=", ExQ_l1, ExQ_l2)
        for i in range(w):
            for j in range(h):
                axs[j, i].imshow(Q1[i + j * w])
                axs[j + h, i].imshow(Q2[i + j * w])
        plt.pause(0.01)


def experiment2(w, h, plot=False, epsilon=0.01, steps=8000, epochs=2000000):
    n = 28 * 28
    m = h * w
    patch_size = np.array([28,28])
    d = SdrDataset()
    X = [d.rand_patch() for _ in range(20000)]
    layer = L2(n, m, epsilon)

    layer.W = ecc.Tensor.load('dynamic_inhibition_experiment2.W')
    layer.a = ecc.Tensor.load('dynamic_inhibition_experiment2.a')
    K = np.array([layer.run(x, False) for x in d.X])
    classifier = np.zeros((10, m))
    for split in [20000]:
        for k, lbl in zip(K[:split],LABELS[:split]):
            classifier[lbl,k] += 1
    predicted_lbl_per_k = classifier.argmax(axis=0)
    predicted_lbl = np.array([predicted_lbl_per_k[k] for k in K])
    c = (predicted_lbl==LABELS.numpy()).sum()
    print("Correct=", c, ", %=", c/len(MNIST))  # 0.930825
    exit()
    if plot:
        fig, axs = plt.subplots(h, w)
        for a in axs:
            for b in a:
                b.set_axis_off()

    for _ in range(epochs):
        for _ in range(steps):
            layer.run(d.rand_patch(), True)
        if plot:
            Q2 = calculate_Q(layer, X, patch_size)
            for i in range(w):
                for j in range(h):
                    axs[j, i].imshow(Q2[:,:,i + j * w])
            plt.pause(0.01)
        else:
            K = np.array([layer.run(x, False) for x in X])
            p_y = np.bincount(K, minlength=m)
            print("Zeroed-out receptive fields=", (p_y==0).sum())
        # layer.W.save('dynamic_inhibition_experiment2.W')
        # layer.a.save('dynamic_inhibition_experiment2.a')



# experiment(7,7,np.array([6,6]))
experiment2(45, 45)
