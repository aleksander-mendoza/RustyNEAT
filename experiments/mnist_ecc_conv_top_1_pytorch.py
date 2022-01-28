import rusty_neat
from rusty_neat import ndalgebra as nd
from rusty_neat import ecc
from rusty_neat import htm
import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import numpy as np
from torch.utils.data import DataLoader

EXPERIMENT = 3
if EXPERIMENT == 1:
    w, h = 8, 5
    m = ecc.CpuEccDense(output=[1, 1], kernel=[5, 5], stride=[1, 1],
                        in_channels=1, out_channels=40, k=2)
    m.top1_per_region = True
    m.dropout_per_kernel(0.2)
    #   train= 0.9539 eval= 0.9723
    #   train= 0.99205 eval= 0.9744
elif EXPERIMENT == 2:
    w, h = 8, 5
    m = ecc.CpuEccDense(output=[1, 1], kernel=[5, 5], stride=[1, 1],
                        in_channels=1, out_channels=40, k=2)
    m.top1_per_region = True
    m.dropout_per_kernel(0.2)
    # processed= [0.9939, 0.9939, 0.9939, 0.9939, 0.9939, 0.9939, 0.9939, 0.9939, 0.9939, 0.9939]
    # train= 0.957275 eval= 0.97325
    # train= 0.993925 eval= 0.97455
elif EXPERIMENT == 3:
    w, h = 4, 5
    m = ecc.CpuEccDense(output=[1, 1], kernel=[5, 5], stride=[1, 1],
                        in_channels=1, out_channels=20, k=1)
    m.top1_per_region = True
    m.dropout_per_kernel(0.2)
    # train= 0.949925 eval= 0.96975
    # train= 0.98585 eval= 0.97215
    # train= 0.994575 eval= 0.9736
enc = htm.EncoderBuilder()
enc_img = enc.add_image(28, 28, 1, 0.8)

MNIST, LABELS = torch.load('htm/data/mnist.pt')


def rand_patch(patch_size):
    r = np.random.rand(2)
    img = MNIST[int(np.random.rand() * len(MNIST))]
    left_bottom = (img.shape - patch_size) * r
    left_bottom = left_bottom.astype(int)
    top_right = left_bottom + patch_size
    patch = img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]]
    return patch


def pretrain(w, h, iterations=1000000, interval=100000, test_patches=20000):
    assert w * h == m.out_shape[2]
    patch_size = np.array(m.in_shape)
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
    all_processed = []
    total_sums = []
    for s in tqdm(range(iterations), desc="training"):
        img, sdr = normalise_img(rand_patch(patch_size[:2]))
        m.run(sdr, learn=True)
        if s % interval == 0:
            stats = torch.zeros([patch_size[0], patch_size[1], w * h])
            processed = 0
            total_sum = 0
            for img, sdr in tqdm(test_patches, desc="eval"):
                out_sdr = m.infer(sdr)
                for top in out_sdr:
                    stats[:, :, top] += img
                    processed += 1
                    total_sum += m.sums[top]
            all_processed.append(processed / len(test_patches))
            total_sums.append(total_sum)
            print("processed=", all_processed)
            print("total_sums=", total_sums)
            for i in range(w):
                for j in range(h):
                    axs[i, j].imshow(stats[:, :, i + j * w])
            plt.pause(0.01)
    plt.show()


pretrain(w, h)
m = m.repeat_column([24, 24])


class D(torch.utils.data.Dataset):
    def __init__(self, imgs, lbls):
        self.imgs = imgs
        self.lbls = lbls

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        sdr = htm.CpuSDR()
        enc_img.encode(sdr, img.unsqueeze(2).numpy())
        img_bits = m.infer(sdr)
        img = torch.zeros(m.out_volume)
        img[list(img_bits)] = 1
        return img, self.lbls[idx]


train_mnist = MNIST[:40000]
train_labels = LABELS[:40000]
train_d = D(train_mnist, train_labels)

eval_mnist = MNIST[40000:60000]
eval_labels = LABELS[40000:60000]
eval_d = D(eval_mnist, eval_labels)

linear = torch.nn.Linear(m.out_volume, 10)
loss = torch.nn.NLLLoss()
optim = torch.optim.Adam(linear.parameters())

train_dataloader = DataLoader(train_d, batch_size=64, shuffle=True)
eval_dataloader = DataLoader(eval_d, batch_size=64, shuffle=True)

for epoch in range(100):
    train_accuracy = 0
    train_total = 0
    for x, y in tqdm(train_dataloader, desc="train"):
        optim.zero_grad()
        x = linear(x)
        x = torch.log_softmax(x, dim=1)
        d = loss(x, y)
        train_accuracy += (x.argmax(1) == y).sum().item()
        train_total += x.shape[0]
        d.backward()
        optim.step()

    eval_accuracy = 0
    eval_total = 0
    for x, y in tqdm(eval_dataloader, desc="eval"):
        x = linear(x)
        eval_accuracy += (x.argmax(1) == y).sum().item()
        eval_total += x.shape[0]
    print("train=", train_accuracy / train_total, "eval=", eval_accuracy / eval_total)
