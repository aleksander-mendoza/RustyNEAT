import rusty_neat
from rusty_neat import ndalgebra as nd
from rusty_neat import ecc
from rusty_neat import htm
import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
import os
from tqdm import tqdm
from scipy import ndimage
import numpy as np
from torch.utils.data import DataLoader

enc = htm.EncoderBuilder()
enc_img = enc.add_image(28, 28, 1, 0.8)

MNIST, LABELS = torch.load('htm/data/mnist.pt')
layer1_save_file = 'mnist_ecc_conv_conv_conv_conv_pytorch2 layer1.model'
layer2_save_file = 'mnist_ecc_conv_conv_conv_conv_pytorch2 layer2.model'
layer3_save_file = 'mnist_ecc_conv_conv_conv_conv_pytorch2 layer3.model'
layer4_save_file = 'mnist_ecc_conv_conv_conv_conv_pytorch2 layer4.model'


def rand_patch(patch_size):
    r = np.random.rand(2)
    img = MNIST[int(np.random.rand() * len(MNIST))]
    left_bottom = (img.shape - patch_size) * r
    left_bottom = left_bottom.astype(int)
    top_right = left_bottom + patch_size
    patch = img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]]
    return patch


def pretrain(base, head, w, h, iterations=1000000, interval=100000, test_patches=20000):
    assert w * h == head.out_channels
    assert base.out_shape == head.in_shape
    patch_size = np.array(head.in_shape if base is None else base.in_shape)
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
        if base is not None:
            base.infer(sdr)
            sdr = base.last_output_sdr()
        head.run(sdr, learn=True)
        if s % interval == 0:
            stats = torch.zeros([patch_size[0], patch_size[1], w * h])
            processed = 0
            total_sum = 0
            for img, sdr in tqdm(test_patches, desc="eval"):
                if base is not None:
                    base.infer(sdr)
                    sdr = base.last_output_sdr()
                out_sdr = head.infer(sdr)
                for top in out_sdr:
                    stats[:, :, top] += img
                    processed += 1
                    total_sum += head.sums[top]
            all_processed.append(processed / len(test_patches))
            total_sums.append(total_sum)
            print("processed=", all_processed)
            print("total_sums=", total_sums)
            for i in range(w):
                for j in range(h):
                    axs[i, j].imshow(stats[:, :, i + j * w])
            plt.pause(0.01)
    plt.show()


if os.path.exists(layer1_save_file):
    l1_col = ecc.CpuEccDense.load(layer1_save_file)
else:
    l1_col = ecc.CpuEccDense(output=[1, 1], kernel=[5, 5], stride=[1, 1], in_channels=1, out_channels=40, k=1)
    pretrain(None, l1_col, 8, 5)
    l1_col.save(layer1_save_file)

if os.path.exists(layer2_save_file):
    l2_col = ecc.CpuEccDense.load(layer2_save_file)
else:
    l2_col = ecc.CpuEccDense(output=[1, 1], kernel=[5, 5], stride=[1, 1], in_channels=l1_col.out_channels,
                             out_channels=40, k=1)
    l1 = l1_col.repeat_column(l2_col.in_shape[:2]).to_machine()
    pretrain(l1, l2_col, 8, 5)
    l2_col.save(layer2_save_file)

if os.path.exists(layer3_save_file):
    l3_col = ecc.CpuEccDense.load(layer3_save_file)
else:
    l3_col = ecc.CpuEccDense(output=[1, 1], kernel=[5, 5], stride=[1, 1], in_channels=l2_col.out_channels,
                             out_channels=64, k=1)
    l2 = l2_col.repeat_column(l3_col.in_shape[:2])
    l1 = l1_col.repeat_column(l2.in_shape[:2]).to_machine()
    l1.push(l2)
    pretrain(l1, l3_col, 8, 8)
    l3_col.save(layer3_save_file)


if os.path.exists(layer4_save_file):
    l4_col = ecc.CpuEccDense.load(layer4_save_file)
else:
    l4_col = ecc.CpuEccDense(output=[1, 1], kernel=[8,8], stride=[1, 1], in_channels=l3_col.out_channels,
                             out_channels=15*15, k=1)
    l3 = l3_col.repeat_column(l4_col.in_shape[:2])
    l2 = l2_col.repeat_column(l3.in_shape[:2])
    l1 = l1_col.repeat_column(l2.in_shape[:2])
    full = l1.to_machine()
    full.push(l2)
    full.push(l3)
    del l1, l2, l3
    pretrain(full, l4_col, 15, 15)
    l4_col.save(layer4_save_file)


l1 = l1_col.repeat_column(htm.conv_out_size([28, 28], l1_col.stride, l1_col.kernel)[:2])
l2 = l2_col.repeat_column(htm.conv_out_size(l1.out_shape, l2_col.stride, l2_col.kernel)[:2])
l3 = l3_col.repeat_column(htm.conv_out_size(l2.out_shape, l3_col.stride, l3_col.kernel)[:2])
l4 = l4_col.repeat_column(htm.conv_out_size(l3.out_shape, l4_col.stride, l4_col.kernel)[:2])
full = l1.to_machine()
full.push(l2)
full.push(l3)
full.push(l4)


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
        full.infer(sdr)
        sdr = full.last_output_sdr()
        img = torch.zeros(full.out_volume)
        img[list(sdr)] = 1
        return img, self.lbls[idx]


train_mnist = MNIST[:40000]
train_labels = LABELS[:40000]
train_d = D(train_mnist, train_labels)

eval_mnist = MNIST[40000:60000]
eval_labels = LABELS[40000:60000]
eval_d = D(eval_mnist, eval_labels)

linear = torch.nn.Linear(full.out_volume, 10)
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

# train= 0.845 eval= 0.8967
# train= 0.926975 eval= 0.90795
# train= 0.951975 eval= 0.9112
