import rusty_neat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import numpy as np
import random
import torch
from rusty_neat import htm

MNIST, LABELS = torch.load('htm/data/mnist.pt')
shuffle = torch.randperm(len(MNIST))
MNIST = MNIST[shuffle]
LABELS = LABELS[shuffle]
M = [torch.zeros((28, 28), dtype=torch.int32) for i in range(10)]
for m, l in zip(MNIST, LABELS):
    M[l] += m

# plt.imshow(M5/sum(sum(M5)))
# plt.show()
MM = torch.zeros((28, 28), dtype=torch.int32)
for m in M:
    MM += m
# plt.imshow(MM/sum(sum(MM)))
# plt.show()
sp = htm.cpu_htm4_new(28 * 28, 128)

# t = torch.zeros((28 * 28,28 * 28), dtype=torch.float)
# M5_ =
#     for i, _ in sp.get_synapses():
#         M5_[i] = 1
#     M5_ = M5_.reshape((28, 28))
#     plt.imshow(M5_)
#     plt.show()
enc = rusty_neat.htm.EncoderBuilder()
lbl_enc = enc.add_categorical(10, 128)
enc = rusty_neat.htm.EncoderBuilder()
img_enc = enc.add_bits(28 * 28)
for m in M:
    m_prob = m/sum(sum(m))
    MM_prob = MM / sum(sum(MM))
    diff = (MM_prob - m_prob).clip(0, 1)
    diff = (diff*1000).type(torch.int)
    from_col = sp.minicolumn_count
    sp.add_with_input_distribution_exact_inhibitory(m.reshape(28 * 28).tolist(), diff.reshape(28 * 28).tolist(), lbl_enc.sdr_cardinality, 256)
    to_col = sp.minicolumn_count
    # for col in range(from_col, to_col):
    #     synapses = torch.zeros(28* 28, dtype=torch.int32)
    #     for inp, perm, is_inhib in sp.get_synapses_and_inhibitions(col):
    #         synapses[inp] = -1 if is_inhib else 1
    #     plt.imshow(synapses.reshape((28,28)))
    #     plt.show()

sp.set_all_permanences(1.)
b = htm.CpuBitset(28 * 28)
confusion_matrix = np.zeros((10, 10), dtype=int)
for m, l in tqdm(zip(MNIST, LABELS), total=len(MNIST)):
    m = m.type(torch.float) / 255
    m = m.clip(0, 1) > 0.8
    m = m.reshape(28 * 28).tolist()
    b.clear()
    img_enc.encode(b, m)
    predicted = sp.compute(b)
    predicted = lbl_enc.find_category_with_highest_overlap(predicted)
    confusion_matrix[predicted, l] += 1

print(sum(confusion_matrix.diagonal()) / sum(sum(confusion_matrix)))
print(confusion_matrix)
# 0.6184
# [[4760    4   71   76    0  443   59    4   40    8]
#  [  49 6228 1219 1074  238  564  124  951 1777  309]
#  [  75    4 2639   53   20   10   16   36   33   32]
#  [  52    5  158 2789    1  242    3    3  350   44]
#  [  10    4   57   10 1871   89   41  110   32  136]
#  [  37   18   25 1092    4 2395   30    0  324   32]
#  [ 653  472 1473  462  689  761 5638  121  682  187]
#  [ 236    0   42   72   13  200    1 4277   20  131]
#  [   9    2  149   58    7  227    4   49 1514   77]
#  [  42    5  125  445 2999  490    2  714 1079 4993]]
exit()

sp = htm.CpuHTM2(3, 2, 2, 3, 345)
sp.permanence_increment = -sp.permanence_decrement

data = [
    htm.bitset_from_indices([0, 1], 3),
    htm.bitset_from_indices([1, 2], 3),
]
active_columns = [
    htm.bitset_from_indices([0], 2),
    htm.bitset_from_indices([1], 2),
]

permanences = [[], [], [], [], [], []]

for _ in range(0, 20):
    for d, a in zip(data, active_columns):
        sp.update_permanence_and_penalize(a, d, -1)
        plt.clf()
        for s in range(0, sp.synapse_count):
            i, p = sp.get_synapse_input_and_permanence(s)
            if s >= 3:
                i += 3
            permanences[i].append(p)
        for i in range(0, 6):
            plt.plot(permanences[i], label=str(i))
        plt.legend()
        plt.pause(0.001)
plt.show()
