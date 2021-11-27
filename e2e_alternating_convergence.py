import rusty_neat
from rusty_neat import htm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage
import time

in_size = 512
out_size = 512
in_card = 64
out_card = 64
synapses = int(in_size * 0.7)


def make_sdr(card, size):
    sdr = htm.CpuSDR()
    sdr.add_unique_random(card, 0, size)
    sdr.normalize()
    return sdr


def make_bitset(card, size):
    return make_sdr(card, size).to_bitset(size)


def make_sp():
    return htm.CpuHTM2(in_size, out_size, out_card, synapses)


in_bits = [make_bitset(in_card, in_size), make_bitset(in_card, in_size)]
out_sdrs = [make_sdr(out_card, out_size), make_sdr(out_card, out_size)]
spacial_poolers = [make_sp(), make_sp()]
overlaps0 = []
overlaps1 = []
for _ in range(100):
    out_00 = spacial_poolers[0].compute(in_bits[0])
    out_01 = spacial_poolers[0].compute(in_bits[1])
    out_10 = spacial_poolers[1].compute(in_bits[0])
    out_11 = spacial_poolers[1].compute(in_bits[1])
    spacial_poolers[0].update_permanence(out_sdrs[0], in_bits[0])
    spacial_poolers[0].update_permanence(out_sdrs[1], in_bits[1])
    spacial_poolers[1].update_permanence(out_sdrs[0], in_bits[0])
    spacial_poolers[1].update_permanence(out_sdrs[1], in_bits[1])
    overlap0 = out_00.overlap(out_10)
    overlap1 = out_01.overlap(out_11)
    overlaps0.append(overlap0)
    overlaps1.append(overlap1)
    plt.clf()
    plt.plot(overlaps0, label='0')
    plt.plot(overlaps1, label='1')
    plt.pause(0.01)

plt.show()
