from rusty_neat import htm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
import random

input_size = 128
input_cardinality = 16
output_cardinality = 16
inputs_per_minicolumn = 64
excitatory_connection_probability = 1
learning_iterations = 0
trials = 64
output_overlaps_128 = [0] * (input_cardinality + 1)
output_overlaps_256 = [0] * (input_cardinality + 1)
output_overlaps_512 = [0] * (input_cardinality + 1)


def gen_rand_sdr_pair(sdr_cardinality, common_overlap):
    perm = np.random.permutation(input_size)
    sdr1 = perm[:sdr_cardinality]
    b = sdr_cardinality - common_overlap
    sdr2 = perm[b:b + sdr_cardinality]
    return htm.bitset_from_indices(sdr1, input_size), htm.bitset_from_indices(sdr2, input_size)


for trial in range(trials):
    sp128 = htm.CpuHTM4(input_size,
                        128,
                        output_cardinality,
                        inputs_per_minicolumn,
                        excitatory_connection_probability)
    sp256 = htm.CpuHTM4(input_size,
                        256,
                        output_cardinality,
                        inputs_per_minicolumn,
                        excitatory_connection_probability)
    sp512 = htm.CpuHTM4(input_size,
                        512,
                        output_cardinality,
                        inputs_per_minicolumn,
                        excitatory_connection_probability)

    for input_overlap in range(input_cardinality + 1):
        bitset1, bitset2 = gen_rand_sdr_pair(input_cardinality, input_overlap)
        assert bitset1.overlap(bitset2) == input_overlap


        def run(sp, output_overlaps):
            for _ in range(learning_iterations):
                sp(bitset1, True)
                sp(bitset2, True)
            output1 = sp(bitset1)
            output2 = sp(bitset2)
            output_overlap = output1.overlap(output2)
            output_overlaps[input_overlap] += output_overlap


        run(sp128, output_overlaps_128)
        run(sp256, output_overlaps_256)
        run(sp512, output_overlaps_512)
        plt.clf()
        plt.plot([o / (trial + 1) for o in output_overlaps_128], label="128")
        plt.plot([o / (trial + 1) for o in output_overlaps_256], label="256")
        plt.plot([o / (trial + 1) for o in output_overlaps_512], label="512")
        plt.legend()
        plt.pause(0.001)

plt.show()
