from rusty_neat import htm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
import random


sp = htm.CpuHTM3(3, 2, 2, 3, 345)
# sp.permanence_increment = 0.01
# sp.permanence_decrement = -0.04

data = [
    htm.bitset_from_indices([0, 1], 3),
    htm.bitset_from_indices([1, 2], 3),
]
active_columns = [
    htm.bitset_from_indices([0], 2),
    htm.bitset_from_indices([1], 2),
]

permanences = [[], [], [], [], [], []]

for _ in range(0, 200):
    for d, a in zip(data, active_columns):
        sp.update_permanence_and_penalize(a, d)
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
