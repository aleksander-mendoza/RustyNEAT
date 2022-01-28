import rusty_neat
from rusty_neat import ndalgebra as nd
import pandas as pd
from matplotlib import pyplot as plt

overlap = [0] * 512
s = 64
brk1 = 140
brk2 = 240
brk3 = 320
for _ in range(s):
    # overlap_64_128_16_16_05
    htm = rusty_neat.htm.CpuHTM4(64, 128, 16, 16, 0.5)
    in1 = rusty_neat.htm.bitset_from_indices([1, 5, 8, 25, 56, 41], 64)
    in2 = rusty_neat.htm.bitset_from_indices([1, 5, 8, 25, 56, 40], 64)
    in3 = rusty_neat.htm.bitset_from_indices([1, 5, 8, 25, 56, 42], 64)
    in4 = rusty_neat.htm.bitset_from_indices([1, 5, 8, 25, 56, 43], 64)
    in5 = rusty_neat.htm.bitset_from_indices([1, 5, 8, 25, 56, 44], 64)

    for i in range(0, len(overlap)):
        out1 = htm(in1, True)
        out2 = htm(in2 if i < brk1 else (in3 if i < brk2 else (in4 if i < brk3 else in5)), True)
        overlap[i] += out1.overlap(out2)
        # print("1", out1)
        # print("2", out2)

overlap = [o / s for o in overlap]
plt.plot(overlap)
plt.show()
