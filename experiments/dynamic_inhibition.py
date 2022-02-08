import math
import re
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
import json
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader

ex_w = ecc.ConvWeights.new_in([6, 6, 1], [6, 6], [1, 1], 49)
ex_p = ecc.CpuEccPopulation(ex_w.out_shape, 1)
inh_w = ecc.ConvWeights.new_in([6, 6, 1], [6, 6], [1, 1], 6)
inh_p = ecc.CpuEccPopulation(inh_w.out_shape, 1)

mnist = htm.CpuSdrDataset.load("predictive_coding_stacked8/c1 data.pickle")
patch_indices = mnist.gen_rand_conv_subregion_indices_with_ker(ex_w.kernel, ex_w.stride, 2000)
input_patches = mnist.conv_subregion_indices_with_ker(ex_w.kernel, ex_w.stride, patch_indices)
input_patches.filter_by_cardinality_threshold(2)
for patch in input_patches:
    inh_w
plt.imshow(input_patches.to_numpy(4))

plt.pause()


