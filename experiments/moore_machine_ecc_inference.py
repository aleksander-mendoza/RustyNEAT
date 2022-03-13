import rusty_neat
from rusty_neat import ndalgebra as nd, ecc, htm
import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import numpy as np

A = ['a', 'b']
MM = [
    {'a': 0, 'b': 1},  # 0
    {'a': 0, 'b': 1},  # 1
]
n = 32

c = 4
potential_pool = 8
for mm in MM:
    mm['obs'] = np.random.permutation(n)[:potential_pool]

W = np.random.rand(n, n)
prev_x = np.array([])
state = 0
while True:
    state_data = MM[state]
    a = np.random.choice(A)
    x = np.random.choice(state_data['obs'], size=c, replace=False)

    prev_x = x
    state = state_data[a]




