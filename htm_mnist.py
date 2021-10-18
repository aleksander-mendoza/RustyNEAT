import rusty_neat
from rusty_neat import ndalgebra as nd
import pandas as pd
from matplotlib import pyplot as plt

#######
# Here you can find code samples
# showcasing usage of higher order memory (HOM).
# The temporal memory algorithm can be used on top
# of spacial pooling algorithm (CpuHTM/OclHTM).
#######

SHOW_PLOTS = True

context = rusty_neat.make_gpu_context()  # First create OpenCL context

# First let's build some input encoder. We will go with a simple categorical input this time.
# There will be four musical notes: A B C# D
encoder_builder = rusty_neat.htm.EncoderBuilder()
integer_encoder = encoder_builder.add_categorical(
    4,  # number of categories
    8  # how many neurons to assign to each category (equivalently, SDR cardinality)
)




