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
assert encoder_builder.input_size == 4*8  # categories are entirely independent.
# No neuron will be used by more then one category. Hence the total number of neurons (SDR size)
# is equal to number_of_categories * sdr_cardinality

# Next let's build the spacial pooler (CpuHTM2)
number_of_minicolumns = 100
htm = rusty_neat.htm.CpuHTM2(
    encoder_builder.input_size,  # number of input neurons (input SDR size)
    16,  # how many minicolumns to activate. Here we take top 16 minicolumns with maximum overlap
    number_of_minicolumns,  # number of minicolumns
    int(encoder_builder.input_size*0.8)  # the size of potential pool of each minicolumn (how many inputs should each minicolumn be connected to)
)

# Finally we create HOM. Initially the network is completely empty and has no connections.
# New segments, dendrites and synapses will grow during the process of learning.
# Old, unused synapses may also be forgotten and destroyed.
hom = rusty_neat.htm.CpuHOM(
    8,  # Number of cells that make up a single minicolumn
    number_of_minicolumns  # Number of minicolumns
)

sdr = rusty_neat.htm.CpuSDR()
NOTE_A = 0
NOTE_B = 1
NOTE_C_SHARP = 2
NOTE_D = 3
integer_encoder.encode(sdr, NOTE_A)
active_minicolumns = htm(
    sdr,
    True  # Should the HTM learn. If False, then only inference will run and connections will be learned
)
predicted_minicolumns = hom(
    active_minicolumns,
    True  # Should HOM learn
)
assert predicted_minicolumns == []  # At the beginning there is no memory of any previous
# inputs and so there are no predictions for the future


# Let's put the above code into a nice reusable function
def play_note(note):
    sdr.clear()
    integer_encoder.encode(sdr, note)
    active_minicolumns = htm(sdr, True)
    predicted_minicolumns = hom(active_minicolumns,True)
    return active_minicolumns, predicted_minicolumns


for _ in range(40):  # Let's repeat a few times. That's how HOM learns
    play_note(NOTE_B)
    play_note(NOTE_C_SHARP)
    play_note(NOTE_D)
    _, predicted_after_a = play_note(NOTE_A)
activated_by_b, predicted_after_b = play_note(NOTE_B)
activated_by_c, predicted_after_c = play_note(NOTE_C_SHARP)
activated_by_d, predicted_after_d = play_note(NOTE_D)
activated_by_a, _ = play_note(NOTE_A)
assert predicted_after_a == activated_by_b
assert predicted_after_b == activated_by_c
assert predicted_after_c == activated_by_d
assert predicted_after_d == activated_by_a





