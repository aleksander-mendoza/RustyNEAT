import rusty_neat
from rusty_neat import ndalgebra as nd
import pandas as pd
from matplotlib import pyplot as plt

SHOW_PLOTS = True

context = rusty_neat.make_gpu_context()  # First create OpenCL context
sdr = rusty_neat.htm.CpuSDR()

sdr.active_neurons = [1, 2]  # neuron 0 is off, neuron 1 is on, neuron 2 is on
assert len(sdr) == 2  # cardinality of SDR is 2
assert str(sdr) == "[1, 2]"
assert [x for x in sdr] == [1, 2]
# Notice CpuSDR has cardinality len(sdr) but its size is unspecified.
# We can add a new neuron of any index
sdr.push_active_neuron(365747)  # this is some huge SDR! It doesn't really
# matter. SDRs are sparse. While SDRs do not have any specific cap on the
# max index of neuron, the HTM neurons must have a specific size.
assert str(sdr) == "[1, 2, 365747]"
assert [x for x in sdr] == [1, 2, 365747]
assert len(sdr) == 3  # cardinality of SDR is 3, size is unbounded but we can tell that it's at least 365748

encoder_builder = rusty_neat.htm.EncoderBuilder()
integer_encoder = encoder_builder.add_integer(50,  # min integer (inclusive). Any lower value will be rounded to 50
                                              100,
                                              # max integer (exclusive). Any higher or equal value will be rounded to 99
                                              20,  # number of neurons to use for encoding (size of SDR)
                                              5)  # number of active neurons (cardinality of SDR)
assert encoder_builder.input_size == 20  # This is the expected size of SDR
float_encoder = encoder_builder.add_float(50.,  # min float. Any lower value will be rounded to 50
                                          100.,  # max float. Any higher value will be rounded to 100
                                          30,  # number of neurons to use for encoding (size of SDR)
                                          5)  # number of active neurons (cardinality of SDR)
assert encoder_builder.input_size == 50  # 20 for integer + 30 for float

sdr.active_neurons = []  # clear SDR
integer_encoder.encode(sdr, 75)
assert str(sdr) == "[7, 8, 9, 10, 11]"  # neurons between 0 and 19
float_encoder.encode(sdr, 91.3)
assert str(
    sdr) == "[7, 8, 9, 10, 11, 40, 41, 42, 43, 44]"  # neurons between 0 and 19 for integer; neurons between 20 and 49 for float

# Now it's time for a real-life example. We will use hotgym dataset (the same one as Numenta used).
hotgym = pd.read_csv('htm/data/hotgym.csv')
hotgym['timestamp'] = pd.to_datetime(hotgym['timestamp'])
hotgym['kw_energy_consumption'] = pd.to_numeric(hotgym['kw_energy_consumption'])
hotgym = hotgym.set_index('timestamp')
# hotgym['kw_energy_consumption'].plot()  # Run this if you want to see what the data looks like

# We will build a fresh new encoder
encoder_builder = rusty_neat.htm.EncoderBuilder()
energy_consumption_encoder = encoder_builder.add_float(0, 100, 40, 5)
time_of_day_encoder = encoder_builder.add_time_of_day(40, 5)
is_weekend_encoder = encoder_builder.add_is_weekend(40, 5)


def evaluate_hot_gym(learning_enabled, hotgym):
    # There are several implementations of HTM. CpuHTM refers to those that run on CPU,
    # whereas OclHTM stands for OpenCL implementation. OclHTM can run on GPU.
    # Every implementation will have some number.
    htm = rusty_neat.htm.CpuHTM2(
        encoder_builder.input_size,  # number of input neurons (input SDR size)
        100,  # number of minicolumns
        60,  # the size of potential pool of each minicolumn (how many inputs should each minicolumn be connected to)
        16  # how many minicolumns to activate. Here we take top 16 minicolumns with maximum overlap
    )

    produced_sdrs = []  # Here we will keep one output SDR produced by HTM for each record from hotgym
    for record_idx, (date, energy) in enumerate(zip(hotgym.index, hotgym['kw_energy_consumption'])):
        sdr.active_neurons = []

        energy_consumption_encoder.encode(sdr, energy)
        time_of_day_encoder.encode(sdr, date)
        is_weekend_encoder.encode(sdr, date)

        output_sdr = htm(sdr, learning_enabled)  # learning_enabled is optional and is assumed False by default
        output_sdr.normalize()  # sort neuron indices and remove duplicates (HTM never actually produces duplicates,
        # but there is no guarantee about order)
        if len(produced_sdrs) > 1:
            # Now let's calculate the overlapping bits for each previous SDR.
            overlaps = [(previous_sdr.overlap(output_sdr), data) for previous_sdr, data in
                        zip(produced_sdrs, zip(hotgym.index, hotgym['kw_energy_consumption']))]
            overlaps.sort(key=lambda overlap_date: overlap_date[0], reverse=True)
            # Note that the overlap is calculated correctly only if both SDRs are normalized!
            overlaps = overlaps[:15]  # Save the 15 most similar SDRs
            most_similar_timestamps = list(map(lambda x: x[1][0], overlaps))
            most_similar_energy = list(map(lambda x: x[1][1], overlaps))

            if SHOW_PLOTS:
                plt.clf()
                data_subset = hotgym['kw_energy_consumption'][:record_idx]
                data_subset.plot()
                plt.scatter(most_similar_timestamps, most_similar_energy)
                plt.pause(0.01)

        produced_sdrs.append(output_sdr)


# evaluate_hot_gym(False, hotgym[0:128])  # Observe how HTM works without learning
# evaluate_hot_gym(True, hotgym[0:128])  # Observe how HTM works with learning enabled


def evaluate_hot_gym_both_learning_and_non_learning(hotgym):
    learning_htm = rusty_neat.htm.CpuHTM2(encoder_builder.input_size, 100, 60, 16)
    non_learning_htm = rusty_neat.htm.CpuHTM2(encoder_builder.input_size, 100, 60, 16)

    # plt.gcf().canvas.mlp_connect('key_press_event', press)
    produced_sdrs_with_learning = []
    produced_sdrs_without_learning = []
    for record_idx, (date, energy) in enumerate(zip(hotgym.index, hotgym['kw_energy_consumption'])):
        sdr.active_neurons = []

        energy_consumption_encoder.encode(sdr, energy)
        time_of_day_encoder.encode(sdr, date)
        is_weekend_encoder.encode(sdr, date)

        output_sdr_with_learning = learning_htm(sdr, True)
        output_sdr_with_learning.normalize()
        output_sdr_without_learning = non_learning_htm(sdr)
        output_sdr_without_learning.normalize()

        def find_top_overlap(hotgym, output_sdr, produced_sdrs):
            overlaps = [(previous_sdr.overlap(output_sdr), data) for previous_sdr, data in
                        zip(produced_sdrs, zip(hotgym.index, hotgym['kw_energy_consumption']))]
            overlaps.sort(key=lambda overlap_date: overlap_date[0], reverse=True)
            overlaps = overlaps[:15]
            most_similar_timestamps = list(map(lambda x: x[1][0], overlaps))
            most_similar_energy = list(map(lambda x: x[1][1], overlaps))
            produced_sdrs.append(output_sdr)
            return most_similar_timestamps, most_similar_energy

        similar_no_learning = find_top_overlap(hotgym, output_sdr_without_learning, produced_sdrs_without_learning)
        similar_learning = find_top_overlap(hotgym, output_sdr_with_learning, produced_sdrs_with_learning)

        if len(produced_sdrs_with_learning) > 1:

            if SHOW_PLOTS:
                plt.clf()
                data_subset = hotgym['kw_energy_consumption'][:record_idx]
                data_subset.plot()
                plt.scatter(similar_no_learning[0], similar_no_learning[1], c='green', marker='x')
                plt.scatter(similar_learning[0], similar_learning[1], c='red', marker='o')
                plt.pause(0.1)


evaluate_hot_gym_both_learning_and_non_learning(hotgym)
