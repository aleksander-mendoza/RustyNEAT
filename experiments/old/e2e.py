import rusty_neat
from rusty_neat import htm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage
import time


def genGabor(sz, omega, theta, func=np.cos, K=np.pi):
    radius = (int(sz[0] / 2.0), int(sz[1] / 2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))
    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)
    gauss = omega ** 2 / (4 * np.pi * K ** 2) * np.exp(- omega ** 2 / (8 * K ** 2) * (4 * x1 ** 2 + y1 ** 2))
    sinusoid = func(omega * x1) * np.exp(K ** 2 / 2)
    gabor = gauss * sinusoid
    return gabor


def visualize_gabor(img, filters, threshold):
    img = img.astype(float) / 255
    fig, axs = plt.subplots(len(filters) + 1, 2)
    axs[0, 0].imshow(img)
    for k, kernel in enumerate(filters):
        i = ndimage.convolve(img, kernel, mode='constant')
        i = i > threshold
        axs[1 + k, 0].imshow(i)
        axs[1 + k, 1].imshow(kernel)
    plt.show()


def visualize_img_list(original, filtered, filters):
    fig, axs = plt.subplots(len(filters) + 1, 2)
    axs[0, 0].imshow(original)
    for k, (img, kernel) in enumerate(zip(filtered, filters)):
        axs[1 + k, 0].imshow(img)
        axs[1 + k, 1].imshow(kernel)
    plt.show()


env = Image.open('map.jpeg')
env_grayscale = env.convert('L')
env = np.array(env)
env_grayscale = np.array(env_grayscale)
gabor_filters = [genGabor((16, 16), 1, np.pi * x / 8, func=np.cos) for x in range(0, 8)]
gabor_threshold = 5
fov_span = 32
fov_size = np.array([fov_span * 2, fov_span * 2])
x, y = 64, 64
vote_kernel_size = np.array([2, 2])
vote_kernel_stride = np.array([1, 1])
vote_num_of_layer1_columns_per_layer2_column = vote_kernel_size.prod()
vote_threshold = int(vote_num_of_layer1_columns_per_layer2_column * 0.4)
image_patch_width, image_patch_height = 16, 16
image_patch_overlap = 0.25
image_patch_kernel = np.array([image_patch_height, image_patch_width])
image_patch_stride = (image_patch_kernel * np.sqrt(image_patch_overlap)).astype(int)
speed = image_patch_stride[1]
image_patch_total_size = image_patch_kernel.prod()
assert (image_patch_kernel <= fov_size).all(), "Kernel is larger than field of view"
assert ((fov_size - image_patch_kernel) % image_patch_stride == 0).all(), "Stride does not divide field of view evenly"
image_patch_count = (fov_size - image_patch_kernel) // image_patch_stride + 1
sensor_column_size = np.array([image_patch_kernel[0], image_patch_kernel[1] * len(gabor_filters)])
sensor_column_count = image_patch_count
assert (sensor_column_count == image_patch_count).all()
layer1_column_count = sensor_column_count
assert (layer1_column_count == sensor_column_count).all()  # This must be the same
layer1_column_size = np.array([16, 16])
layer1_column_card = int(layer1_column_size.prod() * 0.05)
sublayer2_column_count = (layer1_column_count - vote_kernel_size) // vote_kernel_stride + 1
working_memory_span = np.array([2, 2])
layer2_column_count = sublayer2_column_count + 2 * working_memory_span
layer2_column_size = layer1_column_size
layer2_column_card = layer1_column_card
assert (layer2_column_size == layer1_column_size).all()
sensor_to_layer1_synapses = int(sensor_column_size.prod() * 0.7)
sublayer2_total_shape = sublayer2_column_count * layer2_column_size
layer2_total_shape = layer2_column_count * layer2_column_size
layer2_total_size = layer2_total_shape.prod()
sublayer2_total_size = sublayer2_total_shape.prod()
layer3_size = layer2_total_size # * 4
layer3_card = int(layer3_size * 0.005)
layer3_synapse_span = layer2_column_size * 3
layer2_to_layer3_synapses = int(layer2_total_size * 0.05)
map_size = 64 * 64
map_card = int(map_size * 0.02)
layer3_to_map_synapses = int(layer3_size * 0.8)


def make_sensor_to_layer1_column():
    return htm.CpuHTM2(sensor_column_size.prod(),
                       layer1_column_card,
                       layer1_column_size.prod(),
                       sensor_to_layer1_synapses)


sensor_layer = [[htm.CpuBitset(sensor_column_size.prod()) for _ in range(sensor_column_count[1])] for _ in
                range(sensor_column_count[0])]
sensor_single_column_encoder = htm.EncoderBuilder()
sensor_single_filter_encoder = [sensor_single_column_encoder.add_bits(image_patch_total_size) for _ in gabor_filters]
sensor_to_layer1 = [[make_sensor_to_layer1_column() for _ in range(sensor_column_count[1])] for _ in
                    range(sensor_column_count[0])]


def make_layer2_column():
    sdr = htm.CpuSDR()
    sdr.add_unique_random(layer2_column_card, 0, layer2_column_size.prod())
    sdr.normalize()
    return sdr


layer2 = [[make_layer2_column() for _ in range(layer2_column_count[1])] for _ in range(layer2_column_count[0])]
layer2_bitset = htm.CpuBitset2d(layer2_total_shape[0], layer2_total_shape[1])
layer3 = htm.CpuInput(layer3_size)
map_activity = htm.CpuSDR()
map_activity.add_unique_random(map_card, 0, map_size)
map_activity.normalize()
layer2_to_layer3 = htm.CpuDG2_2d((layer2_total_shape[0], layer2_total_shape[1]), layer3_size, layer3_card,
                                 (layer3_synapse_span[0], layer3_synapse_span[1]), layer2_to_layer3_synapses)
layer3_to_map = htm.CpuBigHTM(layer3_size, map_size, map_card)
layer3_to_map.activation_threshold = int(10)
map_to_map = htm.CpuHOM(4, map_size)


def agent_moved(displacement):
    global layer2
    displacement_in_columns = displacement // image_patch_stride
    new_layer2 = [[None for _ in range(layer2_column_count[1])] for _ in range(layer2_column_count[0])]
    for column_y in range(layer2_column_count[0]):
        for column_x in range(layer2_column_count[1]):
            original_y = column_y + displacement_in_columns[0]
            original_x = column_x + displacement_in_columns[1]
            if 0 <= original_y < layer2_column_count[0] and 0 <= original_x < layer2_column_count[1]:
                new_layer2[column_y][column_x] = layer2[original_y][original_x]
            else:
                new_layer2[column_y][column_x] = make_layer2_column()
    layer2 = new_layer2


def move_agent(steps_x, steps_y):
    steps = np.array([steps_y, steps_x]) * speed
    global x, y
    y += steps[0]
    x += steps[1]
    agent_moved(steps)


def get_fov(img_src=env):
    img = img_src[y - fov_span:y + fov_span, x - fov_span:x + fov_span]
    assert (img.shape[0:2] == fov_size).all()
    return img


def run(learn=False, update_map=False):
    global map_activity
    # ====== 1. get sensory inputs ======
    time1_get_sensory_input = time.time()
    fov_img = get_fov(env_grayscale)
    fov_img = fov_img.astype(float) / 255

    def convolve(img, kernel):
        img = ndimage.convolve(img, kernel, mode='constant')
        img = img > gabor_threshold
        return img

    gabor_edges = [convolve(fov_img, kernel) for kernel in gabor_filters]
    # visualize_img_list(fov_img, gabor_edges, gabor_filters)
    # ====== 2. encode sensory inputs ======
    time2_encode_sensory_input = time.time()
    for column_y in range(sensor_column_count[0]):
        for column_x in range(sensor_column_count[1]):
            sensor_layer_patch = sensor_layer[column_y][column_x]
            sensor_layer_patch.clear()
            patch_offset = np.array([column_y, column_x]) * image_patch_stride
            patch_end = patch_offset + image_patch_kernel
            for encoder, filtered_img in zip(sensor_single_filter_encoder, gabor_edges):
                img_patch = filtered_img[patch_offset[0]:patch_end[0], patch_offset[1]:patch_end[1]]
                assert np.array(img_patch.shape).prod() == sensor_column_size.prod() / len(gabor_filters)
                img_patch = img_patch.reshape(image_patch_total_size)
                img_patch = img_patch.tolist()
                encoder.encode(sensor_layer_patch, img_patch)
                assert encoder.len == len(img_patch)
    # ====== 3. compute layer1 from sensory inputs  ======
    time3_compute_layer1 = time.time()
    assert (layer1_column_count == sensor_column_count).all()
    layer1 = [[None for _ in range(layer1_column_count[1])] for _ in range(layer1_column_count[0])]
    for column_y in range(sensor_column_count[0]):
        for column_x in range(sensor_column_count[1]):
            sensor_layer_patch = sensor_layer[column_y][column_x]
            spacial_pooler = sensor_to_layer1[column_y][column_x]
            layer1_sdr = spacial_pooler.compute(sensor_layer_patch)
            layer1[column_y][column_x] = layer1_sdr
    # ====== 4. vote for sublayer2 using layer1  ======
    time4_vote = time.time()
    cast_votes = htm.vote_conv2d(layer1, layer2_column_card, vote_threshold,
                                 (vote_kernel_stride[0], vote_kernel_stride[1]),
                                 (vote_kernel_size[0], vote_kernel_size[1]))
    assert len(cast_votes) == sublayer2_column_count[0]
    assert len(cast_votes[0]) == sublayer2_column_count[1]
    # ====== 5. update sublayer2 if votes carry enough consensus  ======
    time5_sublayer2 = time.time()
    for column_y in range(sublayer2_column_count[0]):
        for column_x in range(sublayer2_column_count[1]):
            new_votes = cast_votes[column_y][column_x]
            prev_activations = layer2[working_memory_span[0] + column_y][working_memory_span[1] + column_x]
            prev_activations.randomly_extend_from(new_votes)
    # ====== use layer2 as apical labels for training layer1  ======
    time5_apical_learn = time.time()
    if learn:
        apical_feedback = htm.vote_conv2d_transpose(layer2,
                                                    (vote_kernel_stride[0], vote_kernel_stride[1]),
                                                    (vote_kernel_size[0], vote_kernel_size[1]),
                                                    (layer1_column_count[0], layer1_column_count[1]))
        assert len(apical_feedback) == layer1_column_count[0]
        assert len(apical_feedback[0]) == layer1_column_count[1]
        for column_y in range(layer1_column_count[0]):
            for column_x in range(layer1_column_count[1]):
                column_apical_feedback = apical_feedback[column_y][column_x]
                layer1_column_activity = layer1[column_y][column_x]
                sensor_column_activity = sensor_layer[column_y][column_x]
                spacial_pooler = sensor_to_layer1[column_y][column_x]
                # TODO: perhaps use layer1_column_activity.intersect(column_apical_feedback) ?
                spacial_pooler.update_permanence(column_apical_feedback, sensor_column_activity)
    # ====== 6. join all layer2 SDRs into a single bitset  ======
    time6_join_layer2 = time.time()
    layer2_bitset.clear()
    for column_y in range(layer2_column_count[0]):
        for column_x in range(layer2_column_count[1]):
            column_y_offset = column_y * layer2_column_size[0]
            column_x_offset = column_x * layer2_column_size[1]
            new_activations = layer2[column_y][column_x]
            layer2_bitset.set_bits_at(column_y_offset, column_x_offset,
                                      layer2_column_size[0], layer2_column_size[1],
                                      new_activations)
    # ====== 7. use layer2 bitset to obtain layer3 sparse fingerprint  ======
    time7_layer3 = time.time()
    layer3_sdr = layer2_to_layer3.compute_translation_invariant(layer2_bitset,
                                                                (layer2_column_size[0], layer2_column_size[1]))

    layer3.set_sparse(layer3_sdr)
    # ====== 8. update map  ======
    time8_update_map = time.time()
    if update_map:
        map_activity = layer3_to_map.infer(layer3, learn)
        map_activity.normalize()
    elif learn:
        # ====== associate a new fingerprint with current place cells on the map ======
        layer3_to_map.update_permanence(layer3, map_activity)
    time9_end = time.time()
    print("get_sensory_input", time2_encode_sensory_input - time1_get_sensory_input,
          "encode_sensory_input", time3_compute_layer1 - time2_encode_sensory_input,
          "compute_layer1", time4_vote - time3_compute_layer1,
          "vote", time5_sublayer2 - time4_vote,
          "sublayer2", time5_apical_learn - time5_sublayer2,
          "apical_learn", time6_join_layer2 - time5_apical_learn,
          "join_layer2", time7_layer3 - time6_join_layer2,
          "layer3", time8_update_map - time7_layer3,
          "update_map", time9_end - time8_update_map)


def dump():
    for r, row in enumerate(layer2):
        for c, cell in enumerate(row):
            print("l2(" + str(c) + "," + str(r) + ")=", cell)
    print("l3=", layer3)
    print("map=", map_activity)


fig, axs = plt.subplots(1, 2)
full_map = axs[0]
fov = axs[1]


def draw():
    full_map.cla()
    full_map.imshow(env)
    full_map.plot(x, y, 'ro')
    fov.cla()
    fov.imshow(get_fov())
    plt.draw()


dump()
draw()


def move(event):
    print(event.key)
    if event.key == 'right':
        move_agent(1, 0)
    elif event.key == 'left':
        move_agent(-1, 0)
    elif event.key == 'up':
        move_agent(0, 1)
    elif event.key == 'down':
        move_agent(0, -1)
    elif event.key == ' ':
        run()
    elif event.key == 'l':
        run(learn=True)
    elif event.key == 'u':
        run(learn=True, update_map=True)
        dump()
    else:
        return
    draw()


fig.canvas.mpl_connect('key_press_event', move)
plt.show()