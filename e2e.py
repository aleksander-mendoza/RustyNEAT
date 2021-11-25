import rusty_neat
from rusty_neat import htm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage


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
x, y = 64, 64
speed = 10

vote_kernel_size = np.array([2, 2])
vote_kernel_stride = np.array([1, 1])
vote_num_of_layer1_columns_per_layer2_column = vote_kernel_size.prod()
vote_threshold = int(vote_num_of_layer1_columns_per_layer2_column * 0.4)
image_patch_width, image_patch_height = 16, 16
image_patch_size = np.array([image_patch_height, image_patch_width])
sensor_column_size = np.array([image_patch_height, image_patch_width * len(gabor_filters)])
sensor_column_count = np.array([4, 4])
layer1_column_count = sensor_column_count  # This must be the same
layer1_column_size = np.array([16, 16])
layer1_column_card = int(layer1_column_size.prod() * 0.05)
layer2_column_count = layer1_column_count // vote_kernel_stride - vote_kernel_size + 1
layer2_column_size = layer1_column_size
layer2_column_card = layer1_column_card
sensor_to_layer1_synapses = int(sensor_column_size.prod() * 0.7)
layer2_total_shape = layer2_column_count * layer2_column_size
layer2_total_size = layer2_total_shape.prod()
layer3_size = layer2_total_size * 4
layer3_card = int(layer3_size * 0.05)
layer3_synapse_span = layer2_column_size * 3
layer2_to_layer3_synapses = int(layer2_total_size * 0.05)
map_size = 64 * 64
map_card = int(map_size * 0.2)
layer3_to_map_synapses = int(layer3_size * 0.8)


def make_sensor_to_layer1_column():
    return htm.CpuHTM2(sensor_column_size.prod(),
                       layer1_column_size.prod(),
                       layer1_column_card,
                       sensor_to_layer1_synapses)


sensor_layer = [[htm.CpuBitset(sensor_column_size.prod()) for _ in range(sensor_column_count[1])] for _ in
                range(sensor_column_count[0])]
sensor_single_column_encoder = htm.EncoderBuilder()
sensor_single_filter_encoder = [sensor_single_column_encoder.add_bits(image_patch_size.prod()) for _ in gabor_filters]
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
layer2_to_layer3 = htm.CpuDG2_2d((layer2_total_shape[0], layer2_total_shape[1]), layer3_size, layer3_card,
                                 (layer3_synapse_span[0], layer3_synapse_span[1]), layer2_to_layer3_synapses)
layer3_to_map = htm.CpuBigHTM(layer3_size, map_size, map_card)
map_to_map = htm.CpuHOM(4, map_size)


def agent_moved(self, x_displacement, y_displacement):
    pass


def get_fov(img_src=env):
    return img_src[y - fov_span:y + fov_span, x - fov_span:x + fov_span]


def run(learn=False):
    fov_img = get_fov(env_grayscale)
    fov_img = fov_img.astype(float) / 255

    def convolve(img, kernel):
        img = ndimage.convolve(img, kernel, mode='constant')
        img = img > gabor_threshold
        return img

    gabor_edges = [convolve(fov_img, kernel) for kernel in gabor_filters]
    # visualize_img_list(fov_img, gabor_edges, gabor_filters)
    for column_y in range(sensor_column_count[0]):
        for column_x in range(sensor_column_count[1]):
            sensor_layer_patch = sensor_layer[column_y][column_x]
            sensor_layer_patch.clear()
            patch = np.array([[column_y, column_y + 1], [column_x, column_x + 1]]) * image_patch_size
            for encoder, filtered_img in zip(sensor_single_filter_encoder, gabor_edges):
                img_patch = filtered_img[patch[0, 0]:patch[0, 1], patch[1, 0]:patch[1, 1]]
                assert np.array(img_patch.shape).prod() == sensor_column_size.prod() / len(gabor_filters)
                img_patch = img_patch.reshape(image_patch_size.prod())
                img_patch = img_patch.tolist()
                encoder.encode(sensor_layer_patch, img_patch)
                assert encoder.len == len(img_patch)
    assert (layer1_column_count == sensor_column_count).all()
    layer1 = [[None] * layer1_column_count[1]] * layer1_column_count[0]
    for column_y in range(sensor_column_count[0]):
        for column_x in range(sensor_column_count[1]):
            sensor_layer_patch = sensor_layer[column_y][column_x]
            spacial_pooler = sensor_to_layer1[column_y][column_x]
            layer1_sdr = spacial_pooler.compute(sensor_layer_patch)
            layer1[column_y][column_x] = layer1_sdr

    cast_votes = htm.vote_conv2d(layer1, layer2_column_card, vote_threshold,
                                 (vote_kernel_stride[0], vote_kernel_stride[1]),
                                 (vote_kernel_size[0], vote_kernel_size[1]))
    assert len(cast_votes) == len(layer2) == layer2_column_count[0]
    assert len(cast_votes[0]) == len(layer2[0]) == layer2_column_count[1]
    for column_y in range(layer2_column_count[0]):
        for column_x in range(layer2_column_count[1]):
            new_votes = cast_votes[column_y][column_x]
            prev_activations = layer2[column_y][column_x]
            prev_activations.randomly_extend_from(new_votes)

    if learn:
        apical_feedback = htm.vote_conv2d_transpose(cast_votes,
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

    layer2_bitset.clear()
    for column_y in range(layer2_column_count[0]):
        for column_x in range(layer2_column_count[1]):
            column_y_offset = column_y * layer2_column_size[0]
            column_x_offset = column_x * layer2_column_size[1]
            new_activations = layer2[column_y][column_x]
            layer2_bitset.set_bits_at(column_y_offset, column_x_offset,
                                      layer2_column_size[0], layer2_column_size[1],
                                      new_activations)
    layer3_sdr = layer2_to_layer3.compute_translation_invariant(layer2_bitset,
                                                                (layer2_column_size[0], layer2_column_size[1]))
    layer3.set_sparse(layer3_sdr)
    if learn:
        layer3_to_map.update_permanence(layer3, map_activity)
        

run(learn=True)

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


draw()


def move(event):
    global x, y
    if event.key == 'right':
        x += speed
    elif event.key == 'left':
        x -= speed
    elif event.key == 'up':
        y += speed
    elif event.key == 'down':
        y -= speed
    elif event.key == ' ':
        run()
    draw()


fig.canvas.mpl_connect('key_press_event', move)
plt.show()
