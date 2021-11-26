import rusty_neat
from rusty_neat import htm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage
import time
import cv2 as cv

env = cv.imread('map.jpeg')
env_grayscale = cv.Canny(env, threshold1=300, threshold2=400)
env = np.array(env)
env_grayscale = np.array(env_grayscale)
fov_span = 16
fov_size = np.array([fov_span * 2, fov_span * 2])
x, y = 64, 64
speed = 8
layer2_total_shape = fov_size
layer2_total_size = layer2_total_shape.prod()
layer3_size = fov_size.prod() * 4
layer3_card = int(layer3_size * 0.005)
layer3_synapse_span = np.array([speed * 3, speed * 3])
layer2_to_layer3_synapses = int(layer2_total_size * 0.05)
map_size = 64 * 64
map_card = int(map_size * 0.02)
layer3_to_map_synapses = int(layer3_size * 0.8)
sensor_encoder = htm.EncoderBuilder()
fov_encoder = sensor_encoder.add_bits(layer2_total_size)
layer2_bitset = htm.CpuBitset2d(layer2_total_shape[0], layer2_total_shape[1])
layer2_to_layer3 = htm.CpuDG2_2d((layer2_total_shape[0], layer2_total_shape[1]), layer3_size, layer3_card,
                                 (layer3_synapse_span[0], layer3_synapse_span[1]), layer2_to_layer3_synapses)
layer3 = htm.CpuInput(layer3_size)
map_activity = htm.CpuSDR()
map_activity.add_unique_random(map_card, 0, map_size)
map_activity.normalize()
layer3_to_map = htm.CpuBigHTM(layer3_size, map_size, map_card)
layer3_to_map.activation_threshold = int(10)
map_to_map = htm.CpuHOM(4, map_size)


def move_agent(steps_x, steps_y):
    steps = np.array([steps_y, steps_x]) * speed
    global x, y
    y += steps[0]
    x += steps[1]


def get_fov(img_src=env):
    img = img_src[y - fov_span:y + fov_span, x - fov_span:x + fov_span]
    assert (img.shape[0:2] == fov_size).all()
    return img


prev_layer2_activities = []
prev_layer3_activities = []
prev_map_activities = []


def run(learn=False, update_map=False):
    global map_activity
    # ====== 1. get sensory inputs ======
    time1_get_sensory_input = time.time()
    fov_img = get_fov(env_grayscale)
    # ====== 2. encode sensory inputs ======
    time2_encode_sensory_input = time.time()
    img = (fov_img > 0).reshape(layer2_total_size)
    img = img.tolist()
    layer2_bitset.clear()
    fov_encoder.encode(layer2_bitset, img)
    # ====== 7. use layer2 bitset to obtain layer3 sparse fingerprint  ======
    time7_layer3 = time.time()
    layer3_sdr = layer2_to_layer3.compute_translation_invariant(layer2_bitset, (speed, speed))
    print("l2 overlap over time=",
          [layer2_bitset.overlap(prev_layer2_sdr) for prev_layer2_sdr in prev_layer2_activities])
    print("l2 cardinality over time=", [prev_layer2_sdr.cardinality for prev_layer2_sdr in prev_layer2_activities])
    prev_layer2_activities.append(layer2_bitset.clone())
    print("l3=", layer3_sdr)
    print("l3 overlap=", layer3.overlap(layer3_sdr), "card=", layer3_sdr.cardinality)
    print("l3 overlap over time=", [layer3_sdr.overlap(prev_layer3_sdr) for prev_layer3_sdr in prev_layer3_activities])
    prev_layer3_activities.append(layer3_sdr)
    layer3.set_sparse(layer3_sdr)
    # ====== 8. update map  ======
    time8_update_map = time.time()
    new_map_activity = layer3_to_map.infer(layer3)
    new_map_activity.normalize()
    print("map overlap=", map_activity.overlap(new_map_activity), "card=", map_activity.cardinality)
    print("map=", new_map_activity)
    print("map overlap over time=", [new_map_activity.overlap(prev_map_sdr) for prev_map_sdr in prev_map_activities])
    prev_map_activities.append(new_map_activity)
    if update_map:
        map_activity = new_map_activity
    if learn:
        # ====== associate a new fingerprint with current place cells on the map ======
        layer3_to_map.update_permanence(layer3, map_activity)
    time9_end = time.time()
    # print("get_sensory_input", time2_encode_sensory_input - time1_get_sensory_input,
    #       "encode_sensory_input", time7_layer3 - time2_encode_sensory_input,
    #       "layer3", time8_update_map - time7_layer3,
    #       "update_map", time9_end - time8_update_map)


def dump():
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
