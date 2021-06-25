import rusty_neat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
import random
from rusty_neat import ndalgebra as nd

context = rusty_neat.make_gpu_context()
borders = mpimg.imread('map.png')
borders = borders.mean(2)  # RGB -> greyscale
borders = (1 - borders)*256
h, w = borders.shape
food = np.random.rand(h, w) * 256
food[food > 200] = 0
food[:150, :150] = 0
borders += food
borders = borders.clip(0, 255)
borders = borders.astype(np.ubyte)
borders = nd.from_numpy(borders, context=context)  # Move map contours to GPU
AGENT_ATTRIBUTES = 6  # This constant is imposed by environment implementation
LIDAR_ATTRIBUTES = 2  # This constant is imposed by environment implementation
ACTION_SPACE = 2  # This one is up to us, but environment will always use exactly two (rotation and movement action)
# We could increase action space in order to for example introduce memory for agents

lidar_angles = [-0.1, 0, 0.1]
lidar_steps = 30
step_length = 3.
population_size = 8
central_lidar_idx = 1
max_distance_change = 10
max_angle_change = 0.1
hunger_change_per_step = -10
initial_hunger = 200
funcs = ["identity", "sigmoid", "sin", "abs", "square", "gaussian", "floor", "fraction", "neg"]
agents = np.empty((population_size, AGENT_ATTRIBUTES-2+ACTION_SPACE), dtype=np.float32)
agents[:, 0] = 100  # x
agents[:, 1] = 100  # y
agents[:, 2] = 0  # angle
agents[:, 3] = initial_hunger  # hunger
agents[:, 4] = 0  # rotation action
agents[:, 5] = 0  # movement action
agents = nd.from_numpy(agents, context=context)
# Initially lidars should be all zero
lidars = nd.zeros((population_size, len(lidar_angles), LIDAR_ATTRIBUTES), dtype=nd.float32, context=context)

env = rusty_neat.envs.Evol(w, h,
                           hunger_change_per_step,
                           lidar_angles,
                           lidar_steps,
                           step_length,
                           context)
neat = rusty_neat.Neat32(len(lidar_angles) * LIDAR_ATTRIBUTES, ACTION_SPACE, funcs)

cppns = neat.new_cppns(population_size)
for _ in range(100):
    neat.mutate_population(cppns,
                           0.1,
                           0.2,
                           0.1,
                           0.1,
                           0.1,
                           0.01)
nets = [cppn.build_feed_forward_net() for cppn in cppns]

while True:

    imgplot = plt.imshow(borders.numpy(), cmap='gray')
    env(borders, agents, lidars)
    for agent in range(population_size):
        x, y, angle, hunger, action_rot, action_rot = agents[agent]
        agent_lidars = lidars[agent]
        net = nets[agent]
        if hunger <= 0:
            agents[agent] = [100, 100, 0, initial_hunger, 0, 0]
            new_cppn = random.choice(cppns).crossover(random.choice(cppns))
            cppns[agent] = new_cppn
            neat.mutate(new_cppn,
                        0.1,
                        0.2,
                        0.1,
                        0.1,
                        0.1,
                        0.01)
            nets[agent] = new_cppn.build_feed_forward_net()
        else:
            plt.scatter(x, y, color='r', marker="o")
            movement, rotation = net(list(agent_lidars.reshape(-1)))
            how_far_can_go = agent_lidars[central_lidar_idx, 0].item()
            qx, qy = x + math.sin(angle) * how_far_can_go, y + math.cos(angle) * how_far_can_go
            plt.scatter(qx, qy, color='y', marker=".")

    plt.pause(interval=0.01)
    plt.clf()
