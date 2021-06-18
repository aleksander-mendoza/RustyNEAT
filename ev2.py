import rusty_neat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
import random

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
agents = np.empty((population_size, 4), dtype=np.float32)
agents[:, 0] = 100  # x
agents[:, 1] = 100  # y
agents[:, 2] = 0  # angle
agents[:, 3] = initial_hunger  # hunger
lidars = np.empty((population_size, len(lidar_angles), 2), dtype=np.float32)

env = rusty_neat.envs.Evol(borders,
                           w, h,
                           hunger_change_per_step,
                           lidar_angles,
                           lidar_steps,
                           step_length)
neat = rusty_neat.Neat32(len(lidar_angles) * 2, 2, funcs)

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

def norm(x):
    return math.tanh(x)


while True:

    imgplot = plt.imshow(env.borders, cmap='gray')
    env(agents, lidars)
    for agent in range(population_size):
        x, y, angle, hunger = agents[agent]
        agent_lidars = lidars[agent]
        net = nets[agent]
        if hunger <= 0:
            agents[agent] = [100, 100, 0, initial_hunger]
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
            movement, rotation = net.numpy(agent_lidars.reshape(-1))
            how_far_can_go = agent_lidars[central_lidar_idx, 0]
            qx, qy = x + math.sin(angle) * how_far_can_go, y + math.cos(angle) * how_far_can_go
            plt.scatter(qx, qy, color='y', marker=".")
            movement = min(max(0., norm(movement) * max_distance_change), how_far_can_go)
            rotation = norm(rotation) * max_angle_change
            angle = rotation + angle
            qx, qy = x + math.sin(angle) * movement, y + math.cos(angle) * movement
            agents[agent, 0] = int(qx)
            agents[agent, 1] = int(qy)
            agents[agent, 2] = angle

    plt.pause(interval=0.01)
    plt.clf()
