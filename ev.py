import rusty_neat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
import random

img = mpimg.imread('map.png')
img = img.mean(2)
funcs = ["identity", "sigmoid", "sin", "abs", "square", "gaussian", "floor", "fraction", "neg"]
initial_hunger = 5


class Agent:

    def __init__(self, x, y, cppn):
        self.x = x
        self.y = y
        self.u = 0.
        self.prev_dist = 0.
        self.prev_angle = 0.
        self.cppn = cppn
        self.hunger = 5
        self.net = cppn.build_feed_forward_net()

    def lidar(self, dist, angle):
        angle = self.u + angle
        qx, qy = self.x + math.sin(angle) * dist, self.y + math.cos(angle) * dist
        return int(qx), int(qy)

    def move(self, dist, angle):
        self.prev_dist = dist
        self.prev_angle = angle
        self.x, self.y = self.lidar(dist, angle)
        self.u = (self.u + angle) % math.tau


hunger_limit = 10
img = img.T
h, w = img.shape
lidars = [0]
distance = 30
step_length = 5.
neat = rusty_neat.Neat32(len(lidars) + 2, 2, funcs)
population = neat.new_cppns(16)
for _ in range(100):
    neat.mutate_population(population,
                           0.1,
                           0.2,
                           0.1,
                           0.1,
                           0.1,
                           0.01)
population = [Agent(100, 100, cppn) for cppn in population]


def norm(x):
    return math.tanh(x)


# hyper_neat_inputs = [[1, 1]]
# hyper_neat_outputs = [[1, 1], [-1, 1]]
net_input = [0.0] * (len(lidars) + 2)
central_lidar_idx = 0
max_distance_change = 10
max_angle_change = 0.1
food = np.random.rand(h, w)
food[food < 0.5] = 0
food[:150, :150] = 0
hunger_per_step = 0.1
food_eating_span = 10
while True:

    imgplot = plt.imshow(img - food, cmap='gray')

    for agent in population:
        plt.scatter(agent.x, agent.y, color='r', marker="o")
        touched_food = food[agent.y-food_eating_span:agent.y+food_eating_span,
                            agent.x-food_eating_span:agent.x+food_eating_span]
        agent.hunger += hunger_per_step - touched_food.sum()
        touched_food.fill(0)
        if agent.hunger > hunger_limit:
            agent.x = 100
            agent.y = 100
            agent.cppn = random.choice(population).cppn.crossover(random.choice(population).cppn)
            neat.mutate(agent.cppn,
                        0.1,
                        0.2,
                        0.1,
                        0.1,
                        0.1,
                        0.01)
            agent.net = agent.cppn.build_feed_forward_net()
            agent.hunger = initial_hunger
        else:
            for li, lidar_angle in enumerate(lidars):
                for i in range(distance):
                    net_input[li] = i * step_length
                    lx, ly = agent.lidar(net_input[li], lidar_angle)
                    if lx <= 0 or lx > w or ly <= 0 or ly > h:
                        break
                    # plt.scatter(lx, h-ly, color='y', marker=".")
                    if not img[ly, lx] > 0.5:
                        break
                plt.scatter(lx, ly, color='y', marker=".")
            net_input[len(lidars)] = agent.prev_angle
            net_input[len(lidars) + 1] = agent.prev_dist

            movement, rotation = agent.net(net_input)
            how_far_can_go = net_input[central_lidar_idx]
            movement = min(max(0., norm(movement) * max_distance_change), how_far_can_go)
            rotation = norm(rotation) * max_angle_change
            agent.move(movement, rotation)

    plt.pause(interval=0.01)
    plt.clf()
