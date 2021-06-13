import rusty_neat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

img = mpimg.imread('map.png')

funcs = ["identity", "sigmoid", "sin", "abs", "square", "gaussian", "floor", "fraction", "neg"]


class Agent:

    def __init__(self, x, y, cppn):
        self.x = x
        self.y = y
        self.u = 0.
        self.cppn = cppn
        self.net = cppn.build_feed_forward_net()

    def lidar(self, dist, angle):
        angle = self.u + angle
        qx, qy = self.x+math.sin(angle)*dist, self.y+math.cos(angle)*dist
        return int(qx), int(qy)


h, w, _ = img.shape
lidars = [0]
distance = 10
step_length = 3.
neat = rusty_neat.Neat32(len(lidars), 2, funcs)
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
lidar_distances = [0] * len(lidars)
central_lidar_idx = 0
max_lidar_length = step_length * distance
while True:
    imgplot = plt.imshow(img)
    for agent in population:
        plt.scatter(agent.x, h-agent.y, color='r', marker="o")

        for li, lidar_angle in enumerate(lidars):
            for i in range(distance):
                lidar_distances[li] = i * step_length
                lx, ly = agent.lidar(lidar_distances[li], lidar_angle)
                if lx < 0 or lx > w or ly < 0 or ly > h:
                    break
                # plt.scatter(lx, h-ly, color='y', marker=".")
                if not (img[lx, ly] == [1, 1, 1]).all():
                    break
        plt.scatter(lx, h - ly, color='y', marker=".")
        movement, rotation = agent.net(lidar_distances)
        how_far_can_go = lidar_distances[central_lidar_idx]
        movement = min(max(0., norm(movement) * max_lidar_length), how_far_can_go)
        rotation = norm(rotation)
        agent.x, agent.y = agent.lidar(movement, rotation)
        agent.u = (agent.u + rotation) % math.tau

    plt.pause(interval=0.01)
    plt.clf()
