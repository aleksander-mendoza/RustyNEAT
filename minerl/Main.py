from multiprocessing import freeze_support

import gym
import minerl
import logging
from matplotlib import pyplot as plt
import cv2
from minerl.data import BufferedBatchIter
import numpy as np
from incsfa import IncSFA2
import expansion as e
import sfa
from sfa import SFA


# logging.basicConfig(level=logging.DEBUG)

def plot_f(f):
    x = np.linspace(0, 10, 100)
    y = f(x)
    plt.plot(x, y)


def f(x):
    return np.sin(x) + np.sin(x * 8)


sfa = IncSFA2(2, 2, 2)

for i in range(1000000000):
    i = i / 100
    sfa.update(np.array([[f(i), 1]]))
exit()


def run():
    data = minerl.data.make('MineRLObtainDiamond-v0', data_dir='data')

    iterator = BufferedBatchIter(data)
    for current_state, action, reward, next_state, done in iterator.buffered_batch_iter(batch_size=1, num_epochs=1):
        for frame in current_state['pov']:
            # Normally frames have dimension (64,64,3)
            frame = cv2.resize(frame, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
            plt.clf()
            plt.imshow(frame)
            plt.pause(interval=0.001)


if __name__ == '__main__':
    freeze_support()
    run()
exit()

env = gym.make('MineRLNavigateDense-v0')
done = False
obs = env.reset()
net_reward = 0

while not done:
    env.render()
    action = env.action_space.noop()

    action['camera'] = [0, 0.03 * obs["compass"]["angle"]]
    action['back'] = 0
    action['forward'] = 1
    action['jump'] = 1
    action['attack'] = 1

    obs, reward, done, info = env.step(action)

    net_reward += reward
