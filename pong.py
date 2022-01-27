import gym
from rusty_neat import ndalgebra as nd, ecc, htm

env = gym.make("Pong-ram-v0")
env.reset()
print(env.env.get_action_meanings())
for _ in range(1000):
    env.render()
    # action = env.action_space.sample()
    # print(action)
    action = int(input())
    action = max(min(action, 3), 2)
    ram, reward, done, info = env.step(action)  # take a random action
    cpu_score = ram[13]  # computer/ai opponent score
    player_score = ram[14]  # your score
    cpu_paddle_y = ram[21]  # Y coordinate of computer paddle
    player_paddle_y = ram[51]  # Y coordinate of your paddle
    ball_x = ram[49]  # X coordinate of ball
    ball_y = ram[54]  # Y coordinate of ball

env.close()
