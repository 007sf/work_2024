import sys

import torch

sys.path.append("..")
from work_rl.env.grid_world_in_sssf import GridWorld
import random
import numpy as np
import argparse
import copy
# Example usage:
if __name__ == "__main__":
    env = GridWorld()
    state = env.reset()
    for t in range(1000):
        env.render()
        action = np.random.randint(5)

        next_state, reward, done, info = env.step(action)
        # state = copy.deepcopy(next_state)

        print(f"Step: {t}, Action: {action}, State: {next_state},Reward: {reward}, Done: {done}")

    env.render(animation_interval=100)
    # # # env.render(animation_interval=100)
    # # # grid_data = np.zeros((7, 7))
    # # # selected_positions = (2,1)
    # # # selected = selected_positions[0]+selected_positions[1]*7
    # # # grid_data.flat[selected] = 1
    # # # print(grid_data)
    # # a = torch.tensor([(0,1)])
    # # print(a.shape)
    # mag = np.array([[2,1],[1,2]])
    # print(mag.reshape(-1))
