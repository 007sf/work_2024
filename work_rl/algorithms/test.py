import sys
from scipy.signal import find_peaks
import torch
import matplotlib.pyplot as plt
sys.path.append("..")
from work_rl.env.grid_world_in_sssf import GridWorld
import random
import numpy as np
import argparse
import copy
import gymnasium
import env
# Example usage:
if __name__ == "__main__":
    # env = GridWorld()
    # state = env.reset()
    # for t in range(1000):
    #     env.render()
    #     action = np.random.randint(7)
    #
    #     next_state, reward, done, info = env.step(action)
    #     # state = copy.deepcopy(next_state)
    #
    #     print(f"Step: {t}, Action: {action}, State: {next_state},Reward: {reward}, Done: {done}")
    #
    # env.render(animation_interval=100)

    env = gymnasium.make("Gridworld-v0")