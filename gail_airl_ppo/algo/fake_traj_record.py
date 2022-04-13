from tabnanny import verbose
import gym
import argparse
from stable_baselines3.common.utils import obs_as_tensor
import torch
from tqdm import tqdm
import numpy as np
import os


env = gym.make('ma_gym:HuRoSorting-v0')

total_timesteps = 10 ** 2

init_fixed = True
overwrite_ppo = True
done = False
state = env.reset(init_fixed)
state_robot = state[:11].copy()
state_human = state[11:].copy()
state_robot_input = np.concatenate([state_robot.copy(), state_human.copy()])
state_human_input = np.concatenate([state_human.copy(), state_robot.copy()])
trajs_robot_states = []
trajs_human_states = []
trajs_robot_next_states = []
trajs_human_next_states = []
trajs_robot_actions = []
trajs_human_actions = []
trajs_rewards = []
trajs_dones = []

reward_stats = []
length_stats = []
ep_reward = 0
ep_length = 0

OLOC = {
    0: 'Unknown',
    1: 'OnConveyor',
    2: 'InFront',
    3: 'AtHome'
}
EEFLOC = {
    0: 'InBin',
    1: 'OnConveyor',
    2: 'InFront',
    3: 'AtHome'
}
PRED = {
    0: 'Unknown',
    1: 'Bad',
    2: 'Good'
}
ACTIONS = {
    0: 'Noop',
    1: 'Detect',
    2: 'Pick',
    3: 'Inspect',
    4: 'PlaceOnConveyor',
    5: 'PlaceinBin'
}



for _ in tqdm(range(total_timesteps)):

    