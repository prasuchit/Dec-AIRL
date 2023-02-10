from tabnanny import verbose
import gym
import argparse
from stable_baselines3.common.utils import obs_as_tensor
import torch
from tqdm import tqdm
import numpy as np
import os
import random
from time import time
path = os.path.dirname (os.path.realpath (__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

''' NOTE: This file is now outdated. May not work correctly. Feel free to update and use it.'''

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

actions_r = None
actions_h = None

robot_state_EOF = 12

env_id = 'ma_gym:DecHuRoSorting-v0'

env = gym.make(env_id)

total_timesteps = 10 ** 4

init_fixed = False
failure_traj = False
done = False
if not failure_traj:
    obs = env.reset()
else:
    obs = env.failure_reset()

states_rollout = []
next_states_rollout = []
actions_rollout = []
rewards_rollout = []
dones_rollout = []
infos_rollout = []

length_stats = []
reward_stats = []

length = 0
reward = 0

for step in tqdm(range(total_timesteps)):
    with torch.no_grad():
        actions = []
        oloc_r, eefloc_r, pred_r = np.argmax(obs[0][:4]), np.argmax(obs[0][4:8]), np.argmax(obs[0][8:11])
        oloc_h, eefloc_h, pred_h = np.argmax(obs[1][:4]), np.argmax(obs[1][4:8]), np.argmax(obs[1][8:11])
        # Independent states
        ##### ROBOT ########
        if oloc_r == pred_r == 0:   # Unknown
            actions_r = torch.tensor(1) # Detect
        elif oloc_r == 1 and pred_r != 0 and eefloc_r != 1: # onion onconv, eef not onconv, pred known
            actions_r = torch.tensor(2) # Pick
        elif oloc_r == eefloc_r == 3 and pred_r == 2:   # Athome, good
            actions_r = torch.tensor(3) # Inspect
        elif oloc_r == eefloc_r == 2 and pred_r == 2:   # Infront, good
            actions_r = torch.tensor(4) # Placeonconv
        elif oloc_r == eefloc_r == 3 and pred_r == 1:   # Athome, bad
            actions_r = torch.tensor(5) # Placeinbin
        elif oloc_r == eefloc_r == 2 and pred_r == 1:   # Infront, bad
            actions_r = torch.tensor(5) # Placeinbin
        ##### HUMAN ########
        if oloc_h == pred_h == 0:   # Unknown
            actions_h = torch.tensor(1) # Detect
        elif oloc_h == 1 and pred_h != 0 and eefloc_h != 1: # onion onconv, eef not onconv, pred known
            actions_h = torch.tensor(2) # Pick
        elif oloc_h == eefloc_h == 3 and pred_h == 2:   # Athome, good
            actions_h = torch.tensor(3) # Inspect
        elif oloc_h == eefloc_h == 2 and pred_h == 2:   # Infront, good
            actions_h = torch.tensor(4) # Placeonconv
        elif oloc_h == eefloc_h == 3 and pred_h == 1:   # Athome, bad
            actions_h = torch.tensor(5) # Placeinbin
        elif oloc_h == eefloc_h == 2 and pred_h == 1:   # Infront, bad
            actions_h = torch.tensor(5) # Placeinbin
        
        # Interaction states
        ###### JOINT ACTIONS ######
        if not failure_traj:
            if oloc_r == pred_r == oloc_h == pred_h == 0:   # Both unknown
                actions_r = torch.tensor(0) # Noop
                actions_h = torch.tensor(1) # Detect
            elif oloc_r == oloc_h == 1 and (pred_r != 0 and pred_h != 0) and (eefloc_r != 1 and eefloc_h != 1): # Both onion on conv
                actions_r = torch.tensor(0) # Noop
                actions_h = torch.tensor(2) # Pick
            elif oloc_r == eefloc_r == oloc_h == eefloc_h == 2 and (pred_r == pred_h == 2): # Both infront, good
                actions_r = torch.tensor(0) # Noop
                actions_h = torch.tensor(4) # Placeonconv
        else:
            if oloc_r == pred_r == oloc_h == pred_h == 0:   # Both unknown
                actions_r = torch.tensor(1) # Detect
                actions_h = torch.tensor(1) # Detect
            elif oloc_r == oloc_h == 1 and (pred_r != 0 and pred_h != 0) and (eefloc_r != 1 and eefloc_h != 1): # Both onion on conv
                actions_r = torch.tensor(2) # Pick
                actions_h = torch.tensor(2) # Pick
            elif oloc_r == eefloc_r == oloc_h == eefloc_h == 2 and (pred_r == pred_h == 2): # Both infront, good
                actions_r = torch.tensor(4) # Placeonconv
                actions_h = torch.tensor(4) # Placeonconv
            elif oloc_r == eefloc_r == oloc_h == eefloc_h == 3 and (pred_r == pred_h == 2): # Both athome, good
                actions_r = torch.tensor(4) # Placeonconv
                actions_h = torch.tensor(4) # Placeonconv

        assert actions_r != None, f"Check the exception oloc_r: {OLOC[oloc_r]}, eefloc_r: {EEFLOC[eefloc_r]}, pred_r: {PRED[pred_r]}, actions_r: {ACTIONS[actions_r]}"
        assert actions_h != None, f"Check the exception oloc_h: {OLOC[oloc_h]}, eefloc_r: {EEFLOC[eefloc_h]}, pred_r: {PRED[pred_h]}, actions_h: {ACTIONS[actions_h]}"
        actions.append(actions_r.item())
        actions.append(actions_h.item())

    new_obs, rewards, dones, infos = env.step(actions, verbose=0)
    # print(rewards, actions, obs)
    rewards = sum(rewards) / 2
    dones = all(dones)

    states_rollout.append(obs)
    next_states_rollout.append(new_obs)
    actions_rollout.append(actions)
    rewards_rollout.append(rewards)
    dones_rollout.append([dones])
    infos_rollout.append(infos)

    if dones:
        if not failure_traj:
            obs = env.reset()
        else:
            obs = env.failure_reset()

        length_stats.append(length)
        reward_stats.append(reward)

        length = 0
        reward = 0
    else:
        obs = new_obs

        length += 1
        reward += rewards
        
states_rollout = torch.tensor(states_rollout).float()
next_states_rollout = torch.tensor(next_states_rollout).float()
actions_rollout = torch.tensor(actions_rollout).float()
rewards_rollout = torch.tensor(rewards_rollout).float()
dones_rollout = torch.tensor(dones_rollout).float()

trajectories = {
'state': states_rollout,
'action': actions_rollout,
'reward': rewards_rollout,
'done': dones_rollout,
'next_state': next_states_rollout
}

save_path = f'{PACKAGE_PATH}/buffers/{env_id}'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
torch.save(trajectories, f'{save_path}/data.pt')    

print(f'Collect Episodes: {len(length_stats)} | Avg Length: {round(np.mean(length_stats), 2)} | Avg Reward: {round(np.mean(reward_stats), 2)}')