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

robot_state_EOF = 11

env_id = 'ma_gym:HuRoSorting-v0'

env = gym.make(env_id, add_interac_flag=False)

total_timesteps = 10 ** 4

init_fixed = False
failure_traj = False
done = False

reward_stats = []
length_stats = []
ep_reward = 0
ep_length = 0
path = os.path.dirname (os.path.realpath (__file__))
state = env.reset(init_fixed)

state_action_list = []
traj = -1

print("State len: ", len(state))

for _ in tqdm(range(total_timesteps)):
    state_robot = state[:robot_state_EOF].copy()
    state_human = state[robot_state_EOF:].copy()
    state_robot_input = np.concatenate([state_robot.copy(), state_human.copy()])
    state_human_input = np.concatenate([state_human.copy(), state_robot.copy()])
    oloc_r, eefloc_r, pred_r = np.argmax(state_robot[:4]), np.argmax(state_robot[4:8]), np.argmax(state_robot[8:11])
    oloc_h, eefloc_h, pred_h = np.argmax(state_human[:4]), np.argmax(state_human[4:8]), np.argmax(state_human[8:11])
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
    next_state, reward, done, _ = env.step([actions_r, actions_h], verbose = 0)

    ep_reward += reward
    ep_length += 1

    next_state_robot = next_state[:robot_state_EOF].copy()
    next_state_human = next_state[robot_state_EOF:].copy()
    next_state_robot_input = np.concatenate([next_state_robot.copy(), next_state_human.copy()])
    next_state_human_input = np.concatenate([next_state_human.copy(), next_state_robot.copy()])

    if done:
        traj += 1
        with open(path+'/expert_data/'+str(traj)+'.txt','w') as file:
            file.write('\n'.join(' '.join(map(str, row)) for row in state_action_list))

        file.close()
        init_fixed = not init_fixed
        state = env.reset(init_fixed)
        reward_stats.append(ep_reward)
        length_stats.append(ep_length)
        done = False
        ep_reward = 0
        ep_length = 0
        state_action_list = []
    else:
        # one_hot_action = np.concatenate([env.get_one_hot(actions_r.item(), 6), env.get_one_hot(actions_h.item(), 6)])
        state_action_list.append(np.concatenate([state, [actions_r.item() * env.nAAgent + actions_h.item()]]))
        state = next_state

