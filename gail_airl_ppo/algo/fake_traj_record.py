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

env_id = 'ma_gym:HuRoSorting-v0'

env = gym.make(env_id)

total_timesteps = 10 ** 6

init_fixed = False
failure_traj = False
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

def failure_reset(env):
    random.seed(time())
    env._step_count = 0
    env.reward = env.step_cost
    env._agent_dones = False
    env.steps_beyond_done = None
    state = random.choice([[[2,2,2],[2,2,2]],
            [[3,3,2],[3,3,2]],
            [[1,random.choice([0,2,3]),random.choice([1,2])],[1,random.choice([0,2,3]),random.choice([1,2])]],
            [[0,random.choice([0,2,3]),0],[0,random.choice([0,2,3]),0]]])
    env.set_prev_obsv(0, env.vals2sid(state[0]))
    env.set_prev_obsv(1, env.vals2sid(state[1]))
    onehot = env.get_global_onehot(state)
    return onehot


if not failure_traj:
    state = env.reset(init_fixed)
else:
    state = failure_reset(env)

for _ in tqdm(range(total_timesteps)):
    state_robot = state[:11].copy()
    state_human = state[11:].copy()
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

    # print("State: ", next_state)
    # print("Reward: ", reward)
    # print("Done: ", done)

    next_state_robot = next_state[:11].copy()
    next_state_human = next_state[11:].copy()
    next_state_robot_input = np.concatenate([next_state_robot.copy(), next_state_human.copy()])
    next_state_human_input = np.concatenate([next_state_human.copy(), next_state_robot.copy()])

    trajs_robot_states.append(state_robot_input.copy())
    trajs_human_states.append(state_human_input.copy())
    trajs_robot_next_states.append(next_state_robot_input.copy())
    trajs_human_next_states.append(next_state_human_input.copy())
    trajs_robot_actions.append(actions_r)
    trajs_human_actions.append(actions_h)
    trajs_rewards.append(reward)
    trajs_dones.append(int(done))

    if done:
        init_fixed = not init_fixed
        if not failure_traj:
            state = env.reset(init_fixed)
        else: state = failure_reset(env)
        reward_stats.append(ep_reward)
        length_stats.append(ep_length)
        done = False
        ep_reward = 0
        ep_length = 0
    else:
        state = next_state

trajs_robot_states = torch.tensor(trajs_robot_states).float()
trajs_human_states = torch.tensor(trajs_human_states).float()
trajs_robot_next_states = torch.tensor(trajs_robot_next_states).float()
trajs_human_next_states = torch.tensor(trajs_human_next_states).float()
trajs_robot_actions = torch.tensor(trajs_robot_actions).float()
trajs_human_actions = torch.tensor(trajs_human_actions).float()
trajs_rewards = torch.tensor(trajs_rewards).float()
trajs_dones = torch.tensor(trajs_dones).float()

trajectories = {
    'robot_state': trajs_robot_states,
    'human_state': trajs_human_states,
    'robot_action': trajs_robot_actions,
    'human_action': trajs_human_actions,
    'reward': trajs_rewards,
    'done': trajs_dones,
    'robot_next_state': trajs_robot_next_states,
    'human_next_state': trajs_human_next_states
}

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
if not failure_traj:
    torch.save(trajectories, script_dir + f'/buffers/{env_id.split("-")[0].lower()}.pt')
else:   torch.save(trajectories, script_dir + f'/buffers/{env_id.split("-")[0].lower()}_failed.pt')
    
print(f'Reward Mean: {round(np.mean(reward_stats), 2)} | Reward Std: {round(np.std(reward_stats), 2)} | Episode Mean Length: {round(np.mean(length_stats), 2)} | Total Episodes: {len(reward_stats)}')

    