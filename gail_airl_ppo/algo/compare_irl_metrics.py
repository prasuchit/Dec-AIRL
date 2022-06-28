import pickle
import sys
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
# sys.path.append(os.path.dirname(__file__)) #<-- absolute dir the script is in
from airl_sb_ppo_dec import AIRL

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

env_id = 'ma_gym:HuRoSorting-v0'

airl = AIRL(env_id=env_id, buffer_r_exp=None, buffer_h_exp=None, device='cpu', seed=None, 
                load_existing=None, trainpath=None, eval_interval=None)

path = os.getcwd()+f'/gail-airl-ppo/gail_airl_ppo/algo/models_airl/04-30-2022-01-45/'
disc = 'disc_2830336_88.pt'
actor_r = 'actor_r_2830336_88'
actor_h = 'actor_h_2830336_88'

airl.disc.load_state_dict(torch.load(f'{path}/{disc}'))
airl.actor_r.set_parameters(f'{path}/{actor_r}',  device='cpu')
airl.actor_h.set_parameters(f'{path}/{actor_h}', device='cpu')

env = gym.make(env_id)

total_timesteps = 10 ** 4

init_fixed = False
failure_traj = False
done = False
ep_reward = 0
ep_length = 0

reward_dict = {}

_ = env.reset()

for s_r in range(env.nSAgent):
    oloc_r, eefloc_r, pred_r = env.sid2vals(s_r)
    if env.isValidState(oloc_r, eefloc_r, pred_r):
        for s_h in range(env.nSAgent):
            oloc_h, eefloc_h, pred_h = env.sid2vals(s_h)
            if env.isValidState(oloc_h, eefloc_h, pred_h):
                state = env.check_interaction(env.get_global_onehot([[oloc_r, eefloc_r, pred_r], [oloc_h, eefloc_h, pred_h]]))

                state_robot = state[:robot_state_EOF].copy()
                state_human = state[robot_state_EOF:].copy()
                state_robot_input = np.concatenate([state_robot.copy(), state_human.copy()])
                state_human_input = np.concatenate([state_human.copy(), state_robot.copy()])

                state_robot_input = torch.tensor(state_robot_input)[None, :].float()
                state_human_input = torch.tensor(state_human_input)[None, :].float()

                # oloc_r, eefloc_r, pred_r = np.argmax(state_robot[:4]), np.argmax(state_robot[4:8]), np.argmax(state_robot[8:11])
                # oloc_h, eefloc_h, pred_h = np.argmax(state_human[:4]), np.argmax(state_human[4:8]), np.argmax(state_human[8:11])
                action_r, _, action_r_log_prob = airl.actor_r.policy.forward(state_robot_input, deterministic=True)
                action_h, _, action_h_log_prob = airl.actor_h.policy.forward(state_human_input, deterministic=True)

                log_probs = action_r_log_prob + action_h_log_prob

                # if state[-1] == 1:
                #     print(f"Interaction! oloc_r: {oloc_r}, eefloc_r: {eefloc_r}, pred_r: {pred_r}, action_r: {action_r.item()} ; oloc_h: {oloc_h}, eefloc_h: {eefloc_h}, pred_h: {pred_h}, action_h: {action_h.item()}")

                env.set_prev_obsv(0,s_r)
                env.set_prev_obsv(1,s_h)

                next_state, reward, done, _ = env.step([action_r, action_h], verbose = 0)
                action_r_onehot = torch.nn.functional.one_hot(action_r.long(), num_classes=6).float()
                action_h_onehot = torch.nn.functional.one_hot(action_h.long(), num_classes=6).float()
                global_action = torch.cat((action_r_onehot, action_h_onehot), dim=1)
                disc_reward = airl.disc.calculate_reward(state_robot_input, torch.tensor([int(done)])[None, :].float(), log_probs[:, None], torch.tensor(next_state)[None, :].float(), global_action).squeeze()
                if reward < 0:
                    print(f"Reward is: {reward}")
                
                _ = env.reset()
                reward_dict[(s_r,s_h,action_r.item(),action_h.item())] = {'Env_reward': reward, 'Disc_reward': round(disc_reward.item(), 4), 'Difference': (round(disc_reward.item(), 4) - reward)}

with open(os.path.dirname(__file__) + '/results/reward_dict.pickle', 'wb') as handle:
    pickle.dump(reward_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)