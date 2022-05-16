#!/usr/local/bin/python3

# The MIT License (MIT)

# Copyright (c) 2022 Prasanth Suresh and Yikang Gui

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import sys
import argparse

import gym
import numpy as np
import torch
from tqdm import tqdm
path = os.path.dirname (os.path.realpath (__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

sys.path.append(PACKAGE_PATH)
from algo.ppo.ppo import Dec_Train, obs_as_tensor

''' This file records expert trajectories from a trained RL agent(s) for any ma-gym(https://github.com/prasuchit/ma-gym) environment '''

class Record(Dec_Train):
    def __init__(self, env_id, device='cpu', seed=1024):
        super().__init__(env_id, device=device, seed=seed)
    
    def record(self, num_steps=10000):
        obs = self.env.reset()

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

        for step in tqdm(range(num_steps)):
            with torch.no_grad():
                actions = []
                for agent_id in range(self.n_agents):
                    action, _, _ = self.models[agent_id].policy.forward(obs_as_tensor(obs, device=self.device), deterministic=True)
                    actions.append(action.item())

            new_obs, rewards, dones, infos = self.env.step(actions, verbose=0)
            # print(rewards, actions, obs)
            rewards = sum(rewards)
            dones = all(dones)

            states_rollout.append(obs)
            next_states_rollout.append(new_obs)
            actions_rollout.append(actions)
            rewards_rollout.append(rewards)
            dones_rollout.append([dones])
            infos_rollout.append(infos)

            if dones:
                obs = self.env.reset()

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
            os.mkdir(save_path)
        torch.save(trajectories, f'{save_path}/data.pt')    

        print(f'Collect Episodes: {len(length_stats)} | Avg Length: {round(np.mean(length_stats), 2)} | Avg Reward: {round(np.mean(reward_stats), 2)}')    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO forward reinforcement learning')
    parser.add_argument('--env', type=str, default='ma_gym:Checkers-v0', help='Provide the env')
    parser.add_argument('--training_epochs', type=int, default=20, help='Total training epochs')
    args = parser.parse_args()

    env_id = args.env
    ppo = Record(env_id)

    ppo.load(path=f'{PACKAGE_PATH}/models/{env_id}')
    ppo.record(num_steps=10000)