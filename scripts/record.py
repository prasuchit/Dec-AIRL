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
        # print(np.concatenate(list(obs.values())))
        states_rollout = {'robot': [], 'human': []}
        next_states_rollout = {'robot': [], 'human': []}
        actions_rollout = {'robot': [], 'human': []}
        rewards_rollout = {'robot': [], 'human': []}
        dones_rollout = {'robot': [], 'human': []}
        infos_rollout = {'robot': [], 'human': []}

        length_stats = []
        reward_stats = []

        length = 0
        reward = 0

        for step in tqdm(range(num_steps)):
            with torch.no_grad():
                actions, agents = {'robot': None, 'human': None}, ['robot','human']
                # for agent_id in range(self.n_agents):
                for agent_id in agents:
                    local_obs = obs[agent_id]
                    global_obs = np.concatenate([obs['robot'], obs['human']])
                    action, _, _ = self.models[agent_id].policy.forward(local_obs, global_obs)
                    actions[agent_id] = action.squeeze().cpu().numpy()

            next_obs, rewards, dones, info = self.env.step(actions)
            # print(rewards, actions, obs)
            action_robot = actions['robot']
            action_human = actions['human']
            states_rollout['robot'].append(obs['robot'])
            states_rollout['human'].append(obs['human'])
            next_states_rollout['robot'].append(next_obs['robot'])
            next_states_rollout['human'].append(next_obs['human'])
            actions_rollout['robot'].append(action_robot)
            actions_rollout['human'].append(action_human)
            rewards_rollout['robot'].append(rewards['robot'])
            rewards_rollout['human'].append(rewards['human'])
            dones_rollout['robot'].append(dones['robot'])
            dones_rollout['human'].append(dones['human'])
            infos_rollout['robot'].append(info['robot'])
            infos_rollout['human'].append(info['human'])

            rewards = (rewards['robot'] + rewards['human']) / 2
            done = dones['__all__']

            if done:
                obs = self.env.reset()

                length_stats.append(length)
                reward_stats.append(reward)

                length = 0
                reward = 0
            else:
                obs = next_obs

                length += 1
                reward += rewards
                
        # states_rollout = {'robot': states_rollout['robot'].cpu().numpy(), 'human': states_rollout['human'].cpu().numpy()}
        # next_states_rollout = {'robot': next_states_rollout['robot'].cpu().numpy(), 'human': next_states_rollout['human'].cpu().numpy()}
        # actions_rollout = torch.as_tensor(actions_rollout).float()
        # rewards_rollout = torch.as_tensor(rewards_rollout).float()
        # dones_rollout = torch.as_tensor(dones_rollout).float()

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
    parser.add_argument('--env', type=str, default='FeedingSawyerHuman-v1', help='Provide the env')
    parser.add_argument('--training_epochs', type=int, default=20, help='Total training epochs')
    args = parser.parse_args()

    env_id = args.env
    ppo = Record(env_id)

    ppo.load(path=f'{PACKAGE_PATH}/models/{env_id}')
    ppo.record(num_steps=10000)