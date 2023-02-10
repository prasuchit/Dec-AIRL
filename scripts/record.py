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
import time 
import gym
import numpy as np
import torch
from tqdm import tqdm
path = os.path.dirname (os.path.realpath (__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

sys.path.append(PACKAGE_PATH)
from algo.ppo.ppo import Dec_Train, obs_as_tensor

''' This file records expert trajectories from a trained Dec-PPO agent(s) for any ma-gym and assistive-gym environments '''

class Record(Dec_Train):
    def __init__(self, env_id, device='cpu', seed=1024):
        super().__init__(env_id, device=device, seed=seed)
    
    def record(self, num_steps=10000, save_env_id=''):
        self.env.seed(int(time.time()))
        obs = self.env.reset()

        states_rollout = {agent_id: [] for agent_id in self.agents}
        next_states_rollout = {agent_id: [] for agent_id in self.agents}
        actions_rollout = {agent_id: [] for agent_id in self.agents}
        rewards_rollout = {agent_id: [] for agent_id in self.agents}
        dones_rollout = {agent_id: [] for agent_id in self.agents}
        infos_rollout = {agent_id: [] for agent_id in self.agents}

        length_stats = []
        reward_stats = []

        length = 0
        reward = 0

        for step in tqdm(range(num_steps)):
            with torch.no_grad():
                    actions = {agent_id: None for agent_id in self.agents}
                    for agent_id in self.agents:
                        local_obs = obs[agent_id]
                        global_obs = np.concatenate([obs[agent_id] for agent_id in self.agents])
                        action, value, log_prob = self.models[agent_id].policy.forward(local_obs, global_obs)
                        if len(action.shape) == 2:
                            actions[agent_id] = action.squeeze().cpu().numpy()
                        elif len(action.shape) == 1:
                            actions[agent_id] = action.cpu().numpy()
                    
            if self.assistive_gym:
                new_obs, rewards, dones, infos = self.env.step(actions)
            else:
                new_obs, rewards, dones, infos = self.env.step([actions[action] for action in actions])

            for agent_id in self.agents:
                states_rollout[agent_id].append(obs[agent_id])
                next_states_rollout[agent_id].append(new_obs[agent_id])
                actions_rollout[agent_id].append(actions[agent_id])
                rewards_rollout[agent_id].append(rewards[agent_id])
                dones_rollout[agent_id].append(dones[agent_id])
                infos_rollout[agent_id].append({})

            # print(rewards, actions, obs)
            if not self.assistive_gym:
                    rewards = sum(rewards) / 2
                    dones = all(dones)
            else:
                rewards = (rewards['robot'] + rewards['human']) / 2
                dones = dones['__all__']

            if dones:
                self.env.seed(int(time.time()))
                obs = self.env.reset()

                length_stats.append(length)
                reward_stats.append(reward)

                length = 0
                reward = 0
            else:
                obs = new_obs

                length += 1
                reward += rewards
        
        for agent_id in self.agents:
            states_rollout[agent_id] = torch.tensor(states_rollout[agent_id]).float()
            next_states_rollout[agent_id] = torch.tensor(next_states_rollout[agent_id]).float()
            actions_rollout[agent_id] = torch.tensor(actions_rollout[agent_id]).float()
            rewards_rollout[agent_id] = torch.tensor(rewards_rollout[agent_id]).float()
            dones_rollout[agent_id] = torch.tensor(dones_rollout[agent_id]).float()

        trajectories = {
        'state': states_rollout,
        'action': actions_rollout,
        'reward': rewards_rollout,
        'done': dones_rollout,
        'next_state': next_states_rollout
        }

        save_path = f'{PACKAGE_PATH}/buffers/{save_env_id}'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(trajectories, f'{save_path}/trajectory.pt')    

        print(f'Collect Episodes: {len(length_stats)} | Avg Length: {round(np.mean(length_stats), 2)} | Avg Reward: {round(np.mean(reward_stats), 2)}')    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO forward reinforcement learning')
    # parser.add_argument('--env', type=str, default='ma_gym:DecHuRoSorting-v0', help='Provide the env')
    parser.add_argument('--env', type=str, default='FeedingSawyerHuman-v1', help='Provide the env')
    # parser.add_argument('--training_epochs', type=int, default=20, help='Total training epochs')
    args = parser.parse_args()

    env_id = args.env
    load_env_id = save_env_id = env_id.replace(":", "_")
    ppo = Record(env_id)

    ppo.load(path=f'{PACKAGE_PATH}/models/{load_env_id}')
    ppo.record(num_steps=10000, save_env_id=save_env_id)