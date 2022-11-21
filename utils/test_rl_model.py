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
from test_irl import AIRL_Test

''' This file can be used to test a trained RL agent(s) for any ma-gym(https://github.com/prasuchit/ma-gym) environment '''

class Test(Dec_Train):
    def __init__(self, env_id, device='cpu', seed=1024):
        super().__init__(env_id, device=device, seed=seed)
    
    def tester(self, num_steps=10000):

        # # obs = self.env.reset()

        # s_id_rob = self.env.vals2sid_interact([2, 2, 2, 0])
        # s_id_hum = self.env.vals2sid_interact([2, 2, 1, 0])

        # oloc_r, eefloc_r, pred_r, inter_r = self.env.sid2vals_interact(s_id_rob)
        # oloc_h, eefloc_h, pred_h, inter_h = self.env.sid2vals_interact(s_id_hum)
        
        # obs = self.env.get_global_onehot([[oloc_r, eefloc_r, pred_r, inter_r], [oloc_h, eefloc_h, pred_h, inter_h]])

        # self.env.set_prev_obsv(0, s_id_rob)
        # self.env.set_prev_obsv(1, s_id_hum)
        # self.env._step_count = 0
        # self.env.reward = self.env.step_cost

        # self.env._agent_dones = False
        # self.env.steps_beyond_done = None

        # with torch.no_grad():
        #     actions = []
        #     for agent_id in range(self.n_agents):
        #         action, _, _ = self.models[agent_id].policy.forward(obs_as_tensor(obs, device=self.device), deterministic=True)
        #         actions.append(action.item())
        # rob_act = self.env.get_action_meanings(actions[0])
        # hum_act = self.env.get_action_meanings(actions[1])
        # print(f"Robot state: Onion: {oloc_r}, Eef: {eefloc_r}, Pred: {pred_r}, Interaction: {bool(inter_r)};\nRobot action: {rob_act};")
        # print(f"Human state: Onion: {oloc_h}, Eef: {eefloc_h}, Pred: {pred_h}, Interaction: {bool(inter_h)};\nHuman action: {hum_act};")
        # new_obs, rewards, dones, infos = self.env.step(actions, verbose=0)
        # # rewards = sum(rewards)
        # dones = all(dones)
        # print(f"Reward: {rewards}")
        # if dones:
        #     obs = self.env.reset()
        # else:
        #     obs = new_obs
        # print(f'Collect Episodes: {len(length_stats)} | Avg Length: {round(np.mean(length_stats), 2)} | Avg Reward: {round(np.mean(reward_stats), 2)}')    

        print("Saving learned policies...")

        test_irl = AIRL_Test(env_id)
        
        load_env_id = env_id.replace(":", "_")

        load_dir = f'{PACKAGE_PATH}/models/{load_env_id}/'

        test_irl.save_discrete_policy(path=load_dir, nodisc=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO forward reinforcement learning')
    parser.add_argument('--env', type=str, default='ma_gym:DecHuRoSorting-v0', help='Provide the env')
    parser.add_argument('--training_epochs', type=int, default=20, help='Total training epochs')
    args = parser.parse_args()

    env_id = args.env
    load_env_id = env_id.replace(":", "_")
    ppo = Test(env_id)

    ppo.load(path=f'{PACKAGE_PATH}/models/{load_env_id}')
    ppo.tester(num_steps=10000)