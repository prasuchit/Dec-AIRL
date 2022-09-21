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

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

import gym
import argparse
import sys
import os
import numpy as np

path = os.path.dirname (os.path.realpath (__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

sys.path.append(PACKAGE_PATH)
from algo.airl.airl import AIRL
from algo.ppo.ppo import obs_as_tensor

''' Adversarial IRL class that extends the original paper Fu et al. 2017(https://arxiv.org/pdf/1710.11248.pdf) to work with multiple agents'''

class AIRL_Test(AIRL):
    def __init__(self, env_id, buffers_exp=None, seed=None, eval_interval=500,
                 gamma=0.95, n_steps=2048, device='cpu',
                 batch_size=128, lr_actor=3e-4, lr_disc=3e-4,
                 units_disc_r=(64, 64), units_disc_v=(64, 64),
                 epoch_actor=10, epoch_disc=10, clip_eps=0.2, gae_lambda=0.97,
                 ent_coef=0.0, max_grad_norm=0.5, path = os.getcwd()):

        AIRL.__init__(self, env_id, buffers_exp, seed, eval_interval=eval_interval,
                 gamma=gamma, n_steps=n_steps, device=device,
                 batch_size=batch_size, lr_actor=lr_actor, lr_disc=lr_disc,
                 units_disc_r=units_disc_r, units_disc_v=units_disc_v,
                 epoch_actor=epoch_actor, epoch_disc=epoch_disc, clip_eps=clip_eps, gae_lambda=gae_lambda,
                 ent_coef=ent_coef, max_grad_norm=max_grad_norm, path = path) 

    def test_disc(self, path):
        raise NotImplementedError

    def model_loader(self, path):
        for i in range(self.n_agents):
            self.actors[i].set_parameters(f'{path}/{i}.zip',  device=self.device)
        self.disc.load_state_dict(torch.load(f'{path}/disc.pt'))

    def test(self, path, test_epochs = 1):
        ep_rewards = []
        ep_lengths = []
        verbose = True
        self.model_loader(path)
        test_action_actor = {}
        test_action_actor_log_prob = {}
        action_actor_onehot = {}

        for test_epoch in range(test_epochs):
            test_state = self.test_env.reset()
            test_done = [False, False]
            ep_length = 0
            ep_reward = 0
            while not all(test_done):
                for i in range(self.n_agents):
                    test_action_actor[i], _, test_action_actor_log_prob[i] = self.actors[i].policy.forward(obs_as_tensor(test_state, device=self.device).float(), deterministic=True)

                test_next_state, test_reward, test_done, test_info = self.test_env.step([act for act in test_action_actor.values()], verbose=verbose)

                # for i in range(self.n_agents):            
                #     action_actor_onehot[i] = torch.nn.functional.one_hot(test_action_actor[i].long(), num_classes=6).float()

                # global_test_actions = torch.cat(([act for act in action_actor_onehot.values()]), dim=1)
                
                # log_probs = sum(test_action_actor_log_prob.values())

                # disc_reward = self.disc.calculate_reward(test_state, torch.tensor([int(test_done)])[None, :].float(), log_probs[:, None], torch.tensor(test_next_state)[None, :].float(), global_test_actions).squeeze()

                # print(f'Original reward: {test_reward} | Disc reward: {round(disc_reward.item(), 3)}')

                print(f"Reward: {test_reward}, Done: {test_done}")

                ep_reward += sum(test_reward)
                ep_length += 1
                test_state = test_next_state

            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)

        print(f'mean length: {round(np.mean(ep_lengths), 1)} | mean reward: {round(np.mean(ep_rewards), 1)}')
        return np.mean(ep_rewards)
    
    def save_discrete_policy(self, path):
        global PACKAGE_PATH
        self.model_loader(path)
        action_actor = {}
        action_actor_log_prob = {}
        if self.env.name == 'DecHuRoSorting':
            policy = np.zeros((self.env.nSGlobal, 1))
            for S in range(self.env.nSGlobal):
                oloc_r, eefloc_r, pred_r, interact_r, oloc_h, eefloc_h, pred_h, interact_h = self.env.sGlobal2vals(S)
                global_onehot_s = self.env.get_global_onehot([[oloc_r, eefloc_r, pred_r], [oloc_h, eefloc_h, pred_h]])
                global_onehot_s = self.env.check_interaction(global_onehot_s)
                for i in range(self.n_agents):
                    action_actor[i], _, action_actor_log_prob[i] = self.actors[i].policy.forward(obs_as_tensor(global_onehot_s, device=self.device).float(), deterministic=True)
                A = self.env.vals2aGlobal(action_actor[0], action_actor[1])
                policy[S] = A
            policy_path = PACKAGE_PATH + '/saved_policies/'
            np.savetxt(policy_path+"learned_policy.csv", policy)




if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # p.add_argument('--buffer', type=str, required=True)
    # p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--num_steps', type=int, default=10 ** 7)
    p.add_argument('--eval_interval', type=int, default=4096)
    p.add_argument('--env_id', type=str, default='ma_gym:DecHuRoSorting-v0')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--failure_traj', action='store_true')
    p.add_argument('--load_existing', action='store_true')
    p.add_argument('--test', action='store_true')
    p.add_argument('--model_path', type=str, default='2022-06-07_17-20/step_65536_reward_151')
    args = p.parse_args()

    env_id = args.env_id
    device = 'cuda:0' if args.cuda else 'cpu'

    load_dir = f'{PACKAGE_PATH}/models_airl/' + args.model_path

    airl = AIRL_Test(env_id=env_id, device=device, seed=args.seed)
    airl.save_discrete_policy(path = load_dir)
    # airl.test(path=load_dir)