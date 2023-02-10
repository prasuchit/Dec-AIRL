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
from tqdm import tqdm
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

''' Adversarial IRL class that extends the original paper Fu et al. 2017(https://arxiv.org/pdf/1710.11248.pdf) to work with decentralized agents'''

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

        self.test_env = self.eval_env   # Initiating a test env

    def test_disc(self, path):
        raise NotImplementedError

    def model_loader(self, path, nodisc=False):
        for i in self.agents:
            self.actors[i].set_parameters(f'{path}/{i}.zip',  device=self.device)
        if nodisc:
            pass
        else: self.disc.load_state_dict(torch.load(f'{path}/disc.pt'))

    def test(self, path, test_epochs = 1):
        ep_rewards = []
        ep_lengths = []
        verbose = True
        self.model_loader(path)
        test_actions, test_values, test_log_probs = {agent_id: None for agent_id in self.agents}, {agent_id: None for agent_id in self.agents}, {agent_id: None for agent_id in self.agents}

        self.test_env.render()

        for test_epoch in range(test_epochs):
            test_state = self.test_env.reset()
            test_dones = False
            ep_length = 0
            ep_reward = 0
            while not test_dones:
                # for i in self.agents:
                #     test_action_actor[i], _, test_action_actor_log_prob[i] = self.actors[i].policy.forward(obs_as_tensor(test_state, device=self.device).float(), deterministic=True)
                for i in self.agents:
                    local_obs = torch.as_tensor(test_state[i])
                    global_obs = torch.as_tensor(np.concatenate([test_state[i] for i in self.agents]))
                    test_action, test_value, test_log_prob = self.actors[i].policy.forward(local_obs, global_obs)
                    test_actions[i] = test_action.detach().squeeze().cpu().numpy()
                    test_values[i] = test_value
                    test_log_probs[i] = test_log_prob

                if self.assistive_gym:
                    test_next_state, test_rewards, test_dones, test_infos = self.test_env.step(test_actions)
                else:
                    test_next_state, test_rewards, test_dones, test_infos = self.test_env.step([test_actions[action] for action in test_actions])

                # for i in self.agents:            
                #     action_actor_onehot[i] = torch.nn.functional.one_hot(test_action_actor[i].long(), num_classes=6).float()

                # global_test_actions = torch.cat(([act for act in action_actor_onehot.values()]), dim=1)
                
                # log_probs = sum(test_action_actor_log_prob.values())

                # disc_reward = self.disc.calculate_reward(test_state, torch.tensor([int(test_done)])[None, :].float(), log_probs[:, None], torch.tensor(test_next_state)[None, :].float(), global_test_actions).squeeze()

                # print(f'Original reward: {test_reward} | Disc reward: {round(disc_reward.item(), 3)}')

                if not self.assistive_gym:
                    test_rewards = sum(test_rewards) / 2
                    test_dones = all(test_dones)
                else:
                    test_rewards = (test_rewards['robot'] + test_rewards['human']) / 2
                    test_dones = test_dones['__all__']

                print(f"Reward: {test_rewards}, Done: {test_dones}")
                # if test_dones:
                #     test_state = self.test_env.reset()
                # else:
                test_state = test_next_state
                ep_length += 1
                ep_reward += test_rewards

            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)

        print(f'mean length: {round(np.mean(ep_lengths), 1)} | mean reward: {round(np.mean(ep_rewards), 1)}')
        return np.mean(ep_rewards)
    
    def save_discrete_policy(self, path, nodisc = False):
        global PACKAGE_PATH
        self.model_loader(path, nodisc=nodisc)
        action_actor = {}
        action_actor_log_prob = {}
        if self.env.name == 'DecHuRoSorting':
            policy_r = np.zeros((self.env.nSAgent, 1))
            policy_h = np.zeros((self.env.nSAgent, 1))
            # global_policy = np.zeros((self.env.nSGlobal, 1))
            for S in tqdm(range(self.env.nSGlobal)):
                oloc_r, eefloc_r, pred_r, interact_r, oloc_h, eefloc_h, pred_h, interact_h = self.env.sGlobal2vals(S)
                global_onehot_s = self.env.get_global_onehot([[oloc_r, eefloc_r, pred_r, interact_r], [oloc_h, eefloc_h, pred_h, interact_h]])
                for i in self.agents:
                    local_obs = torch.as_tensor(global_onehot_s[i])
                    global_obs = torch.as_tensor(np.concatenate([global_onehot_s[i] for i in self.agents]))
                    action_actor[i], _, action_actor_log_prob[i] = self.actors[i].policy.forward(local_obs, global_obs)
                A = self.env.vals2aGlobal(action_actor[0], action_actor[1])
                s_r = self.env.vals2sid_interact([oloc_r, eefloc_r, pred_r, interact_r])
                s_h = self.env.vals2sid_interact([oloc_h, eefloc_h, pred_h, interact_h])
                # if interact_r == interact_h == 1:
                #     oloc_r_, eefloc_r_, pred_r_, inter_r_ = self.env.get_state_meanings(oloc_r, eefloc_r, pred_r, interact_r)
                #     oloc_h_, eefloc_h_, pred_h_, inter_h_ = self.env.get_state_meanings(oloc_h, eefloc_h, pred_h, interact_h)
                #     a_r_ = self.env.get_action_meanings(action_actor[0].item())
                #     a_h_ = self.env.get_action_meanings(action_actor[1].item())
                #     print(f"Robot state: Onion: {oloc_r_}, Eef: {eefloc_r_}, Pred: {pred_r_}, Interaction: {inter_r_};\nRobot action: {a_r_};")
                #     print(f"Human state: Onion: {oloc_h_}, Eef: {eefloc_h_}, Pred: {pred_h_}, Interaction: {inter_h_};\nHuman action: {a_h_};")
                policy_r[s_r] = action_actor[0].item()
                policy_h[s_h] = action_actor[1].item()
                # global_policy[S] = A
            policy_path = PACKAGE_PATH + '/saved_policies/'
            # np.savetxt(policy_path+"learned_policy_global.csv", global_policy)
            np.savetxt(policy_path+"learned_policy_rob.csv", policy_r)
            np.savetxt(policy_path+"learned_policy_hum.csv", policy_h)




if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # p.add_argument('--buffer', type=str, required=True)
    # p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--num_steps', type=int, default=10 ** 7)
    p.add_argument('--eval_interval', type=int, default=4096)
    # p.add_argument('--env_id', type=str, default='ma_gym:DecHuRoSorting-v0')
    p.add_argument('--env_id', type=str, default='FeedingSawyerHuman-v1')
    p.add_argument('--cuda', action='store_true', default=False)
    p.add_argument('--seed', type=int, default=321)
    p.add_argument('--failure_traj', action='store_true', default=False)
    p.add_argument('--load_existing', action='store_true', default=False)
    p.add_argument('--model_path', type=str, default='2023-02-09_18-55/step_2101248_reward_134')
    args = p.parse_args()

    env_id = args.env_id

    load_env_id = env_id.replace(':', '_')

    device = 'cuda:0' if args.cuda else 'cpu'

    load_dir = f'{PACKAGE_PATH}/models_airl/{load_env_id}/' + args.model_path

    airl = AIRL_Test(env_id=env_id, seed=args.seed, device=device, units_disc_r = (128, 128), units_disc_v = (128, 128))
    # airl.save_discrete_policy(path = load_dir)
    airl.test(path=load_dir)