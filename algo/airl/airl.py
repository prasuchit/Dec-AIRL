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

from datetime import datetime
from stable_baselines3.common.utils import obs_as_tensor
from algo.ppo.ActorCritic import *
from algo.ppo.ppo import *
from algo.airl.disc import AIRLDiscrimMultiAgent

''' Adversarial IRL class that extends the original paper Fu et al. 2017(https://arxiv.org/pdf/1710.11248.pdf) to work with multiple agents'''

class AIRL(object):
    def __init__(self, env_id, buffers_exp, seed, eval_interval=500,
                 gamma=0.95, n_steps=2048, device='cpu',
                 batch_size=128, lr_actor=3e-4, lr_disc=3e-4,
                 units_disc_r=(64, 64), units_disc_v=(64, 64),
                 epoch_actor=10, epoch_disc=10, clip_eps=0.2, gae_lambda=0.97,
                 ent_coef=0.0, max_grad_norm=0.5, path = os.getcwd()):

        self.env = gym.make(env_id)
        self.env.seed(seed)
        self.seed = seed
        self.n_agents = self.env.n_agents
        self.device = device
        self.path = path

        # if self.env.observation_space.__class__.__name__ == 'Discrete':
        #     self.state_shape = (self.env.observation_space.n,)
        # elif self.env.observation_space.__class__.__name__ == 'Box':
        #     self.state_shape = self.env.observation_space.shape
        # else:
        #     raise ValueError('Cannot recognize env observation space ')

        assert self.env.observation_space.__class__.__name__ == 'MultiAgentObservationSpace', f'Unsupported observation space: {self.env.observation_space.__class__.__name__}'

        # Discriminator.
        self.disc = AIRLDiscrimMultiAgent(
            obs_space=self.env.observation_space,
            gamma=gamma,
            action_space=self.env.action_space,
            n_agents=self.n_agents,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        self.actors = [PPO_Dec(ActorCriticPolicy_Dec, self.env, agent_id=agent_id, verbose=1, custom_rollout=True, device=self.device, seed=self.seed) for agent_id in range(self.n_agents)]

        # print("Load existing", load_existing)

        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc
        self.eval_env = gym.make(env_id)
        self.eval_env.seed(self.seed)
        self.test_env = gym.make(env_id)
        self.test_env.seed(self.seed)
        self.n_steps = n_steps
        self.learning_steps_disc = 0
        self.device = device
        self.eval_interval = eval_interval

        self.buffers_exp = buffers_exp

        self.global_observation_shape = self.env.observation_space[0].shape[0] * self.n_agents
        self.local_observation_shape = self.env.observation_space[0].shape[0]
        self.action_shape = self.env.action_space[0].n
        self.best_reward = -1000
        # self.load_models(load_existing, trainpath)

        self.buffers_policy = {
            'state': torch.zeros(size=(n_steps, self.n_agents, self.local_observation_shape), device=device),
            'action': torch.zeros(size=(n_steps, self.n_agents), device=device),
            'next_state': torch.zeros(size=(n_steps, self.n_agents, self.local_observation_shape), device=device),
            'reward': torch.zeros(n_steps, device=device),
            'done': torch.zeros(size=(n_steps, 1), device=device),
            'value': torch.zeros(size=(n_steps, self.n_agents), device=device),
            'log_prob': torch.zeros(size=(n_steps, self.n_agents), device=device),
            'info': [[{}]] * n_steps,
            'p': 0,
            'record': 0
        }

        self.buffer_size = n_steps

    def update(self, epoch_ratio):
        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            states_policy, actions_policy, next_states_policy, _, dones_policy, _, log_probs_policy = self.buffer_sample(self.buffers_policy, expert=False)

            states_exp, actions_exp, next_states_exp, _, dones_exp = self.buffer_sample(self.buffers_exp, expert=True)
            
            log_probs_exp = []
            actions_policy_onehot = []
            actions_exp_onehot = []

            with torch.no_grad():
                for agent_id in range(self.n_agents):
                    _, log_prob_exp, _ = self.actors[agent_id].policy.evaluate_actions(states_exp, actions_exp[:, agent_id])
                    action_policy_onehot = torch.nn.functional.one_hot(actions_policy[:, agent_id].long(), num_classes=self.action_shape).float()
                    action_exp_onehot = torch.nn.functional.one_hot(actions_exp[:, agent_id].long(), num_classes=self.action_shape).float()

                    log_probs_exp.append(log_prob_exp)
                    actions_policy_onehot.append(action_policy_onehot)
                    actions_exp_onehot.append(action_exp_onehot)

            log_probs_exp = torch.stack(log_probs_exp, dim=1).float()
            global_actions_policy = torch.cat(actions_policy_onehot, dim=1)
            global_actions_exp = torch.cat(actions_exp_onehot, dim=1)

            log_probs_policy = log_probs_policy.sum(dim=1)
            log_probs_exp = log_probs_exp.sum(dim=1)

            self.update_disc(
                states_policy, global_actions_policy, dones_policy, log_probs_policy, next_states_policy, 
                states_exp, global_actions_exp, dones_exp, log_probs_exp, next_states_exp, epoch_ratio
            )

        states, actions, next_states, _, dones, values, log_probs, infos = self.buffer_get(self.buffers_policy)
        
        actions_onehot = []
        for agent_id in range(self.n_agents):
            action_onehot = torch.nn.functional.one_hot(actions[:, agent_id].long(), num_classes=self.action_shape).float()

            actions_onehot.append(action_onehot)
        global_actions = torch.cat(actions_onehot, dim=1)
        actions_onehot = torch.stack(actions_onehot, dim=1)

        global_log_probs = log_probs.sum(dim=1)[:, None]

        # Calculate rewards.
        rewards = self.disc.calculate_reward(
            states, dones, global_log_probs, next_states, global_actions).squeeze()
        
        # rewards = normalize(rewards)

        for agent_id in range(self.n_agents):
            self.actors[agent_id].learn(total_timesteps=1000000, states_rollout=states.cpu().numpy(), next_states_rollout=next_states.cpu().numpy(),
                         actions_rollout=actions[:, agent_id].cpu().numpy(), rewards_rollout=rewards.cpu().numpy(), dones_rollout=dones.squeeze().cpu().numpy(),
                         values_rollout=values[:, agent_id], log_probs_rollout=log_probs[:, agent_id], infos_rollout=infos)

    def update_disc(self, states, actions, dones, log_probs, next_states,
                    states_exp, actions_exp, dones_exp, log_probs_exp,
                    next_states_exp, epoch_ratio):

        logits_pi = self.disc(states, dones, log_probs[:, None], next_states, actions)
        logits_exp = self.disc(
            states_exp, dones_exp, log_probs_exp[:, None], next_states_exp, actions_exp)

        loss_pi = -F.logsigmoid(-logits_pi)
        loss_exp = -F.logsigmoid(logits_exp)
        loss_disc = (loss_pi + loss_exp).mean()

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

    def buffer_add(self, buffer, state, action, next_state, reward, done, value, log_prob, info):
        p = buffer['p']
        buffer['state'][p] = torch.tensor(state).clone().float()
        buffer['action'][p] = torch.tensor(action).clone()
        buffer['next_state'][p] = torch.tensor(next_state).clone().float()
        buffer['reward'][p] = reward
        buffer['done'][p] = torch.tensor([int(done)]).float()
        buffer['value'][p] = torch.tensor(value).float()
        buffer['log_prob'][p] = torch.tensor(log_prob).float()
        buffer['info'][p] = [info]
        buffer['p'] += 1
        buffer['p'] %= self.buffer_size
        buffer['record'] += 1

    def buffer_sample(self, buffer, expert=False):
        if not expert:
            current_buffer_size = min(buffer['record'], self.buffer_size)
            idx = torch.randperm(current_buffer_size)[:self.batch_size]
            return buffer['state'][idx], buffer['action'][idx], buffer['next_state'][idx], buffer['reward'][idx], \
                   buffer['done'][idx], buffer['value'][idx], buffer['log_prob'][idx]
        else:
            current_buffer_size = len(buffer['state'])
            idx = torch.randperm(current_buffer_size)[:self.batch_size]
            return buffer['state'][idx], buffer['action'][idx], buffer['next_state'][idx], buffer['reward'][idx], \
                   buffer['done'][idx]

    def buffer_get(self, buffer):
        current_buffer_size = min(buffer['record'], self.buffer_size)
        return buffer['state'][:current_buffer_size], buffer['action'][:current_buffer_size], buffer['next_state'][:current_buffer_size], buffer['reward'][:current_buffer_size], \
               buffer['done'][:current_buffer_size], buffer['value'][:current_buffer_size], buffer['log_prob'][:current_buffer_size], buffer['info'][:current_buffer_size]

    def model_loader(self, path, only_disc = False, only_gen = False):
        if not only_disc:
            for i in self.agents:
                self.actors[i].set_parameters(f'{path}/{i}.zip',  device=self.device)
        if not only_gen:
            self.disc.load_state_dict(torch.load(f'{path}/disc.pt'))

    def train(self, total_timesteps=100000):
        obs = self.env.reset()

        for airl_step in range(1, total_timesteps):
            with th.no_grad():
                actions, values, log_probs = [], [], []
                for agent_id in range(self.n_agents):
                    action, value, log_prob = self.actors[agent_id].policy.forward(obs_as_tensor(obs, device=self.device).float())
                    actions.append(action.item())
                    values.append(value)
                    log_probs.append(log_prob)

            new_obs, rewards, dones, infos = self.env.step(actions)
            rewards = sum(rewards)
            dones = all(dones)

            self.buffer_add(self.buffers_policy, obs, actions, new_obs, rewards, dones, values, log_probs, infos)

            if dones:
                obs = self.env.reset()
            else:
                obs = new_obs

            if airl_step % self.n_steps == 0:
                # assert self.buffer_p == 0
                self.update(airl_step / total_timesteps)

            if airl_step % self.eval_interval == 0:
                print(f'Timesteps: {airl_step} | ', end='')
                eval_reward = self.evaluate(eval_epochs=10)
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self.save(airl_step, self.best_reward, path = self.path)

    def evaluate(self, eval_epochs=10):
        reward_stats = []
        length_stats = []

        reward = 0
        length = 0

        for epoch in range(eval_epochs):
            obs = self.eval_env.reset()
            dones = False

            while not dones:
                with th.no_grad():
                    actions, values, log_probs = [], [], []
                    for agent_id in range(self.n_agents):
                        action, value, log_prob = self.actors[agent_id].policy.forward(obs_as_tensor(obs, device=self.device), deterministic=True)
                        actions.append(action.item())
                        values.append(value)
                        log_probs.append(log_prob)

                new_obs, rewards, dones, infos = self.eval_env.step(actions)
                rewards = sum(rewards)
                dones = all(dones)

                obs = new_obs
                
                reward += rewards
                length += 1

            reward_stats.append(reward)
            length_stats.append(length)

            reward = 0
            length = 0
        
        eval_rewards = np.mean(reward_stats)
        print(f'Avg Length: {round(np.mean(length_stats), 2)} | Avg Reward: {round(eval_rewards, 2)}')
        return eval_rewards

    def save(self,step, reward, path=os.getcwd()):
        path = f"{path}/step_{step}_reward_{int(reward)}"
        print(f"\nSaving models at: {path}\n")
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.disc.state_dict(), f'{path}/disc.pt')
        for agent_id in range(self.n_agents):
            self.actors[agent_id].save(f'{path}/{agent_id}')
