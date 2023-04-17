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
from gym.spaces import Box, Discrete
import argparse
import sys
import os, importlib
import numpy as np

from datetime import datetime
# from stable_baselines3.common.utils import obs_as_tensor

# path = os.path.dirname (os.path.realpath (__file__))
# PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

# sys.path.append(PACKAGE_PATH)
from algo.ppo.ActorCritic import *
from algo.ppo.ppo import *
from algo.airl.disc import AIRLDiscrimMultiAgent
from algo.airl.buffer import Buffers_AIRL
from utils import get_assistive_gym_envs_list
# import shutup; shutup.please()

''' Adversarial IRL class that extends the original paper Fu et al. 2017(https://arxiv.org/pdf/1710.11248.pdf) to work with decentralized agents'''

class AIRL(object):
    def __init__(self, env_id, buffers_exp, seed, eval_interval=500,
                 gamma=0.95, n_steps=2048, device='cpu',
                 batch_size=128, lr_actor=3e-4, lr_disc=3e-4,
                 units_disc_r=(64, 64), units_disc_v=(64, 64),
                 epoch_actor=10, epoch_disc=10, clip_eps=0.2, gae_lambda=0.97,
                 ent_coef=0.0, max_grad_norm=0.5, path = os.getcwd()):

        self.seed = seed

        # if the env belongs to assistive gym
        if env_id in get_assistive_gym_envs_list():
            # if the env is cooperative
            if 'Human' in env_id:
                import importlib
                module = importlib.import_module('assistive_gym.envs')
                env_class = getattr(module, env_id.split('-')[0] + 'Env')
                self.env = env_class()
                self.eval_env = env_class()
            else:
                self.env = gym.make('assistive_gym:' + env_id)
                self.eval_env = gym.make('assistive_gym:' + env_id)
            self.assistive_gym = True
        else:
            # Training Env
            self.env = gym.make(env_id)
            self.env.seed(self.seed)

            # Testing Env
            self.eval_env = gym.make(env_id)
            self.assistive_gym = False

        self.eval_env.seed(self.seed)
        self.device = device
        # init agents
        if self.assistive_gym:
            self.agents = ['robot', 'human']
        else:
            self.agents = list(range(self.env.n_agents))

        self.actors = {agent_id: PPO_Dec(ActorCriticPolicy_Dec, self.env, agent_id=agent_id, verbose=1, custom_rollout=True, device=self.device, seed=self.seed) for agent_id in self.agents}
        self.path = path

        # Discriminator.
        self.disc = AIRLDiscrimMultiAgent(
            obs_space=self.env.observation_space,
            gamma=gamma,
            action_space=self.env.action_space,
            agents=self.agents,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

        self.n_steps = n_steps
        self.learning_steps_disc = 0
        self.eval_interval = eval_interval

        try:
            assert type(self.env.action_space) == Box and self.assistive_gym
            self.local_observation_shape = {
                agent_id: getattr(self.env,'observation_space_' + agent_id).shape[0]
                for agent_id in self.agents
            }
            self.local_action_shape = {
                agent_id: getattr(self.env,'action_space_' + agent_id).shape[0]
                for agent_id in self.agents
            }
            self.global_observation_shape = self.env.observation_space.shape[0]
            self.global_action_shape = self.env.action_space
            self.action_continuous = True
        except:
            assert type(self.env.action_space[0]) == Discrete and not self.assistive_gym
            self.local_observation_shape = {
                agent_id: self.env.observation_space[agent_id].shape[0]
                for agent_id in self.agents
            }
            self.local_action_shape = {
                agent_id: 1
                for agent_id in self.agents
            }
        

            global_observation_space_low = np.concatenate([self.env.observation_space[i].low for i in range(len(self.env.observation_space))])
            global_observation_space_high = np.concatenate([self.env.observation_space[i].high for i in range(len(self.env.observation_space))])
            self.global_observation_space = Box(low=global_observation_space_low, high=global_observation_space_high)
            self.global_observation_shape = self.global_observation_space.shape[0]
            self.global_action_shape = self.env.action_space[0].n
            self.action_continuous = False

        self.buffers_exp = buffers_exp
        self.buffers = Buffers_AIRL(buffer_size=self.n_steps, 
                                    batch_size=self.batch_size,
                                    agents=self.agents, 
                                    device=self.device,
                                    local_observation_shape=self.local_observation_shape,
                                    local_action_shape=self.local_action_shape)
        
        self.best_reward = - float('inf')
        # self.load_models(load_existing, trainpath)

    def update(self, epoch_ratio):
        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            states_policy, actions_policy, next_states_policy, _, dones_policy, _, log_probs_policy = self.buffers.sample(self.buffers.policy_buffer, expert=False)

            states_exp, actions_exp, next_states_exp, _, dones_exp = self.buffers.sample(self.buffers_exp, expert=True)
            
            log_probs_exp = {}
            if not self.action_continuous:
                actions_policy_onehot = []
                actions_exp_onehot = []

            with torch.no_grad():
                for agent_id in self.agents:
                    local_states_exp = torch.as_tensor(states_exp[agent_id]).to(self.device)
                    global_states_exp = torch.cat([states_exp[i] for i in self.agents], dim=1).to(self.device)
                    _, log_prob_exp, _ = self.actors[agent_id].policy.evaluate_actions(local_states_exp, global_states_exp, actions_exp[agent_id])
                    if not self.action_continuous:
                        action_policy_onehot = torch.nn.functional.one_hot(actions_policy[agent_id].squeeze().long(), num_classes=self.global_action_shape).float()
                        action_exp_onehot = torch.nn.functional.one_hot(actions_exp[agent_id].squeeze().long(), num_classes=self.global_action_shape).float()
                        actions_policy_onehot.append(action_policy_onehot)
                        actions_exp_onehot.append(action_exp_onehot)

                    log_probs_exp[agent_id] = log_prob_exp

            if self.action_continuous:
                global_actions_policy = torch.cat([actions_policy[i] for i in self.agents], dim=1).to(self.device)
                global_actions_exp = torch.cat([actions_exp[i] for i in self.agents], dim=1).to(self.device)
            else:
                global_actions_policy = torch.cat(actions_policy_onehot, dim=1)
                global_actions_exp = torch.cat(actions_exp_onehot, dim=1)

            log_probs_policy = sum([log_probs_policy[i] for i in self.agents]).to(self.device)
            log_probs_exp = sum([log_probs_exp[i] for i in self.agents]).to(self.device)

            self.update_disc(
                states_policy, global_actions_policy, dones_policy, log_probs_policy, next_states_policy, 
                states_exp, global_actions_exp, dones_exp, log_probs_exp, next_states_exp, epoch_ratio
            )

        states, actions, next_states, _, dones, values, log_probs, infos = self.buffers.get(self.buffers.policy_buffer)  
        if self.action_continuous:      
            global_actions = torch.cat([actions[i] for i in self.agents], dim=1).to(self.device)
        else:
            actions_onehot = []
            for agent_id in self.agents:
                action_onehot = torch.nn.functional.one_hot(actions[agent_id].squeeze().long(), num_classes=self.global_action_shape).float()
                actions_onehot.append(action_onehot)
            global_actions = torch.cat(actions_onehot, dim=1)
        global_log_probs = sum([log_probs[i] for i in self.agents])

        # Calculate rewards.
        rewards = self.disc.calculate_reward(
            states, dones, global_log_probs[:, None], next_states, global_actions).squeeze().squeeze()
                
        # rewards = normalize(rewards)

        for agent_id in self.agents:
            states_rollout = {
                i: states[i].cpu().numpy() for i in self.agents
            }
            next_states_rollout = {
                i: next_states[i].cpu().numpy() for i in self.agents
            }
            self.actors[agent_id].learn(total_timesteps=1000000, states_rollout=states_rollout, next_states_rollout=next_states_rollout,
                         actions_rollout=actions[agent_id].cpu().numpy(), rewards_rollout=rewards.cpu().numpy(), dones_rollout=dones[agent_id].cpu().numpy(),
                         values_rollout=values[agent_id], log_probs_rollout=log_probs[agent_id], infos_rollout=infos[agent_id])

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
        
        # print("AIRL disc loss: ", loss_disc.item())

    def model_loader(self, path, only_disc = False, only_gen = False):
        if not only_disc:
            for i in self.agents:
                self.actors[i].set_parameters(f'{path}/{i}.zip', device=self.device)
        if not only_gen:
            self.disc.load_state_dict(torch.load(f'{path}/disc.pt'))

    def train(self, total_timesteps=100000):
        self.env.seed(int(time.time()))
        obs = self.env.reset()

        for airl_step in range(1, total_timesteps):
            with th.no_grad():
                actions, values, log_probs = {agent_id: None for agent_id in self.agents}, {agent_id: None for agent_id in self.agents}, {agent_id: None for agent_id in self.agents}
                for agent_id in self.agents:
                    local_obs = th.as_tensor(obs[agent_id])
                    global_obs = th.as_tensor(np.concatenate([obs[i] for i in self.agents]))
                    action, value, log_prob = self.actors[agent_id].policy.forward(local_obs.to(self.device), global_obs.to(self.device))
                    actions[agent_id] = action.squeeze().cpu().numpy()
                    values[agent_id] = value
                    log_probs[agent_id] = log_prob
            
            if self.assistive_gym:
                new_obs, rewards, dones, infos = self.env.step(actions)
            else:
                new_obs, rewards, dones, infos = self.env.step([actions[action] for action in actions])

            # new_obs, rewards, dones, infos = self.env.step(actions)

            self.buffers.add(self.buffers.policy_buffer, obs, actions, new_obs, rewards, dones, values, log_probs, infos)

            rewards = sum([rewards[i] for i in self.agents]) / 2
            dones = any([dones[i] for i in self.agents])

            if dones:
                self.env.seed(int(time.time()))
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
            self.eval_env.seed(int(time.time()))
            obs = self.eval_env.reset()
            dones = False

            while not dones:
                with th.no_grad():
                    actions, values, log_probs = {}, {}, {}
                    for agent_id in self.agents:
                        local_obs = th.as_tensor(obs[agent_id])
                        global_obs = th.as_tensor(np.concatenate([obs[i] for i in self.agents]))
                        action, value, log_prob = self.actors[agent_id].policy.forward(local_obs.to(self.device), global_obs.to(self.device))
                        actions[agent_id] = action.squeeze().cpu().numpy()
                        values[agent_id] = value
                        log_probs[agent_id] = log_prob

                if self.assistive_gym:
                    new_obs, rewards, dones, infos = self.eval_env.step(actions)
                else:
                    new_obs, rewards, dones, infos = self.eval_env.step([actions[action] for action in actions])

                # new_obs, rewards, dones, infos = self.eval_env.step(actions)
                rewards = sum([rewards[i] for i in self.agents]) / 2
                dones = any([dones[i] for i in self.agents])

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
        for agent_id in self.agents:
            self.actors[agent_id].save(f'{path}/{agent_id}')
