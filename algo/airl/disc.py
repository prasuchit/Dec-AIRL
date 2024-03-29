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
import os
import sys
import shutup; shutup.please()


def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class GAILDiscrim(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(100, 100),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))

    def calculate_reward(self, states, actions):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(states, actions) + 1e-20)


class AIRLDiscrim(nn.Module):

    def __init__(self, state_shape, gamma,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        self.g = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.gamma = gamma

    def f(self, states, dones, next_states):
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, dones, next_states) - log_pis

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones[None, :], log_pis, next_states)
            return -F.logsigmoid(-logits + 1e-20)


class AIRLDiscrimMultiAgent(nn.Module):

    def __init__(self, obs_space, gamma, action_space, agents,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        if type(obs_space).__name__ == 'MultiAgentObservationSpace':
            state_shape = 0
            action_shape = 0
            for i in range(len(obs_space)):
                state_shape += obs_space[i].shape[0]
                assert type(action_space[i]).__name__ == 'Discrete'
                action_shape += action_space[i].n
        else:
            state_shape = obs_space.shape[0]
            action_shape = action_space.shape[0]
        
        self.g = build_mlp(
            input_dim=state_shape + action_shape,
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(
            input_dim=state_shape,
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.gamma = gamma
        self.agents = agents

    def f(self, states, dones, next_states, actions):
        device = states.device
        states_actions = torch.cat((states, actions), dim=1).to(device)
        rs = self.g(states_actions).to(device)
        vs = self.h(states).to(device)
        next_vs = self.h(next_states).to(device)
        gamma = torch.as_tensor(self.gamma).to(device)
        dones = dones.to(device)
        return rs + gamma * (1 - dones) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states, actions):
        # Discriminator's output is sigmoid(f - log_pi).
        # global_states = states.reshape(states.shape[0],-1)
        # global_next_states = next_states.reshape(states.shape[0],-1)
        global_states = torch.cat([states[i] for i in self.agents], dim=1)
        global_next_states = torch.cat([next_states[i] for i in self.agents], dim=1)
        
        return self.f(global_states, dones[self.agents[0]][:, None], global_next_states, actions) - log_pis

    def calculate_reward(self, states, dones, log_pis, next_states, actions):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states, actions)
            return -F.logsigmoid(-logits + 1e-20)
