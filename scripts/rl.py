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

import numpy as np
import torch
import gym
path = os.path.dirname (os.path.realpath (__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

sys.path.append(PACKAGE_PATH)
from algo.ppo.ppo import Dec_Train

''' This file trains an RL agent(s) using Decentralized Proximal Policy Optimization(Dec-PPO) for 
    any ma-gym(https://github.com/prasuchit/ma-gym) or assistive-gym(https://github.com/prasuchit/assistive-gym) environment '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO forward reinforcement learning')
    parser.add_argument('--env', type=str, default='ma_gym:DecHuRoSorting-v0', help='Provide the env')
    # parser.add_argument('--env', type=str, default='FeedingSawyerHuman-v1', help='Provide the env')
    parser.add_argument('--training_epochs', type=int, default=500, help='Total training epochs')
    parser.add_argument('--seed', type=int, default=321, help='Seed for random number gen')
    parser.add_argument('--load_existing', action='store_true', default=False)
    parser.add_argument('--model-path', type=str, default=f'{PACKAGE_PATH}/models/')
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    env_id = args.env
    save_env_id = env_id.replace(":", "_")
    ppo = Dec_Train(env_id, seed = args.seed)

    if not args.test:
        if args.load_existing:
            ppo.load(args.model_path + f'{save_env_id}')
        ppo.train(epochs=args.training_epochs, path=f'{PACKAGE_PATH}/models/{save_env_id}')
        ppo.save(path=f'{PACKAGE_PATH}/models/{save_env_id}')
    else:
        ppo.test(load_model=True, load_path=f'{PACKAGE_PATH}/models/{save_env_id}',env_id=env_id)