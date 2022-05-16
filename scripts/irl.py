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

path = os.path.dirname (os.path.realpath (__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

sys.path.append(PACKAGE_PATH)
from algo.airl.airl import AIRL


''' This file trains a IRL agent(s) to learn from recorded expert trajectories for any ma-gym(https://github.com/prasuchit/ma-gym) environment '''

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # p.add_argument('--buffer', type=str, required=True)
    # p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--num_steps', type=int, default=10 ** 7)
    p.add_argument('--eval_interval', type=int, default=4096)
    p.add_argument('--env_id', type=str, default='ma_gym:Checkers-v0')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--failure_traj', action='store_true')
    p.add_argument('--load_existing', action='store_true')
    p.add_argument('--test', action='store_true')
    args = p.parse_args()

    env_id = args.env_id
    device = 'cuda:0' if args.cuda else 'cpu'

    buffers_exp = torch.load(f'{PACKAGE_PATH}/buffers/{env_id}/data.pt')

    airl = AIRL(env_id=env_id, buffers_exp=buffers_exp, device=device, seed=args.seed, eval_interval=args.eval_interval)
    airl.train()