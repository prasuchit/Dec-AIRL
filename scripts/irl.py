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
from datetime import datetime

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
    # p.add_argument('--env_id', type=str, default='FeedingSawyerHuman-v1')
    p.add_argument('--env_id', type=str, default='ma_gym:DecHuRoSorting-v0')
    p.add_argument('--cuda', action='store_true', default=False)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--failure_traj', action='store_true', default=False)
    p.add_argument('--load_existing', action='store_true', default=False)
    p.add_argument('--model-path', type=str, default=f'{PACKAGE_PATH}/models_airl/')
    args = p.parse_args()

    env_id = args.env_id
    load_env_id = env_id.replace(":", "_")
    device = 'cuda:0' if args.cuda else 'cpu'

    # args.model_path = args.model_path + env_id + '/2022-10-10_18-23/step_249856_reward_75'    # NOTE: Modify this if you're loading an existing model!

    buffers_exp = torch.load(f'{PACKAGE_PATH}/buffers/{load_env_id}/trajectory.pt')

    save_dir = f'{PACKAGE_PATH}/models_airl/{env_id}/' + datetime.now().strftime('%Y-%m-%d_%H-%M')

    airl = AIRL(env_id=env_id, buffers_exp=buffers_exp, device=device, seed=args.seed, eval_interval=args.eval_interval, 
                path = save_dir, units_disc_r = (128, 128), units_disc_v = (128, 128))

    if args.load_existing:
        airl.model_loader(args.model_path, only_disc = False, only_gen = False)

    airl.train(total_timesteps=args.num_steps)