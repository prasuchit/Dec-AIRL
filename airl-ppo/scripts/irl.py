import os
import sys
import argparse

import gym
import numpy as np
import torch

if 'airl-ppo' in os.getcwd():
    PACKAGE_PATH = os.getcwd()
else:
    PACKAGE_PATH = os.getcwd() + f'/airl-ppo/'

sys.path.append(PACKAGE_PATH)
from algo.airl.airl import AIRL


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
    # buffer_exp_success['label'] = torch.ones(len(buffer_exp_success['state']))

    # buffer_exp_failure = torch.load(os.getcwd()+f'/gail-airl-ppo/gail_airl_ppo/algo/buffers/ma_gym:hurosorting_failed.pt')
    # buffer_exp_failure['label'] = -torch.ones(len(buffer_exp_failure['robot_state']))

    airl = AIRL(env_id=env_id, buffers_exp=buffers_exp, device=device, seed=args.seed, eval_interval=args.eval_interval)
    airl.train()