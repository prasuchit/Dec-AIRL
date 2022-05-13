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
from algo.ppo.ppo import Dec_Train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO forward reinforcement learning')
    parser.add_argument('--env', type=str, default='ma_gym:Checkers-v0', help='Provide the env')
    parser.add_argument('--training_epochs', type=int, default=100, help='Total training epochs')
    args = parser.parse_args()

    env_id = args.env
    ppo = Dec_Train(env_id)
    ppo.train(epochs=args.training_epochs)

    ppo.save(path=f'{PACKAGE_PATH}/models/{env_id}')