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

from tqdm import tqdm
import numpy as np
import torch
import torch as th

import sys
import os
# sys.path.append(os.getcwd() + f'/airl-ppo/')
from buffer import Buffer

def get_assistive_gym_envs_list():
    return {
    "ScratchItchPR2-v1", "ScratchItchJaco-v1", "ScratchItchBaxter-v1", "ScratchItchSawyer-v1", "ScratchItchStretch-v1", "ScratchItchPanda-v1",
    "ScratchItchPR2Human-v1", "ScratchItchJacoHuman-v1", "ScratchItchBaxterHuman-v1", "ScratchItchSawyerHuman-v1", "ScratchItchStretchHuman-v1", "ScratchItchPandaHuman-v1",
    "BedBathingPR2-v1", "BedBathingJaco-v1", "BedBathingBaxter-v1", "BedBathingSawyer-v1", "BedBathingStretch-v1", "BedBathingPanda-v1",
    "BedBathingPR2Human-v1", "BedBathingJacoHuman-v1", "BedBathingBaxterHuman-v1", "BedBathingSawyerHuman-v1", "BedBathingStretchHuman-v1", "BedBathingPandaHuman-v1",
    "FeedingPR2-v1", "FeedingJaco-v1", "FeedingBaxter-v1", "FeedingSawyer-v1", "FeedingStretch-v1", "FeedingPanda-v1",
    "FeedingPR2Human-v1", "FeedingJacoHuman-v1", "FeedingBaxterHuman-v1", "FeedingSawyerHuman-v1", "FeedingStretchHuman-v1", "FeedingPandaHuman-v1",
    "DrinkingPR2-v1", "DrinkingJaco-v1", "DrinkingBaxter-v1", "DrinkingSawyer-v1", "DrinkingStretch-v1", "DrinkingPanda-v1",
    "DrinkingPR2Human-v1", "DrinkingJacoHuman-v1", "DrinkingBaxterHuman-v1", "DrinkingSawyerHuman-v1", "DrinkingStretchHuman-v1", "DrinkingPandaHuman-v1",
    "DressingPR2-v1", "DressingJaco-v1", "DressingBaxter-v1", "DressingSawyer-v1", "DressingStretch-v1", "DressingPanda-v1",
    "DressingPR2Human-v1", "DressingJacoHuman-v1", "DressingBaxterHuman-v1", "DressingSawyerHuman-v1", "DressingStretchHuman-v1", "DressingPandaHuman-v1",
    "ArmManipulationPR2-v1", "ArmManipulationJaco-v1", "ArmManipulationBaxter-v1", "ArmManipulationSawyer-v1", "ArmManipulationStretch-v1", "ArmManipulationPanda-v1",
    "ArmManipulationPR2Human-v1", "ArmManipulationJacoHuman-v1", "ArmManipulationBaxterHuman-v1", "ArmManipulationSawyerHuman-v1", "ArmManipulationStretchHuman-v1", "ArmManipulationPandaHuman-v1"
}

def obs_as_tensor(obs, device='cpu'):
    obs = th.as_tensor(obs).float().to(device)
    if len(obs.shape) == 1:
        return obs[None, :].float()
    else:
        return obs.float()


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def collect_demo(env, algo, buffer_size, device, std, p_rand, seed=0):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    t = 0
    episode_return = 0.0

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1

        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            action = algo.exploit(state)
            action = add_random_noise(action, std)

        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        buffer.append(state, action, reward, mask, next_state)
        episode_return += reward

        if done:
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0

        state = next_state

    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer
