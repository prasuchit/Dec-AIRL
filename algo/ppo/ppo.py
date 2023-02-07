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

''' PPO is imported from stable-baselines3 and adjusted to work with our algorithm. '''

from copy import copy
import gym
from gym import spaces
import os
import torch as th
from torch.nn import functional as F
from datetime import datetime
import numpy as np
import time
import warnings
from typing import Any, Dict, Optional, Type, Union

from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer

import random


import collections
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule

import sys
path = os.path.dirname (os.path.realpath (__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

sys.path.append(PACKAGE_PATH)
from utils import normalize
from algo.ppo.ActorCritic import OnPolicyAlgorithm_Dec, ActorCriticPolicy_Dec



SEED = 1
random.seed(SEED)
th.manual_seed(SEED)
np.random.seed(SEED)
th.use_deterministic_algorithms(True)
# robot_state_EOF = 12

assistive_gym_env_id = {
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
    obs = th.tensor(obs).float().to(device)
    if len(obs.shape) == 2:
        return obs[None, :]
    elif len(obs.shape) == 3:
        return obs

class PPO_Dec(OnPolicyAlgorithm_Dec):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)
    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            agent_id: int,
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            custom_rollout: bool = False
    ):

        super(PPO_Dec, self).__init__(
            policy=policy,
            env=env,
            agent_id=agent_id,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
            custom_rollout=custom_rollout
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        assert (
                batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        # if self.env is not None:
        #     # Check that `n_steps * n_envs > 1` to avoid NaN
        #     # when doing advantage normalization
        #     buffer_size = self.env.num_envs * self.n_steps
        #     assert (
        #             buffer_size > 1
        #     ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
        #     # Check that the rollout buffer size is a multiple of the mini-batch size
        #     untruncated_batches = buffer_size // batch_size
        #     if buffer_size % batch_size > 0:
        #         warnings.warn(
        #             f"You have specified a mini-batch size of {batch_size},"
        #             f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
        #             f" after every {untruncated_batches} untruncated mini-batches,"
        #             f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
        #             f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
        #             f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
        #         )
        self.env = env
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.agent_id = agent_id

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(PPO_Dec, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.local_observations, rollout_data.global_observations, actions)
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "PPO",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            **kwargs
    ) -> "PPO":

        return super(PPO_Dec, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
            **kwargs
        )


class Dec_Train():
    def __init__(self, env_id, device='cpu', seed=1024) -> None:
        self.env_id = env_id
        self.seed = seed

        # if the env belongs to assistive gym
        if env_id in assistive_gym_env_id:
            # if the env is cooperative
            if 'Human' in env_id:
                import importlib
                module = importlib.import_module('assistive_gym.envs')
                env_class = getattr(module, env_id.split('-')[0] + 'Env')
                self.env = env_class()
                self.test_env = env_class()
            else:
                self.env = gym.make('assistive_gym:' + env_id)
                self.test_env = gym.make('assistive_gym:' + env_id)
            self.assistive_gym = True
        else:
            # Training Env
            self.env = gym.make(env_id)
            self.env.seed(seed)

            # Testing Env
            self.env_test = gym.make(env_id)
            self.env_test.seed(seed)
            self.assistive_gym = False

        self.device = device
        # init agents
        if self.assistive_gym:
            self.agents = ['robot', 'human']
            self.models = {agent_id: PPO_Dec(ActorCriticPolicy_Dec, self.env, agent_id=agent_id, verbose=1, custom_rollout=True, device=self.device, seed=self.seed) for agent_id in self.agents}
        else:
            self.agents = list(range(self.env.n_agents))
            self.models = {agent_id: PPO_Dec(ActorCriticPolicy_Dec, self.env, agent_id=agent_id, verbose=1, custom_rollout=True, device=self.device, seed=self.seed) for agent_id in self.agents}

    def train(self, epochs=10, n_steps=2048, path = os.getcwd()):
        obs = self.env.reset()
        for epoch in range(epochs):
            # init custom collects
            states_rollout = {agent_id: [] for agent_id in self.agents}
            next_states_rollout = {agent_id: [] for agent_id in self.agents}
            actions_rollout = {agent_id: [] for agent_id in self.agents}
            rewards_rollout = {agent_id: [] for agent_id in self.agents}
            dones_rollout = {agent_id: [] for agent_id in self.agents}
            values_rollout = {agent_id: [] for agent_id in self.agents}
            log_probs_rollout = {agent_id: [] for agent_id in self.agents}
            infos_rollout = {agent_id: [] for agent_id in self.agents}

            for step in range(n_steps):
                with th.no_grad():
                    actions, values, log_probs = {agent_id: None for agent_id in self.agents}, {agent_id: None for agent_id in self.agents}, {agent_id: None for agent_id in self.agents}
                    for agent_id in self.agents:
                        local_obs = obs[agent_id]
                        global_obs = np.concatenate([obs[agent_id] for agent_id in self.agents])
                        action, value, log_prob = self.models[agent_id].policy.forward(local_obs, global_obs)
                        actions[agent_id] = action.squeeze().cpu().numpy()
                        values[agent_id] = value
                        log_probs[agent_id] = log_prob

                if self.assistive_gym:
                    new_obs, rewards, dones, infos = self.env.step(actions)
                else:
                    new_obs, rewards, dones, infos = self.env.step([actions[action] for action in actions])

                for agent_id in self.agents:
                    states_rollout[agent_id].append(obs[agent_id])
                    next_states_rollout[agent_id].append(new_obs[agent_id])
                    actions_rollout[agent_id].append(actions[agent_id])
                    rewards_rollout[agent_id].append(rewards[agent_id])
                    dones_rollout[agent_id].append(dones[agent_id])
                    values_rollout[agent_id].append(values[agent_id])
                    log_probs_rollout[agent_id].append(log_probs[agent_id])
                    infos_rollout[agent_id].append({})

                if not self.assistive_gym:
                    rewards = sum(rewards)
                    dones = all(dones)
                else:
                    rewards = (rewards['robot'] + rewards['human']) / 2
                    dones = dones['__all__']

                if dones:
                    obs = self.env.reset()
                else:
                    obs = new_obs
            
            for agent_id in self.agents:
                states_rollout[agent_id] = np.array(states_rollout[agent_id])
                next_states_rollout[agent_id] = np.array(next_states_rollout[agent_id])
                actions_rollout[agent_id] = np.array(actions_rollout[agent_id])
                rewards_rollout[agent_id] = np.array(rewards_rollout[agent_id])
                dones_rollout[agent_id] = np.array(dones_rollout[agent_id])
                values_rollout[agent_id] = th.as_tensor(values_rollout[agent_id])
                log_probs_rollout[agent_id] = th.as_tensor(log_probs_rollout[agent_id])

            [self.models[agent_id].learn(total_timesteps=10000000, states_rollout=states_rollout, next_states_rollout=next_states_rollout,
                        actions_rollout=actions_rollout[agent_id], rewards_rollout=rewards_rollout[agent_id], dones_rollout=dones_rollout[agent_id], values_rollout=values_rollout[agent_id],
                        log_probs_rollout=log_probs_rollout[agent_id], infos_rollout=infos_rollout[agent_id]) for agent_id in self.agents]
            
            print(f'epoch: {epoch} | avg length: {round(n_steps / np.sum(dones_rollout[list(dones_rollout.keys())[0]]))} | avg reward: {round(np.sum(rewards_rollout[list(dones_rollout.keys())[0]]) / np.sum(dones_rollout[list(dones_rollout.keys())[0]]), 2)}')
            self.save(path)
    
    def test(self, test_epochs=10, load_model=False, load_path=None, env_id=None):
        if load_model:
            assert load_path and env_id, 'Please provide load path and env id'
            print('Loading Model...')
            self.load(load_path, env_id)
            print('Loaded Models')
        
        test_rewards = []
        test_length = []
        
        for _ in range(test_epochs):
            dones = False
            rewards = 0
            length = 0
            obs = self.env_test.reset()
            while not dones:
                with th.no_grad():
                    actions = {agent_id: None for agent_id in self.agents}
                    for agent_id in self.agents:
                        local_obs = obs[agent_id]
                        global_obs = np.concatenate([obs[agent_id] for agent_id in self.agents])
                        action, value, log_prob = self.models[agent_id].policy.forward(local_obs, global_obs)
                        actions[agent_id] = action.squeeze().cpu().numpy()
                    
                if self.assistive_gym:
                    new_obs, rewards, dones, infos = self.env.step(actions)
                else:
                    new_obs, rewards, dones, infos = self.env.step([actions[action] for action in actions])

                reward = sum(reward)
                dones = all(dones)

                rewards += reward
                length += 1

                obs = new_obs

            test_rewards.append(rewards)
            test_length.append(length)

        print(
            f'mean length: {round(np.mean(test_length), 1)} | mean reward: {round(np.mean(test_rewards), 1)}')
            
    def save(self, path=os.getcwd()):
        for agent_id in self.agents:
            self.models[agent_id].save(f'{path}/{agent_id}')

    def load(self, path):
        for agent_id in self.agents:
            self.models[agent_id].set_parameters(f'{path}/{agent_id}')



if __name__ == '__main__':
    env_id = 'ma_gym:Checkers-v0'
    # env_id = 'ma_gym:DecHuRoSorting-v0'
    ppo = Dec_Train(env_id)
    # ppo.train(epochs=1000)
    ppo.test(load_model=True, load_path=os.getcwd(), env_id=env_id)
    
