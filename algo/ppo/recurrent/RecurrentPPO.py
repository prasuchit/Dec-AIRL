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

from algo.ppo.recurrent.RecurrentActorCritic import RecurrentActorCriticPolicy_Dec
from algo.ppo.recurrent.RecurrentBuffer import RecurrentRolloutBuffer_Dec
from algo.ppo.type_aliases import RNNStates
from algo.ppo.Base import BasePolicy_Dec
from algo.ppo.ppo import PPO_Dec
from typing import TypeVar
from utils import normalize, get_assistive_gym_envs_list, obs_as_tensor
from copy import copy, deepcopy
import gym
from gym import spaces
import os
import torch as th
from torch.nn import functional as F
from datetime import datetime
import numpy as np
import time
from typing import Any, Dict, Optional, Type, Union

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy

import random
import collections
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union


import sys
path = os.path.dirname(os.path.realpath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

sys.path.append(PACKAGE_PATH)

SEED = 1
random.seed(SEED)
th.manual_seed(SEED)
np.random.seed(SEED)
th.use_deterministic_algorithms(True)

SelfRecurrentPPO_Dec = TypeVar(
    "SelfRecurrentPPO_Dec", bound="RecurrentPPO_Dec")


class RecurrentPPO_Dec(PPO_Dec):
    """
    Decentralized Proximal Policy Optimization algorithm (PPO) (clip version)
    with support for recurrent policies (LSTM).

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
    policy_aliases: Dict[str, Type[BasePolicy_Dec]] = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
    }

    def __init__(
            self,
            policy: Union[str, Type[RecurrentActorCriticPolicy_Dec]],
            env: Union[GymEnv, str],
            agent_id: int,
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 128,
            batch_size: Optional[int] = 128,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            custom_rollout: bool = False
    ):

        super().__init__(
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

        self.env = env
        self.env_test = env
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.agent_id = agent_id
        self._last_lstm_states = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        assert not (
            isinstance(self.local_observation_space, spaces.Dict)
        ), "Does not support Dict spaces for observations currently."

        buffer_cls = RecurrentRolloutBuffer_Dec

        self.policy = self.policy_class(
            local_observation_space=self.local_observation_space,
            global_observation_space=self.global_observation_space,
            action_space=self.action_space,
            agent_id=self.agent_id,
            lr_schedule=self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # We assume that LSTM for the actor and the critic
        # have the same architecture
        lstm = self.policy.lstm_actor

        if not isinstance(self.policy, RecurrentActorCriticPolicy_Dec):
            raise ValueError(
                "Policy must subclass RecurrentActorCriticPolicy_Dec")

        single_hidden_state_shape = (
            lstm.num_layers, self.n_envs, lstm.hidden_size)
        # hidden and cell states for actor and critic
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )

        hidden_state_buffer_shape = (
            self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.local_observation_space,
            self.global_observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RecurrentRolloutBuffer_Dec,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        raise NotImplementedError

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
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining)

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

                # Convert mask from float to bool
                mask = rollout_data.mask > 1e-8

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.local_observations,
                    rollout_data.global_observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (
                    advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * \
                    th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = - \
                    th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()[mask]).item()
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
                # Mask padded sequences
                # value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        ((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates",
                           self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
            
    def _custom_rollouts(
            self,
            # env: VecEnv,
            # callback: BaseCallback,
            rollout_buffer: RecurrentRolloutBuffer_Dec,
            # n_rollout_steps: int,
            states_rollout,
            actions_rollout,
            next_states_rollout,
            rewards_rollout,
            dones_rollout,
            values_rollout,
            log_probs_rollout,
            lstm_states_rollout,
            infos_rollout,            
    ) -> bool:
        # assert self._last_obs is None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        keys = list(states_rollout.keys())
        n_rollout_steps = states_rollout[keys[0]].shape[0]

        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            self._last_local_obs = states_rollout[self.agent_id][n_steps]
            self._last_global_obs = np.concatenate([states_rollout[i][n_steps] for i in keys])
            self._last_lstm_states = deepcopy(lstm_states_rollout[self.agent_id][n_steps])
            clipped_actions, rewards, dones, values, log_probs = \
                actions_rollout[n_steps], rewards_rollout[n_steps], dones_rollout[n_steps], \
                values_rollout[n_steps], log_probs_rollout[n_steps]
            local_new_obs = next_states_rollout[self.agent_id][n_steps]
            global_new_obs = np.concatenate([next_states_rollout[i][n_steps] for i in keys])
            self.num_timesteps += 1

            self._update_info_buffer(infos_rollout)
            n_steps += 1
            
            # if isinstance(self.action_space, spaces.Discrete):
            #     # Reshape in case of discrete action
            #     clipped_actions = clipped_actions.reshape(-1, 1)

            # # Handle timeout by bootstraping with value function
            # # see GitHub issue #633
            # for idx, done_ in enumerate(dones_rollout):
            #     if (
            #         done_
            #         and infos_rollout[idx].get("terminal_observation") is not None
            #         and infos_rollout[idx].get("TimeLimit.truncated", False)
            #     ):
            #         terminal_obs = self.policy.obs_to_tensor(infos_rollout[idx]["terminal_observation"])[0]
            #         with th.no_grad():
            #             terminal_lstm_state = (
            #                 lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
            #                 lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
            #             )
            #             # terminal_lstm_state = None
            #             episode_starts = th.tensor([False], dtype=th.float32, device=self.device)
            #             terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]
            #         rewards[idx] += self.gamma * terminal_value

            buffer_full = rollout_buffer.add(self._last_local_obs, self._last_global_obs, clipped_actions, rewards, self._last_episode_starts, values, log_probs, lstm_states=self._last_lstm_states)
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states
            if buffer_full:
                break

        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(global_new_obs, lstm_states.vf, episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        return True            

    def learn(
        self: SelfRecurrentPPO_Dec,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "RecurrentPPO_Dec",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        **kwargs
    ) -> SelfRecurrentPPO_Dec:
        
        iteration = 0
        if self.custom_rollout:
            if self.custom_init:
                total_timesteps, callback = self._setup_learn(
                                                                total_timesteps,
                                                                eval_env, 
                                                                callback,
                                                                eval_freq, 
                                                                n_eval_episodes, 
                                                                eval_log_path,
                                                                reset_num_timesteps,
                                                                tb_log_name,
                                                            )
            callback.on_training_start(locals(), globals())
            self.custom_init = False
            while self.num_timesteps < total_timesteps:
                continue_training = self._custom_rollouts(self.rollout_buffer, **kwargs)

                if continue_training is False:
                    break

                iteration += 1
                self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

                # Display training infos
                if log_interval is not None and iteration % log_interval == 0:
                    time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                    fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                    self.logger.record("time/iterations", iteration, exclude="tensorboard")
                    if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                        self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                        self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("time/fps", fps)
                    self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                    self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                    self.logger.dump(step=self.num_timesteps)

                self.train()

            callback.on_training_end()

            return self
        else:
            raise ValueError('Should not call this')


class RecurrentDec_Train():
    def __init__(self, env_id, device='cpu', seed=1024) -> None:
        self.env_id = env_id
        self.seed = seed
        self.n_envs = 1
        # if the env belongs to assistive gym
        if env_id in get_assistive_gym_envs_list():
            # if the env is cooperative
            if 'Human' in env_id:
                import importlib
                module = importlib.import_module('assistive_gym.envs')
                env_class = getattr(module, env_id.split('-')[0] + 'Env')
                self.env = env_class()
                self.env_test = env_class()
            else:
                self.env = gym.make('assistive_gym:' + env_id)
                self.env_test = gym.make('assistive_gym:' + env_id)
            self.assistive_gym = True
        else:
            # Training Env
            self.env = gym.make(env_id)
            self.env.seed(self.seed)

            # Testing Env
            self.env_test = gym.make(env_id)
            self.env_test.seed(self.seed)
            self.assistive_gym = False

        self.device = device
        # init agents
        if self.assistive_gym:
            self.agents = ['robot', 'human']
            self.models = {agent_id: RecurrentPPO_Dec(RecurrentActorCriticPolicy_Dec, self.env, agent_id=agent_id,
                                                      verbose=1, custom_rollout=True, device=self.device, seed=self.seed) for agent_id in self.agents}
        else:
            self.agents = list(range(self.env.n_agents))
            self.models = {agent_id: RecurrentPPO_Dec(RecurrentActorCriticPolicy_Dec, self.env, agent_id=agent_id,
                                                      verbose=1, custom_rollout=True, device=self.device, seed=self.seed) for agent_id in self.agents}

    def train(self, epochs=10, n_steps=2048, path=os.getcwd()):
        self.env.seed(int(time.time()))
        ep_len = 0
        lengths = []
        obs = self.env.reset()
        # Episode start signals are used to reset the lstm states
        episode_starts = th.ones((self.n_envs,))
        for epoch in range(epochs):
            # init custom collects
            states_rollout = {agent_id: [] for agent_id in self.agents}
            next_states_rollout = {agent_id: [] for agent_id in self.agents}
            actions_rollout = {agent_id: [] for agent_id in self.agents}
            rewards_rollout = {agent_id: [] for agent_id in self.agents}
            dones_rollout = {agent_id: [] for agent_id in self.agents}
            values_rollout = {agent_id: [] for agent_id in self.agents}
            log_probs_rollout = {agent_id: [] for agent_id in self.agents}
            lstm_states_rollout = {agent_id: [] for agent_id in self.agents}
            infos_rollout = {agent_id: [] for agent_id in self.agents}
            
            # lstm = self.models[0].policy.lstm_actor
            # single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)

            for _ in range(n_steps):
                with th.no_grad():
                    actions, values, log_probs, lstm_states = \
                        {agent_id: None for agent_id in self.agents},\
                        {agent_id: None for agent_id in self.agents},\
                        {agent_id: None for agent_id in self.agents},\
                        {agent_id: self.models[agent_id]._last_lstm_states for agent_id in self.agents}

                    for agent_id in self.agents:
                        local_obs = obs[agent_id]
                        global_obs = np.concatenate(
                            [obs[agent_id] for agent_id in self.agents])
                        action, value, log_prob, lstm_states[agent_id] = self.models[agent_id].policy.forward(
                                                                            local_obs, global_obs, lstm_states[agent_id], episode_starts)
                        actions[agent_id] = action.squeeze().cpu().numpy()
                        values[agent_id] = value
                        log_probs[agent_id] = log_prob

                if self.assistive_gym:
                    new_obs, rewards, dones, infos = self.env.step(actions)
                else:
                    new_obs, rewards, dones, infos = self.env.step(
                        [actions[action] for action in actions])

                ep_len += 1

                for agent_id in self.agents:
                    states_rollout[agent_id].append(obs[agent_id])
                    next_states_rollout[agent_id].append(new_obs[agent_id])
                    actions_rollout[agent_id].append(actions[agent_id])
                    rewards_rollout[agent_id].append(rewards[agent_id])
                    dones_rollout[agent_id].append(dones[agent_id])
                    values_rollout[agent_id].append(values[agent_id])
                    log_probs_rollout[agent_id].append(log_probs[agent_id])
                    lstm_states_rollout[agent_id].append(lstm_states[agent_id])
                    infos_rollout[agent_id].append({})

                if not self.assistive_gym:
                    rewards = sum(rewards) / 2
                    dones = all(dones)
                else:
                    rewards = (rewards['robot'] + rewards['human']) / 2
                    dones = dones['__all__']

                if dones:
                    lengths.append(ep_len)
                    ep_len = 0
                    self.env.seed(int(time.time()))
                    obs = self.env.reset()
                else:
                    obs = new_obs

            for agent_id in self.agents:
                states_rollout[agent_id] = np.array(states_rollout[agent_id])
                next_states_rollout[agent_id] = np.array(
                    next_states_rollout[agent_id])
                actions_rollout[agent_id] = np.array(actions_rollout[agent_id])
                rewards_rollout[agent_id] = np.array(rewards_rollout[agent_id])
                dones_rollout[agent_id] = np.array(dones_rollout[agent_id])
                values_rollout[agent_id] = th.as_tensor(
                    values_rollout[agent_id])
                log_probs_rollout[agent_id] = th.as_tensor(
                    log_probs_rollout[agent_id])
                # lstm_states_rollout[agent_id] = th.as_tensor(
                #     lstm_states_rollout[agent_id])

            [self.models[agent_id].learn(total_timesteps=10000000, states_rollout=states_rollout, next_states_rollout=next_states_rollout,
                                         actions_rollout=actions_rollout[agent_id], rewards_rollout=rewards_rollout[agent_id], 
                                         dones_rollout=dones_rollout[agent_id], values_rollout=values_rollout[agent_id], 
                                         log_probs_rollout=log_probs_rollout[agent_id], lstm_states_rollout=lstm_states_rollout, 
                                         infos_rollout=infos_rollout[agent_id]) for agent_id in self.agents]

            print(f'epoch: {epoch} | avg length: {round(np.mean(lengths), 2)} | avg reward: {round(np.sum(rewards_rollout[list(dones_rollout.keys())[0]]) / np.sum(dones_rollout[list(dones_rollout.keys())[0]]), 2)}')
            self.save(path)

    def test(self, test_epochs=10, load_model=False, load_path=None, env_id=None):
        if load_model:
            assert load_path and env_id, 'Please provide load path and env id'
            print(f'Loading Model from {load_path} for env: {env_id}')
            self.load(load_path)
            print('Loaded Models')

        test_rewards = []
        test_length = []
        self.env_test.render()

        for _ in range(test_epochs):
            dones = False
            rewards = 0
            length = 0
            # self.env.seed(int(time.time()))
            obs = self.env_test.reset()
            while not dones:
                with th.no_grad():
                    actions = {agent_id: None for agent_id in self.agents}
                    for agent_id in self.agents:
                        local_obs = obs[agent_id]
                        global_obs = np.concatenate(
                            [obs[agent_id] for agent_id in self.agents])
                        action, value, log_prob = self.models[agent_id].policy.forward(
                            local_obs, global_obs)
                        actions[agent_id] = action.squeeze().cpu().numpy()

                if self.assistive_gym:
                    new_obs, rewards, dones, infos = self.env_test.step(
                        actions)
                else:
                    new_obs, rewards, dones, infos = self.env_test.step(
                        [actions[action] for action in actions])

                if not self.assistive_gym:
                    rewards = sum(rewards) / 2
                    dones = all(dones)
                else:
                    rewards = (rewards['robot'] + rewards['human']) / 2
                    dones = dones['__all__']
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
    ...
