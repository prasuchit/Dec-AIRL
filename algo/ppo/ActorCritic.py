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

''' Actor-Critic class is imported from stable-baselines3 and adjusted to work with our algorithm. '''

from copy import copy
import gym
from gym import spaces
import os
import sys
import torch as th
import torch
from torch.nn import functional as F
from datetime import datetime
import numpy as np
import time
import warnings
from typing import Any, Dict, Optional, Type, Union

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
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

sys.path.append(os.getcwd() + f'/airl-ppo/')
from algo.ppo.Base import BaseAlgorithm_Dec
from algo.ppo.Buffer import RolloutBuffer_Dec
from utils import obs_as_tensor


class ActorCriticPolicy_Dec(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        agent_id: int, 
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(ActorCriticPolicy_Dec, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.agent_id = agent_id

        # Check the observation space

        if type(observation_space).__name__ == 'MultiAgentObservationSpace':
            self.n_agent = len(observation_space)
            self.local_observation_dim = observation_space[0].shape[0]
            self.global_observation_dim = self.n_agent * self.local_observation_dim
        else:
            raise ValueError('Only Support Multi Agent Environment')

        assert type(action_space).__name__ == 'MultiAgentActionSpace'

        self.observation_space_critic_dim = self.global_observation_dim
        self.observation_space_actor_dim = self.local_observation_dim
        self.observation_space_actor = spaces.Box(low=np.zeros(self.observation_space_actor_dim), high=np.ones(self.observation_space_actor_dim))

        obs_low = [observation_space[0].low] * self.n_agent
        obs_high = [observation_space[0].high] * self.n_agent
        self.observation_space_critic = spaces.Box(low=np.array(obs_low).flatten(), high=np.array(obs_high).flatten())

        self.features_extractor_critic = features_extractor_class(self.observation_space_critic, **self.features_extractor_kwargs)
        self.features_extractor_actor = features_extractor_class(self.observation_space_actor, **self.features_extractor_kwargs)
        self.features_dim_critic = self.features_extractor_critic.features_dim
        self.features_dim_actor= self.features_extractor_actor.features_dim

        # Action distribution
        self.action_dist = make_proba_distribution(action_space[0], use_sde=False, dist_kwargs=None)

        self._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor_critic = MlpExtractor(
            self.features_dim_critic,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

        self.mlp_extractor_actor = MlpExtractor(
            self.features_dim_actor,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        assert self.mlp_extractor_critic.latent_dim_pi == self.mlp_extractor_actor.latent_dim_pi
        latent_dim_pi = self.mlp_extractor_critic.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor_critic.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor_critic: np.sqrt(2),
                self.features_extractor_actor: np.sqrt(2),
                self.mlp_extractor_critic: np.sqrt(2),
                self.mlp_extractor_actor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, observation: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        # features = self.extract_features(obs)
        # observation = th.tensor(observation).float()

        # get global and local observation
        global_observation = observation.reshape((observation.shape[0],-1))
        local_observation = observation[:, self.agent_id]

        # critic
        _, latent_vf_critic = self.mlp_extractor_critic(global_observation)

        # actor
        latent_pi_actor, _ = self.mlp_extractor_actor(local_observation)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf_critic)
        
        # get action distribution
        distribution = self._get_action_dist_from_latent(latent_pi_actor)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        # observation = observation.float()

        # get global and local observation
        global_observation = observation.reshape((observation.shape[0],-1))
        local_observation = observation[:, self.agent_id]

        # obs_actor = observation[..., :self.observation_space_actor_dim]
        return self.get_distribution(local_observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, observation: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        # features = self.extract_features(obs)
        # latent_pi, latent_vf = self.mlp_extractor(features)
        # distribution = self._get_action_dist_from_latent(latent_pi)
        # observation = observation.float()

        # get global and local observation
        global_observation = observation.reshape((observation.shape[0],-1))
        local_observation = observation[:, self.agent_id]

        # obs_actor = observation[..., :self.observation_space_actor_dim]
        _, latent_vf_critic = self.mlp_extractor_critic(global_observation)
        latent_pi_actor, _ = self.mlp_extractor_actor(local_observation)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf_critic)
        distribution = self._get_action_dist_from_latent(latent_pi_actor)
        actions = actions.squeeze()
        assert len(actions.shape) == 1
        log_prob = distribution.log_prob(actions)
        # values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, observation: th.tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        # features = self.extract_features(obs)
        # observation = th.tensor(observation).float()
        # obs_actor = observation[..., :self.observation_space_actor_dim]
        latent_pi = self.mlp_extractor_actor.forward_actor(observation)
        return self._get_action_dist_from_latent(latent_pi)


class OnPolicyAlgorithm_Dec(BaseAlgorithm_Dec):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        agent_id: int,
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        policy_base: Type[BasePolicy] = ActorCriticPolicy,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        custom_rollout: bool=True
    ):

        super(OnPolicyAlgorithm_Dec, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.custom_rollout = custom_rollout
        self.agent_id = agent_id
        self.custom_init = True
        
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = RolloutBuffer_Dec

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.agent_id,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
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
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = obs_as_tensor(new_obs, self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def custom_rollouts(
            self,
            # env: VecEnv,
            # callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            # n_rollout_steps: int,
            states_rollout,
            actions_rollout,
            next_states_rollout,
            rewards_rollout,
            dones_rollout,
            values_rollout,
            log_probs_rollout,
            infos_rollout
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
 
        assert states_rollout.shape[0] == actions_rollout.shape[0] == next_states_rollout.shape[0] == \
               rewards_rollout.shape[0] == dones_rollout.shape[0] == log_probs_rollout.shape[0]
        assert states_rollout.shape == next_states_rollout.shape
        assert actions_rollout.shape == rewards_rollout.shape == dones_rollout.shape
        assert log_probs_rollout.shape == values_rollout.shape
        
        n_rollout_steps = states_rollout.shape[0]
        while n_steps < n_rollout_steps:
            self._last_obs = states_rollout[n_steps]
            clipped_actions, new_obs, rewards, dones, infos, values, log_probs = \
                actions_rollout[n_steps], next_states_rollout[n_steps], rewards_rollout[n_steps], dones_rollout[n_steps], \
                infos_rollout[n_steps], values_rollout[n_steps], log_probs_rollout[n_steps]
            self.num_timesteps += 1

            # self._update_info_buffer(infos)
            n_steps += 1

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            # for idx, done in enumerate(dones):
            #     if (
            #             done
            #             and infos[idx].get("terminal_observation") is not None
            #             and infos[idx].get("TimeLimit.truncated", False)
            #     ):
            #         raise NotImplementedError

            rollout_buffer.add(self._last_obs, clipped_actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_episode_starts = dones

        with th.no_grad():
            obs_tensor = obs_as_tensor(new_obs, self.device).float()
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        return True

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "OnPolicyAlgorithm",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            **kwargs
    ) -> "OnPolicyAlgorithm":
        if self.custom_rollout:
            if self.custom_init:
                total_timesteps, callback = self._setup_learn(
                    total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
                    tb_log_name
                )

                callback.on_training_start(locals(), globals())
                self.custom_init = False
            continue_training = self.custom_rollouts(self.rollout_buffer, **kwargs)

            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            self.train()
            return self

        raise ValueError('Should not call this')
