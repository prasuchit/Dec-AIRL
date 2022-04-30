# This PPO is imported from stable-baselines3 and adjusted to our packages.
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
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor

import sys
sys.path.append(os.getcwd()+f'/gail-airl-ppo/')
from gail_airl_ppo.utils import normalize



SEED = 1
random.seed(SEED)
th.manual_seed(SEED)
np.random.seed(SEED)
th.use_deterministic_algorithms(True)
robot_state_EOF = 12


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

        self.observation_space_critic = observation_space
        self.observation_space_actor_dim = int(observation_space.shape[0] / 2)
        self.observation_space_actor = spaces.Box(low=np.zeros(self.observation_space_actor_dim), high=np.ones(self.observation_space_actor_dim))

        self.features_extractor_critic = features_extractor_class(self.observation_space_critic, **self.features_extractor_kwargs)
        self.features_extractor_actor = features_extractor_class(self.observation_space_actor, **self.features_extractor_kwargs)
        self.features_dim_critic = self.features_extractor_critic.features_dim
        self.features_dim_actor= self.features_extractor_actor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

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
        observation = observation.float()
        obs_actor = observation[..., :self.observation_space_actor_dim]
        _, latent_vf_critic = self.mlp_extractor_critic(observation)
        latent_pi_actor, _ = self.mlp_extractor_actor(obs_actor)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf_critic)
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
        observation = observation.float()
        obs_actor = observation[..., :self.observation_space_actor_dim]
        return self.get_distribution(obs_actor).get_actions(deterministic=deterministic)

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
        observation = observation.float()
        obs_actor = observation[..., :self.observation_space_actor_dim]
        _, latent_vf_critic = self.mlp_extractor_critic(observation)
        latent_pi_actor, _ = self.mlp_extractor_actor(obs_actor)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf_critic)
        distribution = self._get_action_dist_from_latent(latent_pi_actor)
        log_prob = distribution.log_prob(actions)
        # values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, observation) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        # features = self.extract_features(obs)
        observation = th.tensor(observation).float()
        obs_actor = observation[..., :self.observation_space_actor_dim]
        latent_pi = self.mlp_extractor_actor.forward_actor(obs_actor)
        return self._get_action_dist_from_latent(latent_pi)


class OnPolicyAlgorithm_Dec(OnPolicyAlgorithm):
    def __init__(self, policy: Type[BasePolicy], env: Union[GymEnv, str, None],
                 learning_rate: Union[float, Schedule], airl=False, **kwargs):
        super().__init__(policy, env, learning_rate, **kwargs)
        self.airl = airl
        self.airl_init = True
        print('Custom On-Policy Algo!!!')

    def collect_rollouts_airl(
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
        n_rollout_steps = states_rollout.shape[0]
        while n_steps < n_rollout_steps:
            self._last_obs = states_rollout[n_steps]
            clipped_actions, new_obs, rewards, dones, infos, values, log_probs = \
                actions_rollout[n_steps], next_states_rollout[n_steps], rewards_rollout[n_steps], dones_rollout[n_steps], \
                infos_rollout[n_steps], values_rollout[n_steps], log_probs_rollout[n_steps]
            self.num_timesteps += 1

            self._update_info_buffer(infos)
            n_steps += 1

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    raise NotImplementedError

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
        if self.airl:
            if self.airl_init:
                total_timesteps, callback = self._setup_learn(
                    total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
                    tb_log_name
                )

                callback.on_training_start(locals(), globals())
                self.airl_init = False
            continue_training = self.collect_rollouts_airl(self.rollout_buffer, **kwargs)

            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            self.train()
            return self

        raise ValueError('Should not call this')


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
            airl: bool = False
    ):

        super(PPO_Dec, self).__init__(
            policy,
            env,
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
            airl=airl
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        assert (
                batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                    buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

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

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = normalize(advantages)

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


class Train(object):
    def __init__(self, env_id, device, seed=SEED):
        self.env = gym.make(env_id)
        self.env.seed(seed)
        self.eval_env = gym.make(env_id)
        self.eval_env.seed(seed)
        self.action_space = self.env.action_space
        self.device = device
        self.seed = seed
        self.model_h = None
        self.model_r = None
        self.fixed_init = False
        print(f'Env: {env_id}')

    def forward_rl(self, total_timesteps=25000):
        print("Normal forward RL")
        model = PPO_Dec("MlpPolicy", self.env, verbose=1, device=self.device, seed=self.seed)
        self.model = model
        model.learn(total_timesteps=total_timesteps)

    def airl(self, airl_epochs=1000, n_steps=2048, batch_size=64, n_epochs=10, eval_interval=5, eval_epochs=10):
        print("AIRL RL")
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y-%H-%M")
        # self.path = f'models/{timestamp}'
        self.path = script_dir + f'/models/{timestamp}'
        os.mkdir(self.path)

        model_r = PPO_Dec(ActorCriticPolicy_Dec, self.env, verbose=1, airl=True, device=self.device, seed=self.seed)
        model_r.set_parameters(script_dir + f'/models/04-12-2022-10-51/model_r_95.zip', device=self.device)
        self.model_r = model_r

        model_h = PPO_Dec(ActorCriticPolicy_Dec, self.env, verbose=1, airl=True, device=self.device, seed=self.seed)
        model_h.set_parameters(script_dir + f'/models/04-12-2022-10-51/model_h_95.zip', device=self.device)
        self.model_h = model_h

        state = self.env.reset(self.fixed_init) # [state of robot, state of human]
        state_robot = state[:robot_state_EOF].copy()
        state_human = state[robot_state_EOF:].copy()
        state_robot_input = np.concatenate([state_robot.copy(), state_human.copy()])
        state_human_input = np.concatenate([state_human.copy(), state_robot.copy()])
        for airl_epoch in range(1, airl_epochs + 1):
            states_h_rollout = []
            next_states_h_rollout = []
            actions_h_rollout = []
            rewards_h_rollout = []
            dones_h_rollout = []
            values_h_rollout = []
            log_probs_h_rollout = []
            infos_h_rollout = []

            states_r_rollout = []
            next_states_r_rollout = []
            actions_r_rollout = []
            rewards_r_rollout = []
            dones_r_rollout = []
            values_r_rollout = []
            log_probs_r_rollout = []
            infos_r_rollout = []

            for step in range(n_steps):
                with th.no_grad():
                    epsilon = 0.0
                    if np.random.rand() < epsilon:
                        actions_r = th.tensor(np.random.randint(low=0, high=self.env.action_space.n))
                        values_r, log_probs_r, _ = self.model_r.policy.evaluate_actions(obs_as_tensor(state_robot_input, device=self.device).float(), actions_r)
                    else:
                        actions_r, values_r, log_probs_r = self.model_r.policy.forward(obs_as_tensor(state_robot_input, device=self.device).float())

                    if np.random.rand() < epsilon:
                        actions_h = th.tensor(np.random.randint(low=0, high=self.env.action_space.n))
                        values_h, log_probs_h, _ = self.model_h.policy.evaluate_actions(obs_as_tensor(state_human_input, device=self.device).float(), actions_h)
                    else:
                        actions_h, values_h, log_probs_h = self.model_h.policy.forward(obs_as_tensor(state_human_input, device=self.device).float())

                actions = [actions_r.cpu().numpy(), actions_h.cpu().numpy()]

                clipped_actions = copy(actions)
                
                new_obs, rewards, dones, infos = self.env.step(clipped_actions, verbose=0)
                n_state_robot = new_obs[:robot_state_EOF].copy()
                n_state_human = new_obs[robot_state_EOF:].copy()
                n_state_robot_input = np.concatenate([n_state_robot.copy(), n_state_human.copy()])
                n_state_human_input = np.concatenate([n_state_human.copy(), n_state_robot.copy()])

                states_r_rollout.append(state_robot_input)
                actions_r_rollout.append(actions_r)
                rewards_r_rollout.append(rewards)
                dones_r_rollout.append([dones])
                values_r_rollout.append(values_r)
                log_probs_r_rollout.append(log_probs_r)
                infos_r_rollout.append([infos])
                next_states_r_rollout.append(n_state_robot_input)

                states_h_rollout.append(state_human_input)
                actions_h_rollout.append(actions_h)
                rewards_h_rollout.append(rewards)
                dones_h_rollout.append([dones])
                values_h_rollout.append(values_h)
                log_probs_h_rollout.append(log_probs_h)
                infos_h_rollout.append([infos])
                next_states_h_rollout.append(n_state_human_input)

                if dones:
                    state = self.env.reset(self.fixed_init)
                else:
                    state = new_obs
                state_robot = state[:robot_state_EOF].copy()
                state_human = state[robot_state_EOF:].copy()
                state_robot_input = np.concatenate([state_robot.copy(), state_human.copy()])
                state_human_input = np.concatenate([state_human.copy(), state_robot.copy()])

            states_r_rollout = np.array(states_r_rollout)
            next_states_r_rollout = np.array(next_states_r_rollout)
            actions_r_rollout = np.array(actions_r_rollout)
            rewards_r_rollout = np.array(rewards_r_rollout)
            dones_r_rollout = np.array(dones_r_rollout)
            values_r_rollout = th.tensor(values_r_rollout)
            log_probs_r_rollout = th.tensor(log_probs_r_rollout)

            states_h_rollout = np.array(states_h_rollout)
            next_states_h_rollout = np.array(next_states_h_rollout)
            actions_h_rollout = np.array(actions_h_rollout)
            rewards_h_rollout = np.array(rewards_h_rollout)
            dones_h_rollout = np.array(dones_h_rollout)
            values_h_rollout = th.tensor(values_h_rollout)
            log_probs_h_rollout = th.tensor(log_probs_h_rollout)

            self.model_r.learn(total_timesteps=10000000, states_rollout=states_r_rollout, next_states_rollout=next_states_r_rollout,
                        actions_rollout=actions_r_rollout, rewards_rollout=rewards_r_rollout, dones_rollout=dones_r_rollout, values_rollout=values_r_rollout,
                        log_probs_rollout=log_probs_r_rollout, infos_rollout=infos_r_rollout)
            
            self.model_h.learn(total_timesteps=10000000, states_rollout=states_h_rollout, next_states_rollout=next_states_h_rollout,
                        actions_rollout=actions_h_rollout, rewards_rollout=rewards_h_rollout, dones_rollout=dones_h_rollout, values_rollout=values_h_rollout,
                        log_probs_rollout=log_probs_h_rollout, infos_rollout=infos_h_rollout)

            print(f'epoch: {airl_epoch} | avg length: {round(n_steps / np.sum(dones_h_rollout))} | training number of positive rewards: {round(np.sum(rewards_r_rollout > 0), 2)} | training number of negative rewards: {round(np.sum(rewards_r_rollout < 0), 2)}')

            if airl_epoch % eval_interval == 0:
                self.evaluation(1, verbose=True, deterministic=False)
                self.model_r.save(f'{self.path}/model_r_{airl_epoch}')
                self.model_h.save(f'{self.path}/model_h_{airl_epoch}')

    def evaluation(self, eval_epochs=10, verbose=False, deterministic=True):
        ep_rewards = []
        ep_lengths = []
        for eval_epoch in range(eval_epochs):
            eval_state = self.eval_env.reset(self.fixed_init)
            state_robot = eval_state[:robot_state_EOF].copy()
            state_human = eval_state[robot_state_EOF:].copy()
            state_robot_input = np.concatenate([state_robot.copy(), state_human.copy()])
            state_human_input = np.concatenate([state_human.copy(), state_robot.copy()])
            eval_done = False
            ep_length = 0
            ep_reward = 0
            while not eval_done:
                eval_action_r, _ = self.model_r.predict(state_robot_input, deterministic=deterministic)
                eval_action_h, _ = self.model_h.predict(state_human_input, deterministic=deterministic)
                eval_next_state, eval_reward, eval_done, eval_info = self.eval_env.step([eval_action_r, eval_action_h], verbose=verbose)
                ep_reward += eval_reward
                ep_length += 1
                eval_state = eval_next_state
                state_robot = eval_state[:robot_state_EOF]
                state_human = eval_state[robot_state_EOF:]
                state_robot_input = np.concatenate([state_robot, state_human])
                state_human_input = np.concatenate([state_human, state_robot])
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)

        print(f'mean length: {round(np.mean(ep_lengths), 1)} | mean reward: {round(np.mean(ep_rewards), 1)}')
    
    def test(self, epoch, path):
        model_h = PPO_Dec(ActorCriticPolicy_Dec, self.env, verbose=1, airl=True, device=self.device, seed=self.seed)
        model_r = PPO_Dec(ActorCriticPolicy_Dec, self.env, verbose=1, airl=True, device=self.device, seed=self.seed)
        model_r.load(f'{path}/model_r_{epoch}')
        model_h.load(f'{path}/model_h_{epoch}')
        eval_state = self.eval_env.reset(self.fixed_init)
        state_robot = eval_state[:robot_state_EOF]
        state_human = eval_state[robot_state_EOF:]
        state_robot_input = np.concatenate([state_robot, state_human])
        state_human_input = np.concatenate([state_human, state_robot])
        eval_done = False
        ep_length = 0
        ep_reward = 0
        while not eval_done:
            eval_action_r, _ = model_r.predict(state_robot_input, deterministic=True)
            eval_action_h, _ = model_h.predict(state_human_input, deterministic=True)
            eval_next_state, eval_reward, eval_done, eval_info = self.eval_env.step([eval_action_r, eval_action_h], verbose=True)
            ep_reward += eval_reward
            ep_length += 1
            eval_state = eval_next_state
            state_robot = eval_state[:robot_state_EOF]
            state_human = eval_state[robot_state_EOF:]
            state_robot_input = np.concatenate([state_robot, state_human])
            state_human_input = np.concatenate([state_human, state_robot])
        print(f'Reward Mean: {round(np.mean(ep_reward), 2)} | Episode Mean Length: {round(np.mean(ep_length), 2)}')

if __name__ == '__main__':
    env_id = 'ma_gym:HuRoSorting-v0'
    airl = Train(env_id, th.device('cpu'))
    airl.test(epoch = 1010, path = 'models/04-12-2022-12-46')
    # airl.airl(airl_epochs=5000)