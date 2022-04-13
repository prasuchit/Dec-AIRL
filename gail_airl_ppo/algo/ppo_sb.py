# This PPO is imported from stable-baselines3 and adjusted to our packages.
import gym
from gym import spaces

import torch as th
from torch.nn import functional as F

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

SEED = 1024
random.seed(SEED)
th.manual_seed(SEED)
np.random.seed(SEED)
th.use_deterministic_algorithms(True)


class OnPolicyAlgorithm_AIRL(OnPolicyAlgorithm):
    def __init__(self, policy: Type[BasePolicy], env: Union[GymEnv, str, None],
                 learning_rate: Union[float, Schedule], airl=False, **kwargs):
        super().__init__(policy, env, learning_rate, **kwargs)
        self.airl = airl
        self.airl_init = True
        print('Custom On-Policy Algo!!!')

    def collect_rollouts_airl(
            self,
            # env: VecEnv,
            callback: BaseCallback,
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
        # Sample new weights for the state dependent exploration
        # if self.use_sde:
        #     self.policy.reset_noise(env.num_envs)

        # callback.on_rollout_start()

        assert states_rollout.shape[0] == actions_rollout.shape[0] == next_states_rollout.shape[0] == \
               rewards_rollout.shape[0] == dones_rollout.shape[0] == log_probs_rollout.shape[0]
        n_rollout_steps = states_rollout.shape[0]
        while n_steps < n_rollout_steps:
            self._last_obs = states_rollout[n_steps]
            #     if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
            #         # Sample a new noise matrix
            #         self.policy.reset_noise(env.num_envs)
            #
            #     with th.no_grad():
            #         # Convert to pytorch tensor or to TensorDict
            # obs_tensor = obs_as_tensor(self._last_obs, self.device)
            #         actions, values, log_probs = self.policy.forward(obs_tensor)
            #     actions = actions.cpu().numpy()
            #
            #     # Rescale and perform action
            # clipped_actions = actions_rollout[n_steps]
            # Clip the actions to avoid out of bound error
            # if isinstance(self.action_space, gym.spaces.Box):
            #     assert all(self.action_space.low <= clipped_actions <= self.action_space.high)
            #     clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            #
            #     new_obs, rewards, dones, infos = env.step(clipped_actions)

            # self._last_obs: np.array with (1, 4)
            # new_obs: np.array with (1, 4)
            # clipped_actions: np.array with (1)
            # rewards: np.array with (1)
            # dones: np.array with (1), boolean
            # infos: list of info [{}]
            # values: tensor with (1, 1)
            # log_probs: tensor with (1)

            clipped_actions, new_obs, rewards, dones, infos, values, log_probs = \
                actions_rollout[n_steps], next_states_rollout[n_steps], rewards_rollout[n_steps], dones_rollout[n_steps], \
                infos_rollout[n_steps], values_rollout[n_steps], log_probs_rollout[n_steps]
            # self.num_timesteps += env.num_envs
            self.num_timesteps += 1

            # Give access to local variables
            # callback.update_locals(locals())
            # if callback.on_step() is False:
            #     return False

            self._update_info_buffer(infos)
            n_steps += 1

            # if isinstance(self.action_space, gym.spaces.Discrete):
            #     # Reshape in case of discrete action
            #     actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, clipped_actions, rewards, self._last_episode_starts, values, log_probs)
            # self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            # values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
            obs_tensor = obs_as_tensor(new_obs, self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        # callback.on_rollout_end()

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

            # iteration = kwargs['iteration']
            continue_training = self.collect_rollouts_airl(callback, self.rollout_buffer, **kwargs)

            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            # if log_interval is not None and iteration % log_interval == 0:
            #     fps = int(self.num_timesteps / (time.time() - self.start_time))
            #     self.logger.record("time/iterations", iteration, exclude="tensorboard")
            #     if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            #         self.logger.record("rollout/ep_rew_mean",
            #                            safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            #         self.logger.record("rollout/ep_len_mean",
            #                            safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            #     self.logger.record("time/fps", fps)
            #     self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
            #     self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            #     self.logger.dump(step=self.num_timesteps)

            self.train()

            # callback.on_training_end()

            return self

        # Normal forward PPO training starts here
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self


class PPO_AIRL(OnPolicyAlgorithm_AIRL):
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

        super(PPO_AIRL, self).__init__(
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
        super(PPO_AIRL, self)._setup_model()

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

        return super(PPO_AIRL, self).learn(
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
        self.model = None
        print(f'Env: {env_id}')

    def forward_rl(self, total_timesteps=25000):
        print("Normal forward RL")
        model = PPO_AIRL("MlpPolicy", self.env, verbose=1, device=self.device, seed=self.seed)
        self.model = model
        model.learn(total_timesteps=total_timesteps)

    def airl(self, airl_epochs=1000, n_steps=2048, batch_size=64, n_epochs=10, eval_interval=5, eval_epochs=10):
        print("AIRL RL")
        model = PPO_AIRL("MlpPolicy", self.env, verbose=1, airl=True, device=self.device, seed=self.seed)
        self.model = model
        state = self.env.reset()[None, :]
        for airl_epoch in range(1, airl_epochs + 1):
            # state = self.env.reset()[None, :]

            states_airl = []
            next_states_airl = []
            actions_airl = []
            rewards_airl = []
            dones_airl = []
            values_airl = []
            log_probs_airl = []
            infos_airl = []

            for step in range(n_steps):
                with th.no_grad():
                    actions, values, log_probs = model.policy.forward(obs_as_tensor(state, device=self.device))

                actions = actions.cpu().numpy()

                clipped_actions = actions
                # Clip the actions to avoid out of bound error
                if isinstance(self.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
                elif isinstance(self.action_space, gym.spaces.Discrete):
                    clipped_actions = clipped_actions.item()

                new_obs, rewards, dones, infos = self.env.step(clipped_actions)
                new_obs = new_obs[None, :]

                states_airl.append(state)
                actions_airl.append(clipped_actions)
                rewards_airl.append(rewards)
                dones_airl.append([dones])
                values_airl.append(values)
                log_probs_airl.append(log_probs)
                infos_airl.append([infos])
                next_states_airl.append(new_obs)

                if dones:
                    state = self.env.reset()[None, :]
                else:
                    state = new_obs

            # self._last_obs: np.array with (1, 4)
            # new_obs: np.array with (1, 4)
            # clipped_actions: np.array with (1)
            # rewards: np.array with (1)
            # dones: np.array with (1), boolean
            # infos: list of info [{}]
            # values: tensor with (1, 1)
            # log_probs: tensor with (1)

            states_airl = np.array(states_airl)
            next_states_airl = np.array(next_states_airl)
            actions_airl = np.array(actions_airl)
            rewards_airl = np.array(rewards_airl)
            dones_airl = np.array(dones_airl)
            values_airl = th.tensor(values_airl)
            log_probs_airl = th.tensor(log_probs_airl)

            model.learn(total_timesteps=10000000, states_rollout=states_airl, next_states_rollout=next_states_airl,
                        actions_rollout=actions_airl, rewards_rollout=rewards_airl, dones_rollout=dones_airl, values_rollout=values_airl,
                        log_probs_rollout=log_probs_airl, infos_rollout=infos_airl)

            if airl_epoch % eval_interval == 0:
                print(f'epoch: {airl_epoch} | ', end='')
                self.evaluation()

    def evaluation(self, eval_epochs=10):
        ep_rewards = []
        ep_lengths = []
        for eval_epoch in range(eval_epochs):
            eval_state = self.eval_env.reset()
            eval_done = False
            ep_length = 0
            ep_reward = 0
            while not eval_done:
                eval_action, _ = self.model.predict(eval_state, deterministic=True)
                eval_next_state, eval_reward, eval_done, eval_info = self.eval_env.step(eval_action)
                ep_reward += eval_reward
                ep_length += 1
                eval_state = eval_next_state
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)

        print(
            f'mean length: {round(np.mean(ep_lengths), 1)} | mean reward: {round(np.mean(ep_rewards), 1)}')


if __name__ == '__main__':
<<<<<<< HEAD
    env_id = 'ma_gym:HuRoSorting-v0'
    # airl = Train(env_id, th.device('cpu'))
    # airl.airl(airl_epochs=500)
    rl = Train(env_id, th.device('cpu'))
    rl.forward_rl(total_timesteps=2500000)
=======
    env_id = 'LunarLander-v2'
    airl = Train(env_id, th.device('cpu'))
    airl.airl(airl_epochs=500)
    # rl = Train(env_id, th.device('cpu'))
    # rl.forward_rl(total_timesteps=250000)
>>>>>>> 2bc46b5c733b5422518582cf498f39a125f8a33a
