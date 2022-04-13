<<<<<<< HEAD
import os
=======
>>>>>>> 2bc46b5c733b5422518582cf498f39a125f8a33a
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

import gym
import argparse
import sys
import numpy as np

from stable_baselines3.common.utils import obs_as_tensor, safe_mean

from gail_airl_ppo.algo.ppo_sb import PPO_AIRL
from gail_airl_ppo.network.disc import AIRLDiscrim


class AIRL(object):
    def __init__(self, env_id, buffer_exp, device, seed, eval_interval=500,
                 gamma=0.995, n_steps=2048,
                 batch_size=64, lr_actor=3e-4, lr_disc=3e-4,
                 units_disc_r=(64, 64), units_disc_v=(64, 64),
                 epoch_actor=10, epoch_disc=10, clip_eps=0.2, gae_lambda=0.97,
                 ent_coef=0.0, max_grad_norm=0.5):
        self.env = gym.make(env_id)

        if self.env.observation_space.__class__.__name__ == 'Discrete':
            self.state_shape = (self.env.observation_space.n,)
        elif self.env.observation_space.__class__.__name__ == 'Box':
            self.state_shape = self.env.observation_space.shape
        else:
            raise ValueError('Cannot recognize env observation space ')

<<<<<<< HEAD
        # if self.env.action_space.__class__.__name__ == 'Discrete':
            # self.action_shape = (self.env.action_space.n,)
            # self.discrete_action = True
        # elif self.env.action_space.__class__.__name__ == 'Box':
            # self.action_shape = self.env.action_space.shape
            # self.discrete_action = False
        # else:
            # raise ValueError('Cannot recognize env action space')
=======
        if self.env.action_space.__class__.__name__ == 'Discrete':
            self.action_shape = (self.env.action_space.n,)
            self.discrete_action = True
        elif self.env.action_space.__class__.__name__ == 'Box':
            self.action_shape = self.env.action_space.shape
            self.discrete_action = False
        else:
            raise ValueError('Cannot recognize env action space')
>>>>>>> 2bc46b5c733b5422518582cf498f39a125f8a33a

        # Discriminator.
        self.disc = AIRLDiscrim(
            state_shape=self.state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        self.actor = PPO_AIRL("MlpPolicy", self.env, verbose=0, n_steps=n_steps, seed=seed, device=device, learning_rate=lr_actor,
                              n_epochs=epoch_actor, max_grad_norm=max_grad_norm, clip_range=clip_eps,
                              gae_lambda=gae_lambda, ent_coef=ent_coef, airl=True, _init_setup_model=True)

        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc
        self.eval_env = gym.make(env_id)
        self.n_steps = n_steps
        self.buffer_exp = buffer_exp
        self.learning_steps_disc = 0
        self.device = device
        self.eval_interval = eval_interval

        self.buffer = {
            'state': torch.zeros(size=(n_steps, 1, self.state_shape[0]), device=device),
            'action': torch.zeros(n_steps, device=device),
            'next_state': torch.zeros(size=(n_steps, 1, self.state_shape[0]), device=device),
            'reward': torch.zeros(n_steps, device=device),
            'done': torch.zeros(size=(n_steps, 1), device=device),
            'value': torch.zeros(n_steps, device=device),
            'log_prob': torch.zeros(n_steps, device=device),
            'info': [[{}]] * n_steps
        }
        self.buffer_size = n_steps
        self.buffer_p = 0
        self.buffer_record = 0

    def train(self, total_timesteps=100000):
        state = self.env.reset()[None, :]
        for airl_step in range(1, total_timesteps):
            with torch.no_grad():
                action, value, log_prob = self.actor.policy.forward(obs_as_tensor(state, self.device))
            actions = action.cpu().numpy()

            if isinstance(self.env.action_space, gym.spaces.Box):
                action = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            elif isinstance(self.env.action_space, gym.spaces.Discrete):
                action = action.item()

            next_state, reward, done, info = self.env.step(action)

            if self.env.action_space.__class__.__name__ == 'Discrete':
                action = [action]

            self.buffer_add(state, action, next_state, reward, done, value, log_prob, info)

            if done:
                state = self.env.reset()[None, :]
            else:
                state = next_state[None, :]

            if airl_step % self.n_steps == 0:
                # assert self.buffer_p == 0
                self.update()

            if airl_step % self.eval_interval == 0:
                print(f'Timesteps: {airl_step} | ', end='')
                self.evaluate()

    def evaluate(self, eval_epochs=10, render=False):
        ep_rewards = []
        ep_lengths = []
        render_first = True
        for eval_epoch in range(eval_epochs):
            eval_state = self.eval_env.reset()
            eval_done = False
            ep_length = 0
            ep_reward = 0
            while not eval_done:
                if render_first and render:
                    self.eval_env.render()
                eval_action, _ = self.actor.predict(eval_state, deterministic=True)
                eval_next_state, eval_reward, eval_done, eval_info = self.eval_env.step(eval_action)
                ep_reward += eval_reward
                ep_length += 1
                eval_state = eval_next_state
            render_first = False
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)

        print(
            f'mean length: {round(np.mean(ep_lengths), 1)} | mean reward: {round(np.mean(ep_rewards), 1)}')

    def update(self):
        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1
            states_policy, _, next_states_policy, _, dones_policy, _, log_probs_policy = self.buffer_sample(self.buffer, expert=False)
            states_exp, actions_exp, next_states_exp, _, dones_exp = self.buffer_sample(self.buffer_exp, expert=True)
            with torch.no_grad():
                _, log_probs_exp, _ = self.actor.policy.evaluate_actions(states_exp, actions_exp)

            self.update_disc(
                states_policy, dones_policy, log_probs_policy, next_states_policy, states_exp,
                dones_exp, log_probs_exp, next_states_exp
            )

        states, actions, next_states, _, dones, values, log_probs, infos = self.buffer_get()

        # Calculate rewards.
        rewards = self.disc.calculate_reward(
            states.squeeze(), dones, log_probs[:, None], next_states.squeeze()).squeeze()

        self.actor.learn(total_timesteps=100000000, states_rollout=states.cpu().numpy(), next_states_rollout=next_states.cpu().numpy(),
                         actions_rollout=actions.cpu().numpy(), rewards_rollout=rewards.cpu().numpy(), dones_rollout=dones.cpu().numpy(),
                         values_rollout=values, log_probs_rollout=log_probs, infos_rollout=infos)

    def update_disc(self, states, dones, log_probs, next_states,
                    states_exp, dones_exp, log_probs_exp,
                    next_states_exp):
        logits_pi = self.disc(states.squeeze(), dones, log_probs[:, None], next_states.squeeze())
        logits_exp = self.disc(
            states_exp.squeeze(), dones_exp, log_probs_exp[:, None], next_states_exp.squeeze())

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

    def buffer_add(self, state, action, next_state, reward, done, value, log_prob, info):
        self.buffer['state'][self.buffer_p] = torch.from_numpy(state).clone().float()
        self.buffer['action'][self.buffer_p] = torch.tensor(action).clone()
        self.buffer['next_state'][self.buffer_p] = torch.from_numpy(next_state).clone().float()
        self.buffer['reward'][self.buffer_p] = reward
        self.buffer['done'][self.buffer_p] = torch.tensor([int(done)]).float()
        self.buffer['value'][self.buffer_p] = value
        self.buffer['log_prob'][self.buffer_p] = log_prob
        self.buffer['info'][self.buffer_p] = [info]
        self.buffer_p += 1
        self.buffer_p %= self.buffer_size
        self.buffer_record += 1

    def buffer_sample(self, buffer, expert=False):
        if not expert:
            current_buffer_size = min(self.buffer_record, self.buffer_size)
            idx = torch.randperm(current_buffer_size)[:self.batch_size]
            return buffer['state'][idx], buffer['action'][idx], buffer['next_state'][idx], buffer['reward'][idx], \
                   buffer['done'][idx], buffer['value'][idx], buffer['log_prob'][idx]
        else:
            current_buffer_size = len(buffer['state'])
            idx = torch.randperm(current_buffer_size)[:self.batch_size]
            return buffer['state'][idx], buffer['action'][idx], buffer['next_state'][idx], buffer['reward'][idx], \
                   buffer['done'][idx]

    def buffer_get(self):
        current_buffer_size = min(self.buffer_record, self.buffer_size)
        return self.buffer['state'][:current_buffer_size], self.buffer['action'][:current_buffer_size], self.buffer['next_state'][:current_buffer_size], self.buffer['reward'][:current_buffer_size], \
               self.buffer['done'][:current_buffer_size], self.buffer['value'][:current_buffer_size], self.buffer['log_prob'][:current_buffer_size], self.buffer['info'][:current_buffer_size]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # p.add_argument('--buffer', type=str, required=True)
    # p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--num_steps', type=int, default=10 ** 7)
    p.add_argument('--eval_interval', type=int, default=4096)
    p.add_argument('--env_id', type=str, default='Hopper-v3')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()

    env_id = args.env_id
    device = 'cuda:0' if args.cuda else 'cpu'
    print(f'Using {device}')
<<<<<<< HEAD
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    buffer_exp = torch.load(script_dir + f'/buffers/ppo_sb/{env_id.split("-")[0].lower()}.pt')
=======

    buffer_exp = torch.load(f'../../buffers/ppo_sb/{env_id.split("-")[0].lower()}.pt')
>>>>>>> 2bc46b5c733b5422518582cf498f39a125f8a33a
    airl = AIRL(env_id=env_id, buffer_exp=buffer_exp, device=device, seed=args.seed, eval_interval=args.eval_interval)
    airl.train(args.num_steps)
