# from msilib.schema import File
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

import gym
import argparse
import sys
import os
import numpy as np

from datetime import datetime
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
sys.path.append(os.getcwd()+f'/gail-airl-ppo/')
from gail_airl_ppo.algo.ppo_sb_dec import PPO_Dec, ActorCriticPolicy_Dec
from gail_airl_ppo.network.disc import AIRLDiscrimAction


class AIRL(object):
    def __init__(self, env_id, buffer_r_exp, buffer_h_exp, device, seed, eval_interval=500,
                 gamma=0.95, n_steps=2048,
                 batch_size=128, lr_actor=3e-4, lr_disc=3e-4,
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

        # Discriminator.
        self.disc = AIRLDiscrimAction(
            state_shape=self.state_shape,
            gamma=gamma,
            num_of_actions=6 * 2,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        self.actor_r = PPO_Dec(ActorCriticPolicy_Dec, self.env, verbose=0, n_steps=n_steps, seed=seed, device=device, learning_rate=lr_actor,
                              n_epochs=epoch_actor, max_grad_norm=max_grad_norm, clip_range=clip_eps,
                              gae_lambda=gae_lambda, ent_coef=ent_coef, airl=True, _init_setup_model=True)

        self.actor_h = PPO_Dec(ActorCriticPolicy_Dec, self.env, verbose=0, n_steps=n_steps, seed=seed, device=device, learning_rate=lr_actor,
                              n_epochs=epoch_actor, max_grad_norm=max_grad_norm, clip_range=clip_eps,
                              gae_lambda=gae_lambda, ent_coef=ent_coef, airl=True, _init_setup_model=True)

        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc
        self.eval_env = gym.make(env_id)
        self.test_env = gym.make(env_id)
        self.n_steps = n_steps
        self.learning_steps_disc = 0
        self.device = device
        self.eval_interval = eval_interval

        self.buffer_r_exp = buffer_r_exp
        self.buffer_h_exp = buffer_h_exp


        self.buffer_r = {
            'state': torch.zeros(size=(n_steps, self.state_shape[0]), device=device),
            'action': torch.zeros(n_steps, device=device),
            'next_state': torch.zeros(size=(n_steps, self.state_shape[0]), device=device),
            'reward': torch.zeros(n_steps, device=device),
            'done': torch.zeros(size=(n_steps, 1), device=device),
            'value': torch.zeros(n_steps, device=device),
            'log_prob': torch.zeros(n_steps, device=device),
            'info': [[{}]] * n_steps,
            'p': 0,
            'record': 0
        }

        self.buffer_h = {
            'state': torch.zeros(size=(n_steps, self.state_shape[0]), device=device),
            'action': torch.zeros(n_steps, device=device),
            'next_state': torch.zeros(size=(n_steps, self.state_shape[0]), device=device),
            'reward': torch.zeros(n_steps, device=device),
            'done': torch.zeros(size=(n_steps, 1), device=device),
            'value': torch.zeros(n_steps, device=device),
            'log_prob': torch.zeros(n_steps, device=device),
            'info': [[{}]] * n_steps,
            'p': 0,
            'record': 0
        }

        self.buffer_size = n_steps

        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y-%H-%M")
        # self.path = f'models/{timestamp}'
        self.path = os.getcwd()+f'/gail-airl-ppo/gail_airl_ppo/algo/models_airl/{timestamp}'

        self.best_reward = -100
        try:
            os.mkdir(self.path)
            # print("New airl model folder created at: ", self.path)
        except FileExistsError:
            print("FileExistsError Exception!")

    def train(self, total_timesteps=100000):
        state = self.env.reset()
        state_robot = state[:11].copy()
        state_human = state[11:].copy()
        state_robot_input = np.concatenate([state_robot.copy(), state_human.copy()])
        state_human_input = np.concatenate([state_human.copy(), state_robot.copy()])

        for airl_step in range(1, total_timesteps):
            with torch.no_grad():
                # action, value, log_prob = self.actor.policy.forward(obs_as_tensor(state, self.device))
                actions_r, values_r, log_probs_r = self.actor_r.policy.forward(obs_as_tensor(state_robot_input, device=self.device).float())
                actions_h, values_h, log_probs_h = self.actor_h.policy.forward(obs_as_tensor(state_human_input, device=self.device).float())
            action_r = actions_r.item()
            action_h = actions_h.item()
            actions = [action_r, action_h]

            next_state, reward, done, info = self.env.step(actions)

            n_state_robot = next_state[:11].copy()
            n_state_human = next_state[11:].copy()
            n_state_robot_input = np.concatenate([n_state_robot.copy(), n_state_human.copy()])
            n_state_human_input = np.concatenate([n_state_human.copy(), n_state_robot.copy()])

            # if self.env.action_space.__class__.__name__ == 'Discrete':
            #     action = [action]

            self.buffer_add(self.buffer_r, state_robot_input, action_r, n_state_robot_input, reward, done, values_r, log_probs_r, info)
            self.buffer_add(self.buffer_h, state_human_input, action_h, n_state_human_input, reward, done, values_h, log_probs_h, info)

            if done:
                state = self.env.reset()
            else:
                state = next_state

            state_robot = state[:11].copy()
            state_human = state[11:].copy()
            state_robot_input = np.concatenate([state_robot.copy(), state_human.copy()])
            state_human_input = np.concatenate([state_human.copy(), state_robot.copy()])

            if airl_step % self.n_steps == 0:
                # assert self.buffer_p == 0
                self.update()

            if airl_step % self.eval_interval == 0:
                print(f'Timesteps: {airl_step} | ', end='')
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    torch.save(self.disc.state_dict(), f'{self.path}/disc_{airl_step}_{int(eval_reward)}.pt')
                    self.actor_r.save(f'{self.path}/actor_r_{airl_step}_{int(eval_reward)}')
                    self.actor_h.save(f'{self.path}/actor_h_{airl_step}_{int(eval_reward)}')
                    self.best_reward = eval_reward

    def evaluate(self, eval_epochs=10, render=False):
        ep_rewards = []
        ep_lengths = []
        render_first = True
        for eval_epoch in range(eval_epochs):
            eval_state = self.eval_env.reset()
            state_robot = eval_state[:11].copy()
            state_human = eval_state[11:].copy()
            state_robot_input = np.concatenate([state_robot.copy(), state_human.copy()])
            state_human_input = np.concatenate([state_human.copy(), state_robot.copy()])
            eval_done = False
            ep_length = 0
            ep_reward = 0
            while not eval_done:
                if render_first and render:
                    self.eval_env.render()
                eval_action_r, _ = self.actor_r.predict(state_robot_input, deterministic=True)
                eval_action_h, _ = self.actor_h.predict(state_human_input, deterministic=True)
                eval_next_state, eval_reward, eval_done, eval_info = self.eval_env.step([eval_action_r, eval_action_h], verbose=0)
                ep_reward += eval_reward
                ep_length += 1
                eval_state = eval_next_state
                state_robot = eval_state[:11]
                state_human = eval_state[11:]
                state_robot_input = np.concatenate([state_robot, state_human])
                state_human_input = np.concatenate([state_human, state_robot])
            render_first = False
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)

        print(
            f'mean length: {round(np.mean(ep_lengths), 1)} | mean reward: {round(np.mean(ep_rewards), 1)}')
        return np.mean(ep_rewards)
    
    def test(self, path, load_best=True, disc=None, actor_r=None, actor_h=None, test_epochs=1, render=False):
        ep_rewards = []
        ep_lengths = []
        render_first = True
        if load_best:
            files = os.listdir(path)
            best_reward = -100
            for file in files:
                # print(f"Filename: {file}")
                reward = int(file.split('_')[-1].split('.')[0])
                step = int(file.split('_')[-2])
                # print(f"Reward: {reward}, best reward: {best_reward}")
                if reward > best_reward:
                    best_reward = reward
                    actor_r = f'actor_r_{step}_{reward}'
                    actor_h = f'actor_h_{step}_{reward}'
                    disc = f'disc_{step}_{reward}.pt'
                    # print("Weights chosen are: ", actor_r)

        self.disc.load_state_dict(torch.load(f'{path}/{disc}'))
        self.actor_r.set_parameters(f'{path}/{actor_r}',  device=self.device)
        self.actor_h.set_parameters(f'{path}/{actor_h}', device=self.device)
        for test_epoch in range(test_epochs):
            test_state = self.test_env.reset(fixed_init = False)
            state_robot = test_state[:11].copy()
            state_human = test_state[11:].copy()
            state_robot_input = np.concatenate([state_robot.copy(), state_human.copy()])
            state_human_input = np.concatenate([state_human.copy(), state_robot.copy()])
            test_done = False
            ep_length = 0
            ep_reward = 0
            while not test_done:
                if render_first and render:
                    self.test_env.render()
                state_robot_input = torch.tensor(state_robot_input)[None, :].float()
                state_human_input = torch.tensor(state_human_input)[None, :].float()
                test_action_r, _, test_action_r_log_prob = self.actor_r.policy.forward(state_robot_input, deterministic=True)
                test_action_h, _, test_action_h_log_prob = self.actor_h.policy.forward(state_human_input, deterministic=True)
                test_next_state, test_reward, test_done, test_info = self.test_env.step([test_action_r, test_action_h], verbose=1)
                
                action_r_onehot = torch.nn.functional.one_hot(test_action_r.long(), num_classes=6).float()
                action_h_onehot = torch.nn.functional.one_hot(test_action_h.long(), num_classes=6).float()
                global_test_actions = torch.cat((action_r_onehot, action_h_onehot), dim=1)
                
                log_probs = test_action_r_log_prob + test_action_h_log_prob
                disc_reward = self.disc.calculate_reward(state_robot_input, torch.tensor([int(test_done)])[None, :].float(), log_probs[:, None], torch.tensor(test_next_state)[None, :].float(), global_test_actions).squeeze()
                print(f'original reward: {test_reward} | disc reward: {round(disc_reward.item(), 3)}')
                ep_reward += test_reward
                ep_length += 1
                test_state = test_next_state
                state_robot = test_state[:11]
                state_human = test_state[11:]
                state_robot_input = np.concatenate([state_robot, state_human])
                state_human_input = np.concatenate([state_human, state_robot])
            render_first = False
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)

        print(
            f'mean length: {round(np.mean(ep_lengths), 1)} | mean reward: {round(np.mean(ep_rewards), 1)}')
        return np.mean(ep_rewards)

    def update(self):
        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1
            states_r_policy, action_r_policy, next_states_r_policy, _, dones_r_policy, _, log_probs_r_policy = self.buffer_sample(self.buffer_r, expert=False)
            states_h_policy, action_h_policy, next_states_h_policy, _, dones_h_policy, _, log_probs_h_policy = self.buffer_sample(self.buffer_h, expert=False)
            states_r_exp, action_r_exp, next_states_r_exp, _, dones_r_exp = self.buffer_sample(self.buffer_r_exp, expert=True)
            states_h_exp, action_h_exp, next_states_h_exp, _, dones_h_exp = self.buffer_sample(self.buffer_h_exp, expert=True)
            with torch.no_grad():
                _, log_probs_r_exp, _ = self.actor_r.policy.evaluate_actions(states_r_exp, action_r_exp)
                _, log_probs_h_exp, _ = self.actor_h.policy.evaluate_actions(states_h_exp, action_h_exp)
            action_r_policy_onehot = torch.nn.functional.one_hot(action_r_policy.long(), num_classes=6).float()
            action_h_policy_onehot = torch.nn.functional.one_hot(action_h_policy.long(), num_classes=6).float()
            global_actions_policy = torch.cat((action_r_policy_onehot, action_h_policy_onehot), dim=1)

            action_r_exp_onehot = torch.nn.functional.one_hot(action_r_exp.long(), num_classes=6).float()
            action_h_exp_onehot = torch.nn.functional.one_hot(action_h_exp.long(), num_classes=6).float()
            global_actions_exp = torch.cat((action_r_exp_onehot, action_h_exp_onehot), dim=1)

            states_policy = states_r_policy.clone()
            dones_policy = dones_r_policy.clone()
            log_probs_policy = log_probs_r_policy + log_probs_h_policy
            next_states_policy = next_states_r_policy.clone()

            states_exp = states_r_exp.clone()
            dones_exp = dones_r_exp.clone()
            log_probs_exp = log_probs_r_exp + log_probs_h_exp
            next_states_exp = next_states_r_exp.clone()

            self.update_disc(
                states_policy, global_actions_policy, dones_policy, log_probs_policy, next_states_policy, 
                states_exp, global_actions_exp, dones_exp, log_probs_exp, next_states_exp
            )

        states_r, actions_r, next_states_r, _, dones_r, values_r, log_probs_r, infos_r = self.buffer_get(self.buffer_r)
        states_h, actions_h, next_states_h, _, dones_h, values_h, log_probs_h, infos_h = self.buffer_get(self.buffer_h)

        states = states_r.clone()

        action_r_onehot = torch.nn.functional.one_hot(actions_r.long(), num_classes=6).float()
        action_h_onehot = torch.nn.functional.one_hot(actions_h.long(), num_classes=6).float()
        global_actions= torch.cat((action_r_onehot, action_h_onehot), dim=1)

        dones = dones_r.clone()
        log_probs = log_probs_r + log_probs_h
        next_states = next_states_r.clone()
        # Calculate rewards.
        rewards = self.disc.calculate_reward(
            states, dones, log_probs[:, None], next_states, global_actions).squeeze()

        self.actor_r.learn(total_timesteps=1000000, states_rollout=states_r.cpu().numpy(), next_states_rollout=next_states_r.cpu().numpy(),
                         actions_rollout=actions_r.cpu().numpy(), rewards_rollout=rewards.cpu().numpy(), dones_rollout=dones_r.cpu().numpy(),
                         values_rollout=values_r, log_probs_rollout=log_probs_r, infos_rollout=infos_r)
        
        self.actor_h.learn(total_timesteps=1000000, states_rollout=states_h.cpu().numpy(), next_states_rollout=next_states_h.cpu().numpy(),
                         actions_rollout=actions_h.cpu().numpy(), rewards_rollout=rewards.cpu().numpy(), dones_rollout=dones_h.cpu().numpy(),
                         values_rollout=values_h, log_probs_rollout=log_probs_h, infos_rollout=infos_h)

    def update_disc(self, states, actions, dones, log_probs, next_states,
                    states_exp, actions_exp, dones_exp, log_probs_exp,
                    next_states_exp):
        logits_pi = self.disc(states, dones, log_probs[:, None], next_states, actions)
        logits_exp = self.disc(
            states_exp, dones_exp, log_probs_exp[:, None], next_states_exp, actions_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

    def buffer_add(self, buffer, state, action, next_state, reward, done, value, log_prob, info):
        p = buffer['p']
        buffer['state'][p] = torch.from_numpy(state).clone().float()
        buffer['action'][p] = torch.tensor(action).clone()
        buffer['next_state'][p] = torch.from_numpy(next_state).clone().float()
        buffer['reward'][p] = reward
        buffer['done'][p] = torch.tensor([int(done)]).float()
        buffer['value'][p] = value
        buffer['log_prob'][p] = log_prob
        buffer['info'][p] = [info]
        buffer['p'] += 1
        buffer['p'] %= self.buffer_size
        buffer['record'] += 1

    def buffer_sample(self, buffer, expert=False):
        if not expert:
            current_buffer_size = min(buffer['record'], self.buffer_size)
            idx = torch.randperm(current_buffer_size)[:self.batch_size]
            return buffer['state'][idx], buffer['action'][idx], buffer['next_state'][idx], buffer['reward'][idx], \
                   buffer['done'][idx], buffer['value'][idx], buffer['log_prob'][idx]
        else:
            current_buffer_size = len(buffer['state'])
            idx = torch.randperm(current_buffer_size)[:self.batch_size]
            return buffer['state'][idx], buffer['action'][idx], buffer['next_state'][idx], buffer['reward'][idx], \
                   buffer['done'][idx]

    def buffer_get(self, buffer):
        current_buffer_size = min(buffer['record'], self.buffer_size)
        return buffer['state'][:current_buffer_size], buffer['action'][:current_buffer_size], buffer['next_state'][:current_buffer_size], buffer['reward'][:current_buffer_size], \
               buffer['done'][:current_buffer_size], buffer['value'][:current_buffer_size], buffer['log_prob'][:current_buffer_size], buffer['info'][:current_buffer_size]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # p.add_argument('--buffer', type=str, required=True)
    # p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--num_steps', type=int, default=10 ** 7)
    p.add_argument('--eval_interval', type=int, default=4096)
    p.add_argument('--env_id', type=str, default='ma_gym:HuRoSorting-v0')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()

    env_id = args.env_id
    device = 'cuda:0' if args.cuda else 'cpu'
    print(f'Using {device}')

    buffer_exp = torch.load(os.getcwd()+f'/gail-airl-ppo/gail_airl_ppo/algo/buffers/ma_gym:hurosorting.pt')
    buffer_r_exp = {
        'state': buffer_exp['robot_state'].clone(),
        'action': buffer_exp['robot_action'].clone(),
        'next_state': buffer_exp['robot_next_state'].clone(),
        'done': buffer_exp['done'].clone(),
        'reward': buffer_exp['reward'].clone()
    }

    buffer_h_exp = {
        'state': buffer_exp['human_state'].clone(),
        'action': buffer_exp['human_action'].clone(),
        'next_state': buffer_exp['human_next_state'].clone(),
        'done': buffer_exp['done'].clone(),
        'reward': buffer_exp['reward'].clone()
    }

    airl = AIRL(env_id=env_id, buffer_r_exp=buffer_r_exp, buffer_h_exp=buffer_h_exp, device=device, seed=args.seed, eval_interval=args.eval_interval)
    airl.train(args.num_steps)
    # airl.test(path=os.getcwd()+f'/gail-airl-ppo/gail_airl_ppo/algo/models_airl/04-12-2022-11-43/', load_best=True)