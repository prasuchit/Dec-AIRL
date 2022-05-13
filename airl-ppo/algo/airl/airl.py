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
# from stable_baselines3.common.running_mean_std import RunningMeanStd
sys.path.append(os.getcwd()+f'/airl-ppo/')
from algo.ppo.ActorCritic import *
from algo.ppo.ppo import *
from algo.airl.Disc import AIRLDiscrimMultiAgent


class AIRL(object):
    def __init__(self, env_id, buffers_exp, seed, eval_interval=500,
                 gamma=0.95, n_steps=2048, device='cpu',
                 batch_size=128, lr_actor=3e-4, lr_disc=3e-4,
                 units_disc_r=(64, 64), units_disc_v=(64, 64),
                 epoch_actor=10, epoch_disc=10, clip_eps=0.2, gae_lambda=0.97,
                 ent_coef=0.0, max_grad_norm=0.5):
        self.env = gym.make(env_id)
        self.env.seed(seed)
        self.seed = seed
        self.n_agents = self.env.n_agents
        self.device = device

        # if self.env.observation_space.__class__.__name__ == 'Discrete':
        #     self.state_shape = (self.env.observation_space.n,)
        # elif self.env.observation_space.__class__.__name__ == 'Box':
        #     self.state_shape = self.env.observation_space.shape
        # else:
        #     raise ValueError('Cannot recognize env observation space ')

        assert self.env.observation_space.__class__.__name__ == 'MultiAgentObservationSpace', f'Unsupported observation space: {self.env.observation_space.__class__.__name__}'

        # Discriminator.
        self.disc = AIRLDiscrimMultiAgent(
            obs_space=self.env.observation_space,
            gamma=gamma,
            action_space=self.env.action_space,
            n_agents=self.n_agents,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        self.actors = [PPO_Dec(ActorCriticPolicy_Dec, self.env, agent_id=agent_id, verbose=1, custom_rollout=True, device=self.device, seed=self.seed) for agent_id in range(self.n_agents)]

        # print("Load existing", load_existing)

        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc
        self.eval_env = gym.make(env_id)
        self.eval_env.seed(self.seed)
        self.test_env = gym.make(env_id)
        self.test_env.seed(self.seed)
        self.n_steps = n_steps
        self.learning_steps_disc = 0
        self.device = device
        self.eval_interval = eval_interval

        self.buffers_exp = buffers_exp

        self.global_observation_shape = self.env.observation_space[0].shape[0] * self.n_agents
        self.local_observation_shape = self.env.observation_space[0].shape[0]
        self.action_shape = self.env.action_space[0].n

        self.path = None
        self.best_reward = -100
        # self.load_models(load_existing, trainpath)

        self.buffers_policy = {
            'state': torch.zeros(size=(n_steps, self.n_agents, self.local_observation_shape), device=device),
            'action': torch.zeros(size=(n_steps, self.n_agents), device=device),
            'next_state': torch.zeros(size=(n_steps, self.n_agents, self.local_observation_shape), device=device),
            'reward': torch.zeros(n_steps, device=device),
            'done': torch.zeros(size=(n_steps, 1), device=device),
            'value': torch.zeros(size=(n_steps, self.n_agents), device=device),
            'log_prob': torch.zeros(size=(n_steps, self.n_agents), device=device),
            'info': [[{}]] * n_steps,
            'p': 0,
            'record': 0
        }

        self.buffer_size = n_steps

    def update(self, epoch_ratio):
        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            states_policy, actions_policy, next_states_policy, _, dones_policy, _, log_probs_policy = self.buffer_sample(self.buffers_policy, expert=False)

            states_exp, actions_exp, next_states_exp, _, dones_exp = self.buffer_sample(self.buffers_exp, expert=True)
            
            log_probs_exp = []
            actions_policy_onehot = []
            actions_exp_onehot = []

            with torch.no_grad():
                for agent_id in range(self.n_agents):
                    _, log_prob_exp, _ = self.actors[agent_id].policy.evaluate_actions(states_exp, actions_exp[:, agent_id])
                    action_policy_onehot = torch.nn.functional.one_hot(actions_policy[:, agent_id].long(), num_classes=self.action_shape).float()
                    action_exp_onehot = torch.nn.functional.one_hot(actions_exp[:, agent_id].long(), num_classes=self.action_shape).float()

                    log_probs_exp.append(log_prob_exp)
                    actions_policy_onehot.append(action_policy_onehot)
                    actions_exp_onehot.append(action_exp_onehot)

            log_probs_exp = torch.stack(log_probs_exp, dim=1).float()
            global_actions_policy = torch.cat(actions_policy_onehot, dim=1)
            global_actions_exp = torch.cat(actions_exp_onehot, dim=1)

            log_probs_policy = log_probs_policy.sum(dim=1)
            log_probs_exp = log_probs_exp.sum(dim=1)

            self.update_disc(
                states_policy, global_actions_policy, dones_policy, log_probs_policy, next_states_policy, 
                states_exp, global_actions_exp, dones_exp, log_probs_exp, next_states_exp, epoch_ratio
            )

        states, actions, next_states, _, dones, values, log_probs, infos = self.buffer_get(self.buffers_policy)
        
        actions_onehot = []
        for agent_id in range(self.n_agents):
            action_onehot = torch.nn.functional.one_hot(actions[:, agent_id].long(), num_classes=self.action_shape).float()

            actions_onehot.append(action_onehot)
        global_actions = torch.cat(actions_onehot, dim=1)
        actions_onehot = torch.stack(actions_onehot, dim=1)

        global_log_probs = log_probs.sum(dim=1)[:, None]

        # Calculate rewards.
        rewards = self.disc.calculate_reward(
            states, dones, global_log_probs, next_states, global_actions).squeeze()
        
        # rewards = normalize(rewards)

        for agent_id in range(self.n_agents):
            self.actors[agent_id].learn(total_timesteps=1000000, states_rollout=states.cpu().numpy(), next_states_rollout=next_states.cpu().numpy(),
                         actions_rollout=actions[:, agent_id].cpu().numpy(), rewards_rollout=rewards.cpu().numpy(), dones_rollout=dones.squeeze().cpu().numpy(),
                         values_rollout=values[:, agent_id], log_probs_rollout=log_probs[:, agent_id], infos_rollout=infos)

    def update_disc(self, states, actions, dones, log_probs, next_states,
                    states_exp, actions_exp, dones_exp, log_probs_exp,
                    next_states_exp, epoch_ratio):
        logits_pi = self.disc(states, dones, log_probs[:, None], next_states, actions)
        logits_exp = self.disc(
            states_exp, dones_exp, log_probs_exp[:, None], next_states_exp, actions_exp)

        # threshold = 0.7
        # traj_label[traj_label==-1] = min(0, (threshold - epoch_ratio) / (1 - threshold))

        loss_pi = -F.logsigmoid(-logits_pi)
        loss_exp = -F.logsigmoid(logits_exp)
        loss_disc = (loss_pi + loss_exp).mean()

        # print(f"Logits learner: {loss_pi.mean().item()}, Logits expert: {loss_exp.mean().item()}")
        # print(f"-F.logsigmoid(-logits_pi): {loss_pi.mean().item()}, -F.logsigmoid(logits_exp): {loss_exp.mean().item()}")
        # print(f"Disc loss: {((loss_pi + loss_exp) * traj_label).mean().item()}\n")
        # print(f"-F.logsigmoid(logits_pi): {-F.logsigmoid(logits_pi).mean().item()}, -F.logsigmoid(-logits_exp): {-F.logsigmoid(-logits_exp).mean().item()}")
        # print(f"Disc loss: {((-F.logsigmoid(logits_pi) + -F.logsigmoid(-logits_exp)) * traj_label).mean().item()}\n")
       
        # if not failure_traj:
        #     # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        #     loss_pi = -F.logsigmoid(-logits_pi).mean()
        #     loss_exp = -F.logsigmoid(logits_exp).mean()
        #     loss_disc = loss_pi + loss_exp
        # else: 
        #     # Discriminator is to minimize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        #     loss_pi = F.logsigmoid(-logits_pi).mean()
        #     loss_exp = F.logsigmoid(logits_exp).mean()
        #     loss_disc = loss_exp \
        #                 + loss_pi

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

    def buffer_add(self, buffer, state, action, next_state, reward, done, value, log_prob, info):
        p = buffer['p']
        buffer['state'][p] = torch.tensor(state).clone().float()
        buffer['action'][p] = torch.tensor(action).clone()
        buffer['next_state'][p] = torch.tensor(next_state).clone().float()
        buffer['reward'][p] = reward
        buffer['done'][p] = torch.tensor([int(done)]).float()
        buffer['value'][p] = torch.tensor(value).float()
        buffer['log_prob'][p] = torch.tensor(log_prob).float()
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

    def train(self, total_timesteps=100000):
        obs = self.env.reset()

        for airl_step in range(1, total_timesteps):
            with th.no_grad():
                actions, values, log_probs = [], [], []
                for agent_id in range(self.n_agents):
                    action, value, log_prob = self.actors[agent_id].policy.forward(obs_as_tensor(obs, device=self.device).float())
                    actions.append(action.item())
                    values.append(value)
                    log_probs.append(log_prob)

            new_obs, rewards, dones, infos = self.env.step(actions)
            rewards = sum(rewards)
            dones = all(dones)

            self.buffer_add(self.buffers_policy, obs, actions, new_obs, rewards, dones, values, log_probs, infos)

            if dones:
                obs = self.env.reset()
            else:
                obs = new_obs

            if airl_step % self.n_steps == 0:
                # assert self.buffer_p == 0
                self.update(airl_step / total_timesteps)

            if airl_step % self.eval_interval == 0:
                print(f'Timesteps: {airl_step} | ', end='')
                eval_reward = self.evaluate(eval_epochs=10)

    def evaluate(self, eval_epochs=10):
        reward_stats = []
        length_stats = []

        reward = 0
        length = 0

        for epoch in range(eval_epochs):
            obs = self.eval_env.reset()
            dones = False

            while not dones:
                with th.no_grad():
                    actions, values, log_probs = [], [], []
                    for agent_id in range(self.n_agents):
                        action, value, log_prob = self.actors[agent_id].policy.forward(obs_as_tensor(obs, device=self.device), deterministic=True)
                        actions.append(action.item())
                        values.append(value)
                        log_probs.append(log_prob)

                new_obs, rewards, dones, infos = self.eval_env.step(actions)
                rewards = sum(rewards)
                dones = all(dones)

                obs = new_obs
                
                reward += rewards
                length += 1

            reward_stats.append(reward)
            length_stats.append(length)

            reward = 0
            length = 0
        
        eval_rewards = np.mean(reward_stats)
        print(f'Avg Length: {round(np.mean(length_stats), 2)} | Avg Reward: {round(eval_rewards, 2)}')
        return eval_rewards


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # p.add_argument('--buffer', type=str, required=True)
    # p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--num_steps', type=int, default=10 ** 7)
    p.add_argument('--eval_interval', type=int, default=4096)
    p.add_argument('--env_id', type=str, default='ma_gym:HuRoSorting-v0')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--failure_traj', action='store_true')
    p.add_argument('--load_existing', action='store_true')
    p.add_argument('--test', action='store_true')
    args = p.parse_args()

    env_id = args.env_id
    device = 'cuda:0' if args.cuda else 'cpu'
    # print(f'args.load_existing {args.load_existing}')

    # if not args.failure_traj:
    #     print("Loading successful trajectories")
    #     buffer_exp = torch.load(os.getcwd()+f'/gail-airl-ppo/gail_airl_ppo/algo/buffers/ma_gym:hurosorting.pt')
    # else:
    #     print("Loading failure trajectories") 
    #     buffer_exp = torch.load(os.getcwd()+f'/gail-airl-ppo/gail_airl_ppo/algo/buffers/ma_gym:hurosorting_failed.pt')

    buffer_exp_success = torch.load(os.getcwd()+f'/gail-airl-ppo/gail_airl_ppo/algo/buffers/ma_gym:hurosorting.pt')
    buffer_exp_success['label'] = torch.ones(len(buffer_exp_success['robot_state']))

    buffer_exp_failure = torch.load(os.getcwd()+f'/gail-airl-ppo/gail_airl_ppo/algo/buffers/ma_gym:hurosorting_failed.pt')
    buffer_exp_failure['label'] = -torch.ones(len(buffer_exp_failure['robot_state']))

    ################################## BOTH SUCCESS AND FAILURE TRAJS ############################################
    # buffer_r_exp = {
    #     'state': torch.cat((buffer_exp_success['robot_state'].clone(), buffer_exp_failure['robot_state'].clone()), 0),
    #     'action': torch.cat((buffer_exp_success['robot_action'].clone(), buffer_exp_failure['robot_action'].clone()), 0),
    #     'next_state': torch.cat((buffer_exp_success['robot_next_state'].clone(), buffer_exp_failure['robot_next_state'].clone()), 0),
    #     'done': torch.cat((buffer_exp_success['done'].clone(), buffer_exp_failure['done'].clone()), 0),
    #     'reward': torch.cat((buffer_exp_success['reward'].clone(), buffer_exp_failure['reward'].clone()), 0),
    #     'label': torch.cat((buffer_exp_success['label'].clone(), buffer_exp_failure['label'].clone()), 0)
    # }

    # buffer_h_exp = {
    #     'state': torch.cat((buffer_exp_success['human_state'].clone(), buffer_exp_failure['human_state'].clone()), 0),
    #     'action': torch.cat((buffer_exp_success['human_action'].clone(), buffer_exp_failure['human_action'].clone()), 0),
    #     'next_state': torch.cat((buffer_exp_success['human_next_state'].clone(), buffer_exp_failure['human_next_state'].clone()), 0),
    #     'done': torch.cat((buffer_exp_success['done'].clone(), buffer_exp_failure['done'].clone()), 0),
    #     'reward': torch.cat((buffer_exp_success['reward'].clone(), buffer_exp_failure['reward'].clone()), 0),
    #     'label': torch.cat((buffer_exp_success['label'].clone(), buffer_exp_failure['label'].clone()), 0)
    # }

    ###################################### ONLY SUCCESS TRAJS ################################################
    buffer_r_exp = {
        'state': buffer_exp_success['robot_state'].clone(),
        'action': buffer_exp_success['robot_action'].clone(),
        'next_state': buffer_exp_success['robot_next_state'].clone(),
        'done': buffer_exp_success['done'].clone(),
        'reward': buffer_exp_success['reward'].clone(),
        'label': buffer_exp_success['label'].clone()
    }

    buffer_h_exp = {
        'state': buffer_exp_success['human_state'].clone(),
        'action': buffer_exp_success['human_action'].clone(),
        'next_state': buffer_exp_success['human_next_state'].clone(),
        'done': buffer_exp_success['done'].clone(),
        'reward': buffer_exp_success['reward'].clone(),
        'label': buffer_exp_success['label'].clone()
    }

    ###################################### ONLY FAILURE TRAJS ################################################
    # buffer_r_exp = {
    #     'state': buffer_exp_failure['robot_state'].clone(),
    #     'action': buffer_exp_failure['robot_action'].clone(),
    #     'next_state': buffer_exp_failure['robot_next_state'].clone(),
    #     'done': buffer_exp_failure['done'].clone(),
    #     'reward': buffer_exp_failure['reward'].clone(),
    #     'label': buffer_exp_failure['label'].clone()
    # }

    # buffer_h_exp = {
    #     'state': buffer_exp_failure['human_state'].clone(),
    #     'action': buffer_exp_failure['human_action'].clone(),
    #     'next_state': buffer_exp_failure['human_next_state'].clone(),
    #     'done': buffer_exp_failure['done'].clone(),
    #     'reward': buffer_exp_failure['reward'].clone(),
    #     'label': buffer_exp_failure['label'].clone()
    # }

    ###########################################################################################################

    trainpath = os.getcwd()+f'/gail-airl-ppo/gail_airl_ppo/algo/models_airl/04-13-2022-19-37/'
    testpath = os.getcwd()+f'/gail-airl-ppo/gail_airl_ppo/algo/models_airl/04-30-2022-01-45/'

    airl = AIRL(env_id=env_id, buffer_r_exp=buffer_r_exp, buffer_h_exp=buffer_h_exp, device=device, seed=args.seed, 
                load_existing=args.load_existing, trainpath=trainpath, eval_interval=args.eval_interval)
    if not args.test:
        airl.train(args.num_steps, args.failure_traj)
    # else: airl.test(testpath, load_best=True)
    else: airl.test_disc(testpath, load_best=True)
