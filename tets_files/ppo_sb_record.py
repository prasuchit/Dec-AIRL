from tabnanny import verbose
import gym
import argparse
from stable_baselines3 import PPO
import torch
from tqdm import tqdm
import numpy as np
import os

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--total_timesteps', type=int, default=10 ** 5)
    p.add_argument('--env_id', type=str, default='ma_gym:HuRoSorting-v0')
    p.add_argument('--env_num', type=int, default=4)
    p.add_argument('--verbose', type=int, default=1)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    device = 'cuda' if args.cuda else 'cpu'
    print(f'Using {device}')
    # print(os.getcwd())
    model = PPO.load(os.getcwd()+ f"/gail-airl-ppo/weights/ppo_sb/ppo_{args.env_id.split('-')[0].lower()}", device=device)

    env = gym.make(args.env_id)
    state = env.reset()
    done = False
    trajs_states = []
    trajs_next_states = []
    trajs_actions = []
    trajs_rewards = []
    trajs_dones = []

    reward_stats = []
    length_stats = []
    ep_reward = 0
    ep_length = 0

    while not done:
        action, _ = model.predict(state + np.random.normal(scale=0.02, size=state.shape))
        next_state, reward, done, _ = env.step(action, verbose = 1)
        print("Reward: ", reward)
        print("Done: ", done)
        if done:
            state = env.reset()
            reward_stats.append(ep_reward)
            length_stats.append(ep_length)
            ep_reward = 0
            ep_length = 0
        else:
            state = next_state


    # for _ in tqdm(range(args.total_timesteps)):
    #     action, _ = model.predict(state + np.random.normal(scale=0.02, size=state.shape))
    #     next_state, reward, done, _ = env.step(action)
    #     ep_reward += reward
    #     ep_length += 1

    #     if env.action_space.__class__.__name__ == 'Discrete':
    #         action = [action]
    #     done = [int(done)]
    #     reward = [reward]

    #     trajs_states.append(state)
    #     trajs_next_states.append(next_state)
    #     trajs_actions.append(action)
    #     trajs_rewards.append(reward)
    #     trajs_dones.append(done)
    #     if len(next_state) > 22:
    #         print(f"state: {state}, action: {action}, nextstate: {next_state}, reward: {reward}, done: {done}")
    #         print("Ah shit, here we go again!")

    #     if done[0]:
    #         state = env.reset()
    #         reward_stats.append(ep_reward)
    #         length_stats.append(ep_length)
    #         ep_reward = 0
    #         ep_length = 0
    #     else:
    #         state = next_state

    # trajs_states = torch.tensor(trajs_states).float()
    # trajs_next_states = torch.tensor(trajs_next_states).float()
    # trajs_actions = torch.tensor(trajs_actions).float()
    # trajs_rewards = torch.tensor(trajs_rewards).float()
    # trajs_dones = torch.tensor(trajs_dones).float()

    # trajectories = {
    #     'state': trajs_states,
    #     'action': trajs_actions,
    #     'reward': trajs_rewards,
    #     'done': trajs_dones,
    #     'next_state': trajs_next_states
    # }

    # torch.save(trajectories, os.getcwd()+ f'/gail-airl-ppo/buffers/ppo_sb/{args.env_id.split("-")[0].lower()}.pt')

    # print(f'Reward Mean: {round(np.mean(reward_stats), 2)} | Reward Std: {round(np.std(reward_stats), 2)} | Episode Mean Length: {round(np.mean(length_stats), 2)} | Total Episodes: {len(reward_stats)}')


