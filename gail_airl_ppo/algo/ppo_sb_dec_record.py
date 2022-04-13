from tabnanny import verbose
import gym
import argparse
from stable_baselines3.common.utils import obs_as_tensor
import torch
from tqdm import tqdm
import numpy as np
import os

from ppo_sb_dec import PPO_Dec, ActorCriticPolicy_Dec

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--total_timesteps', type=int, default=10 ** 2)
    p.add_argument('--env_id', type=str, default='ma_gym:HuRoSorting-v0')
    p.add_argument('--env_num', type=int, default=4)
    p.add_argument('--verbose', type=int, default=1)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    device = 'cuda' if args.cuda else 'cpu'
    print(f'Using {device}')
    # print(os.getcwd())
    env = gym.make(args.env_id)
    model_r = PPO_Dec(ActorCriticPolicy_Dec, env, verbose=1, airl=True, device=device, seed=args.seed)
    model_r.set_parameters(script_dir + f'/models/04-12-2022-10-51/model_r_95.zip', device=device)

    model_h = PPO_Dec(ActorCriticPolicy_Dec, env, verbose=1, airl=True, device=device, seed=args.seed)
    model_h.set_parameters(script_dir + f'/models/04-12-2022-10-51/model_h_95.zip', device=device)

    overwrite_ppo = True
    init_fixed = True
    done = False
    trajs_robot_states = []
    trajs_human_states = []
    trajs_robot_next_states = []
    trajs_human_next_states = []
    trajs_robot_actions = []
    trajs_human_actions = []
    trajs_rewards = []
    trajs_dones = []

    reward_stats = []
    length_stats = []
    ep_reward = 0
    ep_length = 0

    state = env.reset(init_fixed)
    
    # # transition = torch.load(os.getcwd()+f'/gail-airl-ppo/gail_airl_ppo/algo/buffers/ma_gym:hurosorting.pt')
    # # state_r = transition['robot_state'][0]
    # # state_h = transition['human_state'][0]

    # for _ in tqdm(range(args.total_timesteps)):
    #     state_robot = state[:11].copy()
    #     state_human = state[11:].copy()
    #     state_robot_input = np.concatenate([state_robot.copy(), state_human.copy()])
    #     state_human_input = np.concatenate([state_human.copy(), state_robot.copy()])
    #     with torch.no_grad():
    #         actions_r, values_r, log_probs_r = model_r.policy.forward(obs_as_tensor(state_robot_input, device=device).float())
    #         actions_h, values_h, log_probs_h = model_h.policy.forward(obs_as_tensor(state_human_input, device=device).float())
    #         print('DEBUG:', torch.exp(log_probs_r), torch.exp(log_probs_h))
    #     if overwrite_ppo:
    #         if (actions_r == actions_h == torch.tensor(1)):
    #             actions_r = torch.tensor(0)  # OVERWRITING PPO'S ACTION WITH THE RIGHT ACTION
    #         if (actions_r == actions_h == torch.tensor(2)):
    #             actions_r = torch.tensor(0)  # OVERWRITING PPO'S ACTION WITH THE RIGHT ACTION
    #         if (actions_r == actions_h == torch.tensor(4)):
    #             actions_r = torch.tensor(0)  # OVERWRITING PPO'S ACTION WITH THE RIGHT ACTION
    #     next_state, reward, done, _ = env.step([actions_r, actions_h], verbose = 1)

    #     ep_reward += reward
    #     ep_length += 1

    #     # print("State: ", next_state)
    #     print("Reward: ", reward)
    #     print("Done: ", done)
    #     if done:
    #         state = env.reset(init_fixed)
    #         reward_stats.append(ep_reward)
    #         length_stats.append(ep_length)
    #         done = False
    #         ep_reward = 0
    #         ep_length = 0
    #     else:
    #         state = next_state
    # print(f'Reward Mean: {round(np.mean(reward_stats), 2)} | Reward Std: {round(np.std(reward_stats), 2)} | Episode Mean Length: {round(np.mean(length_stats), 2)} | Total Episodes: {len(reward_stats)}')

#################################################################################################################################################################################################################

    for _ in tqdm(range(args.total_timesteps)):

        state_robot = state[:11].copy()
        state_human = state[11:].copy()
        state_robot_input = np.concatenate([state_robot.copy(), state_human.copy()])
        state_human_input = np.concatenate([state_human.copy(), state_robot.copy()])

        with torch.no_grad():
            actions_r, values_r, log_probs_r = model_r.policy.forward(obs_as_tensor(state_robot_input, device=device).float())
            actions_h, values_h, log_probs_h = model_h.policy.forward(obs_as_tensor(state_human_input, device=device).float())
  
        if overwrite_ppo:
            if (actions_r == actions_h == torch.tensor(1)):
                actions_r = torch.tensor(0)  # OVERWRITING PPO'S ACTION WITH THE RIGHT ACTION
            if (actions_r == actions_h == torch.tensor(2)):
                actions_r = torch.tensor(0)  # OVERWRITING PPO'S ACTION WITH THE RIGHT ACTION
            if (actions_r == actions_h == torch.tensor(4)):
                actions_r = torch.tensor(0)  # OVERWRITING PPO'S ACTION WITH THE RIGHT ACTION
        next_state, reward, done, _ = env.step([actions_r, actions_h], verbose = 0)
        ep_reward += reward
        ep_length += 1

        next_state_robot = next_state[:11].copy()
        next_state_human = next_state[11:].copy()
        next_state_robot_input = np.concatenate([next_state_robot.copy(), next_state_human.copy()])
        next_state_human_input = np.concatenate([next_state_human.copy(), next_state_robot.copy()])

        trajs_robot_states.append(state_robot_input.copy())
        trajs_human_states.append(state_human_input.copy())
        trajs_robot_next_states.append(next_state_robot_input.copy())
        trajs_human_next_states.append(next_state_human_input.copy())
        trajs_robot_actions.append(actions_r)
        trajs_human_actions.append(actions_h)
        trajs_rewards.append(reward)
        trajs_dones.append(int(done))

        if done:
            state = env.reset(init_fixed)
            reward_stats.append(ep_reward)
            length_stats.append(ep_length)
            done = False
            ep_reward = 0
            ep_length = 0
        else:
            state = next_state
    
    trajs_robot_states = torch.tensor(trajs_robot_states).float()
    trajs_human_states = torch.tensor(trajs_human_states).float()
    trajs_robot_next_states = torch.tensor(trajs_robot_next_states).float()
    trajs_human_next_states = torch.tensor(trajs_human_next_states).float()
    trajs_robot_actions = torch.tensor(trajs_robot_actions).float()
    trajs_human_actions = torch.tensor(trajs_human_actions).float()
    trajs_rewards = torch.tensor(trajs_rewards).float()
    trajs_dones = torch.tensor(trajs_dones).float()

    trajectories = {
        'robot_state': trajs_robot_states,
        'human_state': trajs_human_states,
        'robot_action': trajs_robot_actions,
        'human_action': trajs_human_actions,
        'reward': trajs_rewards,
        'done': trajs_dones,
        'robot_next_state': trajs_robot_next_states,
        'human_next_state': trajs_human_next_states
    }

    torch.save(trajectories, script_dir + f'/buffers/{args.env_id.split("-")[0].lower()}.pt')

    print(f'Reward Mean: {round(np.mean(reward_stats), 2)} | Reward Std: {round(np.std(reward_stats), 2)} | Episode Mean Length: {round(np.mean(length_stats), 2)} | Total Episodes: {len(reward_stats)}')
