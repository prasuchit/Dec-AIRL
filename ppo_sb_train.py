import gym
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--total_timesteps', type=int, default=10 ** 6)
    p.add_argument('--env_id', type=str, default='CartPole-v1')
    p.add_argument('--env_num', type=int, default=4)
    p.add_argument('--verbose', type=int, default=1)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    # Parallel environments
    env = make_vec_env(args.env_id, n_envs=args.env_num)

    device = 'cuda' if args.cuda else 'cpu'
    print(f'Using {device}')

    model = PPO("MlpPolicy", env, verbose=args.verbose, device=device)
    model.learn(total_timesteps=args.total_timesteps)
    model.save(f"./weights/ppo_sb/ppo_{args.env_id.split('-')[0].lower()}")
