import os
import argparse
from datetime import datetime
import torch

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SAC, PPO
from gail_airl_ppo.trainer import Trainer


def run(args):
    env = make_env(args.env_id)
    env_test = make_env(args.env_id)

    algo = PPO(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        coef_ent=args.coef_ent,
        action_discrete=env.action_space.__class__.__name__ == 'Discrete'
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, 'ppo', f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        action_discrete=env.action_space.__class__.__name__ == 'Discrete'
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=10**6)
    p.add_argument('--eval_interval', type=int, default=10**4)
    p.add_argument('--coef_ent', type=float, default=0.0)
    p.add_argument('--env_id', type=str, default='Hopper-v3')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    print(f'Env: {args.env_id}')
    run(args)
