import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

import gym
import argparse
import sys
import os, importlib
import numpy as np
# import shutup; shutup.please()


path = os.path.dirname (os.path.realpath (__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

sys.path.append(PACKAGE_PATH)
from algo.airl.airl import AIRL
from algo.ppo.ppo import obs_as_tensor

''' Adversarial IRL class that extends the original paper Fu et al. 2017(https://arxiv.org/pdf/1710.11248.pdf) to work with multiple agents'''


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)
    return None

class AIRL_Test(AIRL):
    def __init__(self, env_id, buffers_exp=None, seed=None, eval_interval=500,
                 gamma=0.95, n_steps=2048, device='cpu',
                 batch_size=128, lr_actor=3e-4, lr_disc=3e-4,
                 units_disc_r=(256, 256), units_disc_v=(256, 256),
                 epoch_actor=10, epoch_disc=10, clip_eps=0.2, gae_lambda=0.97,
                 ent_coef=0.0, max_grad_norm=0.5, path = os.getcwd()):

        AIRL.__init__(self, env_id, buffers_exp, seed, eval_interval=eval_interval,
                 gamma=gamma, n_steps=n_steps, device=device,
                 batch_size=batch_size, lr_actor=lr_actor, lr_disc=lr_disc,
                 units_disc_r=units_disc_r, units_disc_v=units_disc_v,
                 epoch_actor=epoch_actor, epoch_disc=epoch_disc, clip_eps=clip_eps, gae_lambda=gae_lambda,
                 ent_coef=ent_coef, max_grad_norm=max_grad_norm, path = path) 

    def test_disc(self, path):
        raise NotImplementedError

    def set_seed(self, seed):
        self.seed = seed

    def model_loader(self, path):
        for i in self.agents:
            self.actors[i].set_parameters(f'{path}/{i}.zip',  device=self.device)
        self.disc.load_state_dict(torch.load(f'{path}/disc.pt'))

    def test(self, path, test_epochs = 1, render = False):
        ep_rewards = []
        ep_lengths = []
        verbose = True
        self.model_loader(path)
        test_action_actor = {}
        test_action_actor_log_prob = {}
        if render:
            self.test_env.render()

        for test_epoch in range(test_epochs):
            test_state = self.test_env.reset()
            test_done = False
            ep_length = 0
            ep_reward = 0
            while not test_done:
                for i in self.agents:
                    local_state = torch.as_tensor(test_state[i]).to(self.device)
                    global_state = torch.cat([obs_as_tensor(test_state['robot'], device=self.device), obs_as_tensor(test_state['human'], device=self.device)], dim=1).to(self.device)
                    test_action_actor[i], _, test_action_actor_log_prob[i] = self.actors[i].policy.forward(local_state.to(self.device), global_state.to(self.device))
                    test_action_actor[i] = test_action_actor[i].detach().clone().numpy().squeeze()
                test_next_state, test_reward, test_done, test_info = self.test_env.step(test_action_actor)

                test_reward = (test_reward['robot'] + test_reward['human']) / 2
                test_done = test_done['__all__']

                # print(f"Reward: {test_reward}, Done: {test_done}")

                ep_reward += test_reward
                ep_length += 1
                test_state = test_next_state

            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)

        print(f'mean length: {round(np.mean(ep_lengths), 1)} | mean reward: {round(np.mean(ep_rewards), 1)}')
        return np.mean(ep_lengths), np.mean(ep_rewards)



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # p.add_argument('--buffer', type=str, required=True)
    # p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--num_steps', type=int, default=10 ** 7)
    p.add_argument('--eval_interval', type=int, default=4096)
    p.add_argument('--env_id', type=str, default='FeedingSawyerHuman-v1')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--failure_traj', action='store_true')
    p.add_argument('--load_existing', action='store_true')
    p.add_argument('--test', action='store_true')
    p.add_argument('--model_path', type=str, default='2022-10-04_15-12/step_1929216_reward_135')
    args = p.parse_args()

    env_id = args.env_id
    device = 'cuda:0' if args.cuda else 'cpu'

    load_dir = f'{PACKAGE_PATH}/models_airl/' + args.model_path

    airl = AIRL_Test(env_id=env_id, device=device, seed=args.seed)
    ep_len, avg_reward = [], []
    import random
    for i in range(1000):
        airl.set_seed(random.randint(0,1000))
        print("Test iter: ", i)
        len, reward = airl.test(path=load_dir, render=False)
        ep_len.append(len)
        avg_reward.append(reward)

    print(f"Avg_len: {np.mean(ep_len)}, Avg_reward: {np.mean(avg_reward)}")
    np.save('avg_len.npy', np.mean(ep_len))
    np.save('avg_rew.npy', np.mean(avg_reward))
    np.save('len_list.npy', ep_len)
    np.save('rew_list.npy', avg_reward)