import argparse
import torch
from tqdm import tqdm
import numpy as np
from scipy import full, sparse
import copy
import math as m
import time
import logging
import os
from operator import mod
import pickle
from mdptoolbox.mdp import PolicyIterationModified
from hurosorting import HuRoSorting
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--total_timesteps', type=int, default=10 ** 6)
    p.add_argument('--env_id', type=str, default='HuRoSorting-v0')
    p.add_argument('--env_num', type=int, default=4)
    p.add_argument('--verbose', type=int, default=1)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    examineTraj = True
    examinePolicy = False
    test = False
    setTransition = False
    setReward = False
    debug = False
    setModelVals = False

    if os.path.exists("models/hurosorting/mdp.pkl"):
        picklefile = open("models/hurosorting/mdp.pkl", 'rb')
        #unpickle the dataframe
        mdp = pickle.load(picklefile)
        if setModelVals:
            newmdp = HuRoSorting(debug = debug, setReward=setReward, setTransition=setTransition)
            newmdp.setModelVals()
            if setTransition: 
                mdp.T_agent = copy.deepcopy(newmdp.T_agent)
                mdp.T_global = copy.deepcopy(newmdp.T_global)
            if setReward:
                mdp.R_global = copy.deepcopy(newmdp.R_global)
            #pickle the dictionary and write it to file
            pickle.dump(mdp, picklefile)
        #close the file
        picklefile.close()
    else:
        mdp = HuRoSorting(debug = debug)
        mdp.setModelVals()
        #create a pickle file
        picklefile = open('models/hurosorting/mdp.pkl', 'wb')
        #pickle the dictionary and write it to file
        pickle.dump(mdp, picklefile)
        #close the file
        picklefile.close()
    
    
    if os.path.exists('results/hurosorting/policy.csv') and not setModelVals:
        piL = np.loadtxt('results/hurosorting/policy.csv').astype(int)
    else:
        MAX_ITERS = 10000
        EPS = 1e-12
        SHOW_MSG = False
        pi = PolicyIterationModified(np.transpose(
            mdp.T_global), mdp.R_global, mdp.discount, max_iter=MAX_ITERS, epsilon=EPS)
        pi.run()
        # Q = utils.QfromV(pi.V, mdp)
        piL = np.reshape(pi.policy, (mdp.nSGlobal, 1))
        # H = evalToolbox(piL, mdp)
        np.savetxt('results/hurosorting/policy.csv', piL, fmt='%i')

    trajs_states = []
    trajs_next_states = []
    trajs_actions = []
    trajs_rewards = []
    trajs_dones = []
    done = False
    reward_stats = []
    length_stats = []
    ep_reward = 0
    ep_length = 0

    oloc_r, eefloc_r, pred_r, oloc_h, eefloc_h, pred_h = 0,3,0,0,3,0
    state = mdp.vals2sGlobal(oloc_r, eefloc_r, pred_r, oloc_h, eefloc_h, pred_h)

    for _ in tqdm(range(args.total_timesteps)):
        action = piL[state]
        next_state = np.random.choice(np.arange(mdp.nSGlobal), p = np.squeeze(mdp.T_global[:, state, int(piL[state])]))
        reward = mdp.R_global[state, action]
        a_r, a_h = mdp.aGlobal2vals(action)
        if a_r in [4,5] or a_h in [4,5]:
            done = True
        else: done = False
        ep_reward += reward
        ep_length += 1
        done = [int(done)]
        reward = [reward]

        trajs_states.append(state)
        trajs_next_states.append(next_state)
        trajs_actions.append(action)
        trajs_rewards.append(reward)
        trajs_dones.append(done)

        if done[0]:
            state = next_state
            reward_stats.append(ep_reward)
            length_stats.append(ep_length)
            ep_reward = 0
            ep_length = 0
        else:
            state = next_state

    trajs_states = torch.tensor(trajs_states).float()
    trajs_next_states = torch.tensor(trajs_next_states).float()
    trajs_actions = torch.tensor(trajs_actions).float()
    trajs_rewards = torch.tensor(trajs_rewards).float()
    trajs_dones = torch.tensor(trajs_dones).float()

    trajectories = {
        'state': trajs_states,
        'action': trajs_actions,
        'reward': trajs_rewards,
        'done': trajs_dones,
        'next_state': trajs_next_states
    }

    torch.save(trajectories, f'../buffers/policy_iteration/{args.env_id.split("-")[0].lower()}.pt')

    print(f'Reward Mean: {round(np.mean(reward_stats), 2)} | Reward Std: {round(np.std(reward_stats), 2)} | Episode Mean Length: {round(np.mean(length_stats), 2)} | Total Episodes: {len(reward_stats)}')