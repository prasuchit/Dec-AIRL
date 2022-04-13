from abc import ABC, abstractmethod
import os
import numpy as np
import torch


class Algorithm(ABC):

    def __init__(self, state_shape, action_shape, device, seed, gamma):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.learning_steps = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma

    def explore(self, state, action_discrete=False):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            # action.shape = torch.Size([1, 1]), log_pi.shape = torch.Size([1, 1])
            if action_discrete:
                action, log_pi = self.actor.sample(state.unsqueeze_(0), action_discrete=action_discrete)
            else: action, log_pi = self.actor.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state, action_discrete=False):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            if action_discrete:
                action = self.actor(state.unsqueeze_(0), action_discrete=action_discrete)
            else: action = self.actor(state.unsqueeze_(0))
        return action.cpu().numpy()[0]

    @abstractmethod
    def is_update(self, step):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
