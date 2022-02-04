import torch
from torch import nn
from torch.distributions import Categorical

from .utils import build_mlp, reparameterize, evaluate_lop_pi


class StateIndependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states, action_discrete=False):
        if action_discrete:
            actions = torch.argmax(self.net(states)).item()
            return torch.tensor([[actions]]).float()
        else:
            return torch.tanh(self.net(states))

    def sample(self, states, action_discrete=False):
        if action_discrete:
            output = self.net(states)
            actions = torch.argmax(output).item()
            # dist = Categorical(torch.nn.functional.softmax(output))
            # actions = dist.sample()
            log_pi = torch.nn.functional.log_softmax(output)[0][actions]
            # log_pi = dist.log_prob(actions)
            return torch.tensor([[actions]]).float(), torch.tensor([[log_pi]]).float()
        else:
            return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions, action_discrete=False):
        if action_discrete:
            logits = self.net(states)
            pis = torch.nn.functional.softmax(logits)
            log_pis = torch.nn.functional.log_softmax(logits)
            entropy = -(pis * log_pis).mean()
            actions = torch.argmax(logits, dim=1)
            log_pi = log_pis.gather(1, actions.view(-1, 1).long())
            return log_pi, entropy
        else:
            return evaluate_lop_pi(self.net(states), self.log_stds, actions), torch.tensor(0)


class StateDependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp_(-20, 2))
