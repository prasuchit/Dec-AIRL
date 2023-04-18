import torch

class Buffers_AIRL(object):
    def __init__(self, 
                 batch_size,
                 agents,
                 local_observation_shape,
                 local_action_shape,
                 device,
                 expert_buffer,        
                 buffer_size: int):
        self.batch_size=batch_size
        self.local_observation_shape = local_observation_shape
        self.local_action_shape = local_action_shape
        self.device = device
        self.agents = agents
        self.policy_buffer = {
            'state': {
                agent_id: torch.zeros(size=(buffer_size, self.local_observation_shape[agent_id]), device=self.device)
                for agent_id in self.agents
            },
            'action': {
                agent_id: torch.zeros(size=(buffer_size, self.local_action_shape[agent_id]), device=self.device)
                for agent_id in self.agents
            },
            'next_state': {
                agent_id: torch.zeros(size=(buffer_size, self.local_observation_shape[agent_id]), device=self.device)
                for agent_id in self.agents
            },
            'reward': {
                agent_id: torch.zeros(buffer_size, device=self.device)
                for agent_id in self.agents
            },
            'done': {
                agent_id: torch.zeros(buffer_size, device=self.device)
                for agent_id in self.agents
            },
            'log_prob': {
                agent_id: torch.zeros(buffer_size, device=self.device)
                for agent_id in self.agents
            },
            'value': {
                agent_id: torch.zeros(buffer_size, device=self.device)
                for agent_id in self.agents
            },
            'info': [[{}]] * buffer_size,
            'p': 0,
            'record': 0
        }
        
        self.expert_buffer = expert_buffer
        
        self.buffer_size = buffer_size
        
    def add(self, state, action, next_state, reward, done, value, log_prob, info):
        p = self.policy_buffer['p']
        for agent_id in self.agents:
            self.policy_buffer['state'][agent_id][p] = torch.as_tensor(state[agent_id]).clone().float().to(self.device)
            self.policy_buffer['action'][agent_id][p] = torch.as_tensor(action[agent_id]).clone().float().to(self.device)
            self.policy_buffer['next_state'][agent_id][p] = torch.as_tensor(next_state[agent_id]).clone().float().to(self.device)
            self.policy_buffer['reward'][agent_id][p] = torch.as_tensor(reward[agent_id]).to(self.device)
            self.policy_buffer['done'][agent_id][p] = torch.as_tensor(int(done[agent_id])).to(self.device)
            self.policy_buffer['value'][agent_id][p] = torch.as_tensor(value[agent_id]).to(self.device)
            self.policy_buffer['log_prob'][agent_id][p] = torch.as_tensor(log_prob[agent_id]).to(self.device)

        self.policy_buffer['info'][p] = [info]  # This might be a dict, so can't convert to tensor
        self.policy_buffer['p'] += 1
        self.policy_buffer['p'] %= self.buffer_size
        self.policy_buffer['record'] += 1

    def sample(self, expert=False):
        if not expert:
            current_buffer_size = min(self.policy_buffer['record'], self.buffer_size)
            idx = torch.randperm(current_buffer_size)[:self.batch_size]
            return \
                {agent_id: self.policy_buffer['state'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: self.policy_buffer['action'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: self.policy_buffer['next_state'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: self.policy_buffer['reward'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: self.policy_buffer['done'][agent_id][idx] for agent_id in self.agents},\
                {agent_id: self.policy_buffer['value'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: self.policy_buffer['log_prob'][agent_id][idx].to(self.device) for agent_id in self.agents}
        else:
            current_buffer_size = len(self.expert_buffer['state'][self.agents[0]])
            idx = torch.randperm(current_buffer_size)[:self.batch_size]
            return \
                {agent_id: self.expert_buffer['state'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: self.expert_buffer['action'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: self.expert_buffer['next_state'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: self.expert_buffer['reward'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: self.expert_buffer['done'][agent_id][idx] for agent_id in self.agents}

    def get(self):
        current_buffer_size = min(self.policy_buffer['record'], self.buffer_size)
        return \
            {agent_id: self.policy_buffer['state'][agent_id][:current_buffer_size].to(self.device) for agent_id in self.agents},\
            {agent_id: self.policy_buffer['action'][agent_id][:current_buffer_size].to(self.device) for agent_id in self.agents},\
            {agent_id: self.policy_buffer['next_state'][agent_id][:current_buffer_size].to(self.device) for agent_id in self.agents},\
            {agent_id: self.policy_buffer['reward'][agent_id][:current_buffer_size].to(self.device) for agent_id in self.agents},\
            {agent_id: self.policy_buffer['done'][agent_id][:current_buffer_size] for agent_id in self.agents},\
            {agent_id: self.policy_buffer['value'][agent_id][:current_buffer_size].to(self.device) for agent_id in self.agents},\
            {agent_id: self.policy_buffer['log_prob'][agent_id][:current_buffer_size].to(self.device) for agent_id in self.agents},\
            {agent_id: {} for agent_id in self.agents}
