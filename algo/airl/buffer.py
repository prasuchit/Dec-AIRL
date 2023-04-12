import torch

class Buffers_AIRL(object):
    def __init__(self, 
                 buffer_size: int):

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
        
        self.buffer_size = buffer_size
        
    def add(self, buffer, state, action, next_state, reward, done, value, log_prob, info):
        p = buffer['p']
        for agent_id in self.agents:
            buffer['state'][agent_id][p] = torch.as_tensor(state[agent_id]).clone().float().to(self.device)
            buffer['action'][agent_id][p] = torch.as_tensor(action[agent_id]).clone().float().to(self.device)
            buffer['next_state'][agent_id][p] = torch.as_tensor(next_state[agent_id]).clone().float().to(self.device)
            buffer['reward'][agent_id][p] = torch.as_tensor(reward[agent_id]).to(self.device)
            buffer['done'][agent_id][p] = torch.as_tensor(int(done[agent_id])).to(self.device)
            buffer['value'][agent_id][p] = torch.as_tensor(value[agent_id]).to(self.device)
            buffer['log_prob'][agent_id][p] = torch.as_tensor(log_prob[agent_id]).to(self.device)

        buffer['info'][p] = [info]  # This might be a dict, so can't convert to tensor
        buffer['p'] += 1
        buffer['p'] %= self.buffer_size
        buffer['record'] += 1

    def sample(self, buffer, expert=False):
        if not expert:
            current_buffer_size = min(buffer['record'], self.buffer_size)
            idx = torch.randperm(current_buffer_size)[:self.batch_size]
            return \
                {agent_id: buffer['state'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: buffer['action'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: buffer['next_state'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: buffer['reward'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: buffer['done'][agent_id][idx] for agent_id in self.agents},\
                {agent_id: buffer['value'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: buffer['log_prob'][agent_id][idx].to(self.device) for agent_id in self.agents}
        else:
            current_buffer_size = len(buffer['state'][self.agents[0]])
            idx = torch.randperm(current_buffer_size)[:self.batch_size]
            return \
                {agent_id: buffer['state'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: buffer['action'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: buffer['next_state'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: buffer['reward'][agent_id][idx].to(self.device) for agent_id in self.agents},\
                {agent_id: buffer['done'][agent_id][idx] for agent_id in self.agents}

    def get(self, buffer):
        current_buffer_size = min(buffer['record'], self.buffer_size)
        return \
            {agent_id: buffer['state'][agent_id][:current_buffer_size].to(self.device) for agent_id in self.agents},\
            {agent_id: buffer['action'][agent_id][:current_buffer_size].to(self.device) for agent_id in self.agents},\
            {agent_id: buffer['next_state'][agent_id][:current_buffer_size].to(self.device) for agent_id in self.agents},\
            {agent_id: buffer['reward'][agent_id][:current_buffer_size].to(self.device) for agent_id in self.agents},\
            {agent_id: buffer['done'][agent_id][:current_buffer_size] for agent_id in self.agents},\
            {agent_id: buffer['value'][agent_id][:current_buffer_size].to(self.device) for agent_id in self.agents},\
            {agent_id: buffer['log_prob'][agent_id][:current_buffer_size].to(self.device) for agent_id in self.agents},\
            {agent_id: {} for agent_id in self.agents}
