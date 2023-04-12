
from typing import NamedTuple, Tuple

import torch as th
from stable_baselines3.common.type_aliases import TensorDict
from sb3_contrib.common.recurrent.type_aliases import (
    RecurrentDictRolloutBufferSamples,
    RecurrentRolloutBufferSamples,
    RNNStates,
)

class RolloutBufferSamples_Dec(NamedTuple):
    local_observations: th.as_tensor
    global_observations: th.as_tensor
    actions: th.as_tensor
    old_values: th.as_tensor
    old_log_prob: th.as_tensor
    advantages: th.as_tensor
    returns: th.as_tensor
    
class RecurrentRolloutBufferSamples_Dec(NamedTuple):
    local_observations: th.as_tensor
    global_observations: th.as_tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor

class RecurrentDictRolloutBufferSamples_Dec(NamedTuple):
    local_observations: th.as_tensor
    global_observations: th.as_tensor
    actions: th.as_tensor
    old_values: th.as_tensor
    old_log_prob: th.as_tensor
    advantages: th.as_tensor
    returns: th.as_tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor