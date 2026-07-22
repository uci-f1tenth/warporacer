"""Actor-critic MLP (torch) with a state-independent Gaussian policy."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from warporacer.sim import ACT_DIM, OBS_DIM

LOGSTD_MIN, LOGSTD_MAX = -1.6, -0.3


def _layer_init(layer, std=np.sqrt(2.0)):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _mlp(out_dim, head_std, hidden):
    return nn.Sequential(
        _layer_init(nn.Linear(OBS_DIM, hidden)),
        nn.Tanh(),
        _layer_init(nn.Linear(hidden, hidden)),
        nn.Tanh(),
        _layer_init(nn.Linear(hidden, out_dim), std=head_std),
    )


class Agent(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.actor = _mlp(ACT_DIM, 0.01, hidden)
        self.critic = _mlp(1, 1.0, hidden)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5))

    def dist(self, obs):
        return Normal(self.actor(obs), self.log_std.clamp(LOGSTD_MIN, LOGSTD_MAX).exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)
