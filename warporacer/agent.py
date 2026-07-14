"""Actor-critic MLP (warp-nn) and the Gaussian policy kernels."""

import numpy as np
import warp as wp
from warp_nn import nn

from warporacer.sim import ACT_DIM, OBS_DIM

LOGSTD_MIN, LOGSTD_MAX = -1.6, -0.3
LOG_SQRT_2PI = float(0.5 * np.log(2.0 * np.pi))


class Agent(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, ACT_DIM)
        )
        self.critic = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.log_std = nn.Parameter(wp.full(ACT_DIM, -0.5, dtype=wp.float32))
        super().__post_init__()


def orthogonal_init(agent: Agent, rng: np.random.Generator):
    """Orthogonal weights, zero bias: sqrt(2) gain hidden, 0.01 policy head, 1.0 value head."""
    for seq, head_gain in ((agent.actor, 0.01), (agent.critic, 1.0)):
        linears = [m for m in seq.modules() if isinstance(m, nn.Linear)]
        gains = [np.sqrt(2.0)] * (len(linears) - 1) + [head_gain]
        for lin, gain in zip(linears, gains):
            lin.weight.data.assign(_orthogonal(lin.weight.shape, gain, rng))
            lin.bias.data.zero_()


def _orthogonal(shape, gain, rng):
    a = rng.standard_normal(shape)
    if shape[0] < shape[1]:
        a = a.T
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    if shape[0] < shape[1]:
        q = q.T
    return (gain * q).astype(np.float32)


@wp.kernel
def sample_kernel(
    mean: wp.array2d(dtype=float),
    log_std: wp.array(dtype=float),
    seed: int,
    tick: wp.array(dtype=wp.int32),  # device RNG clock: keeps randomness fresh across CUDA graph replays
    act: wp.array2d(dtype=float),
    logp: wp.array(dtype=float),
):
    i = wp.tid()
    rng = wp.rand_init(seed, tick[0] * mean.shape[0] + i)
    lp = float(0.0)
    for j in range(ACT_DIM):
        ls = wp.clamp(log_std[j], LOGSTD_MIN, LOGSTD_MAX)
        z = wp.randn(rng)
        act[i, j] = mean[i, j] + wp.exp(ls) * z
        lp -= 0.5 * z * z + ls + LOG_SQRT_2PI
    logp[i] = lp


@wp.kernel
def clamp_log_std_kernel(log_std: wp.array(dtype=float)):
    i = wp.tid()
    log_std[i] = wp.clamp(log_std[i], LOGSTD_MIN, LOGSTD_MAX)
