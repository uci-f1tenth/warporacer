"""PPO on Warp arrays end to end: rollout, GAE, clipped-surrogate updates.

On CUDA the whole rollout and each update epoch run as captured graphs (one replay call
instead of hundreds of kernel launches), so all state they touch lives on the device:
RNG advances via the env's tick array, normalizer counts are arrays, minibatch
permutations come from an in-graph radix sort, and the Adam learning rate is set by
writing its device array between replays.
"""

import numpy as np
import warp as wp
from warp_nn import optimizers

from warporacer.agent import LOGSTD_MAX, LOGSTD_MIN, clamp_log_std_kernel, sample_kernel
from warporacer.sim import ACT_DIM, OBS_DIM, bump_kernel

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP = 0.2
VF_CLIP = 0.2
VF_COEF = 0.5
ENT_COEF = 0.0
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.02
LR_MIN, LR_MAX, LR_FACTOR = 1e-6, 3e-3, 1.5
NORM_CLIP = 10.0
CHUNKS = 256  # parallel chunks for feature-stat reductions
ENTROPY_CONST = float(ACT_DIM * 0.5 * (1.0 + np.log(2.0 * np.pi)))
LOG_SQRT_2PI = float(0.5 * np.log(2.0 * np.pi))


class GraphedCall:
    """Run fn as a CUDA graph: call 1 is eager (JIT + buffer allocation), call 2 records
    and replays, later calls just replay. Eager everywhere on CPU or if capture fails."""

    def __init__(self, fn, device):
        self.fn, self.device, self.graph, self.calls = fn, device, None, 0

    def __call__(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
            return
        self.calls += 1
        if self.device.is_cuda and self.calls == 2:
            try:
                with wp.ScopedCapture(device=self.device) as capture:
                    self.fn()
                self.graph = capture.graph
            except Exception as e:
                print(f"[graph] capture failed, staying eager: {e}")
                self.calls = -(1 << 30)
                self.fn()
                return
            wp.capture_launch(self.graph)
        else:
            self.fn()


@wp.kernel
def normalize_kernel(
    src: wp.array2d(dtype=float),
    mean: wp.array(dtype=float),
    inv_std: wp.array(dtype=float),
    dst: wp.array2d(dtype=float),
):
    i, d = wp.tid()
    dst[i, d] = wp.clamp((src[i, d] - mean[d]) * inv_std[d], -NORM_CLIP, NORM_CLIP)


@wp.kernel
def accum_kernel(x: wp.array2d(dtype=float), row_start: int, acc: wp.array2d(dtype=float)):
    d, c = wp.tid()
    s = float(0.0)
    ss = float(0.0)
    for r in range(row_start + c, x.shape[0], CHUNKS):
        v = x[r, d]
        s += v
        ss += v * v
    wp.atomic_add(acc, 0, d, s)
    wp.atomic_add(acc, 1, d, ss)


@wp.kernel
def rms_merge_kernel(
    acc: wp.array2d(dtype=float),
    n1: float,
    mean: wp.array(dtype=float),
    var: wp.array(dtype=float),
    inv_std: wp.array(dtype=float),
    count: wp.array(dtype=float),
):
    d = wp.tid()
    bm = acc[0, d] / n1
    bv = wp.max(acc[1, d] / n1 - bm * bm, 0.0)
    delta = bm - mean[d]
    tot = count[d] + n1
    mean[d] += delta * n1 / tot
    var[d] = (var[d] * count[d] + bv * n1 + delta * delta * count[d] * n1 / tot) / tot
    inv_std[d] = 1.0 / wp.sqrt(var[d] + 1e-8)
    count[d] = tot
    acc[0, d] = 0.0
    acc[1, d] = 0.0


class RunningMeanStd:
    """Streaming per-feature mean/var: batches accumulate into acc on device, merge() folds them in."""

    def __init__(self, dim: int, device):
        self.mean = wp.zeros(dim, dtype=float, device=device)
        self.var = wp.full(dim, 1.0, dtype=wp.float32, device=device)
        self.inv_std = wp.full(dim, 1.0, dtype=wp.float32, device=device)
        self.count = wp.full(dim, 1e-4, dtype=wp.float32, device=device)
        self.acc = wp.zeros((2, dim), dtype=float, device=device)  # [sum, sum of squares]

    def normalize(self, src, dst):
        wp.launch(normalize_kernel, dim=src.shape, inputs=[src, self.mean, self.inv_std, dst],
                  device=self.mean.device)

    def accumulate(self, x2d, row_start: int = 0):
        wp.launch(accum_kernel, dim=(self.mean.shape[0], CHUNKS), inputs=[x2d, row_start, self.acc],
                  device=self.mean.device)

    def merge(self, batch_count: int):
        wp.launch(rms_merge_kernel, dim=self.mean.shape[0],
                  inputs=[self.acc, float(batch_count), self.mean, self.var, self.inv_std, self.count],
                  device=self.mean.device)


@wp.kernel
def track_stats_kernel(
    rew: wp.array(dtype=float),
    done: wp.array(dtype=wp.int32),
    ret_run: wp.array(dtype=float),
    ep_ret: wp.array(dtype=float),
    ep_len: wp.array(dtype=float),
    ret_acc: wp.array2d(dtype=float),
    fin: wp.array(dtype=float),  # [sum of returns, sum of lengths, count] over finished episodes
):
    i = wp.tid()
    r = rew[i]
    live = wp.where(done[i] != 0, 0.0, 1.0)
    rr = ret_run[i] * GAMMA * live + r
    ret_run[i] = rr
    wp.atomic_add(ret_acc, 0, 0, rr)
    wp.atomic_add(ret_acc, 1, 0, rr * rr)
    ep_ret[i] += r
    ep_len[i] += 1.0
    if done[i] != 0:
        wp.atomic_add(fin, 0, ep_ret[i])
        wp.atomic_add(fin, 1, ep_len[i])
        wp.atomic_add(fin, 2, 1.0)
        ep_ret[i] = 0.0
        ep_len[i] = 0.0


@wp.kernel
def gae_kernel(
    rew: wp.array2d(dtype=float),
    done: wp.array2d(dtype=wp.int32),
    val: wp.array2d(dtype=float),  # (T+1, N); the env self-resets on done, so the
    rew_inv_std: wp.array(dtype=float),  # obs behind val[t+1] at a done step is a fresh
    adv: wp.array2d(dtype=float),  # spawn -- zero the bootstrap there
    ret: wp.array2d(dtype=float),
):
    i = wp.tid()
    last = float(0.0)
    for t in range(rew.shape[0] - 1, -1, -1):
        live = wp.where(done[t, i] != 0, 0.0, 1.0)
        delta = rew[t, i] * rew_inv_std[0] + GAMMA * val[t + 1, i] * live - val[t, i]
        last = delta + GAMMA * GAE_LAMBDA * live * last
        adv[t, i] = last
        ret[t, i] = last + val[t, i]


@wp.kernel
def moments_kernel(x: wp.array(dtype=float), stride: int, acc: wp.array(dtype=float)):
    i = wp.tid()
    s = float(0.0)
    ss = float(0.0)
    for j in range(i, x.shape[0], stride):
        v = x[j]
        s += v
        ss += v * v
    wp.atomic_add(acc, 0, s)
    wp.atomic_add(acc, 1, ss)


@wp.kernel
def normalize_flat_kernel(x: wp.array(dtype=float), acc: wp.array(dtype=float)):
    i = wp.tid()
    n = float(x.shape[0])
    m = acc[0] / n
    var = wp.max(acc[1] / n - m * m, 0.0)
    x[i] = (x[i] - m) / (wp.sqrt(var) + 1e-8)


@wp.kernel
def fill_keys_kernel(
    seed: int,
    tick: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=float),
    values: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    rng = wp.rand_init(seed, tick[0] * keys.shape[0] + i)
    keys[i] = wp.randf(rng)
    values[i] = i


@wp.kernel
def gather_kernel(
    idx: wp.array(dtype=wp.int32),
    offset: int,
    mean: wp.array(dtype=float),
    inv_std: wp.array(dtype=float),
    obs: wp.array2d(dtype=float),
    act: wp.array2d(dtype=float),
    logp: wp.array(dtype=float),
    adv: wp.array(dtype=float),
    ret: wp.array(dtype=float),
    val: wp.array(dtype=float),
    mb_obs: wp.array2d(dtype=float),
    mb_act: wp.array2d(dtype=float),
    mb_logp: wp.array(dtype=float),
    mb_adv: wp.array(dtype=float),
    mb_ret: wp.array(dtype=float),
    mb_val: wp.array(dtype=float),
):
    i = wp.tid()
    src = idx[offset + i]
    for d in range(OBS_DIM):
        mb_obs[i, d] = wp.clamp((obs[src, d] - mean[d]) * inv_std[d], -NORM_CLIP, NORM_CLIP)
    for j in range(ACT_DIM):
        mb_act[i, j] = act[src, j]
    mb_logp[i] = logp[src]
    mb_adv[i] = adv[src]
    mb_ret[i] = ret[src]
    mb_val[i] = val[src]


@wp.kernel
def loss_kernel(
    mean: wp.array2d(dtype=float),
    value: wp.array2d(dtype=float),
    log_std: wp.array(dtype=float),
    act: wp.array2d(dtype=float),
    logp_old: wp.array(dtype=float),
    adv: wp.array(dtype=float),
    ret: wp.array(dtype=float),
    val_old: wp.array(dtype=float),
    loss: wp.array(dtype=float),
    stats: wp.array(dtype=float),  # [kl, clipfrac, pg, v, ent] accumulators
):
    i = wp.tid()
    lp = float(0.0)
    ent = float(ENTROPY_CONST)
    for j in range(ACT_DIM):
        ls = wp.clamp(log_std[j], LOGSTD_MIN, LOGSTD_MAX)
        z = (act[i, j] - mean[i, j]) * wp.exp(-ls)
        lp -= 0.5 * z * z + ls + LOG_SQRT_2PI
        ent += ls
    logratio = lp - logp_old[i]
    ratio = wp.exp(logratio)
    pg = -wp.min(ratio * adv[i], wp.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * adv[i])
    v = value[i, 0]
    v_err = v - ret[i]
    v_clip = val_old[i] + wp.clamp(v - val_old[i], -VF_CLIP, VF_CLIP) - ret[i]
    v_loss = 0.5 * wp.max(v_err * v_err, v_clip * v_clip)

    inv_m = 1.0 / float(mean.shape[0])
    wp.atomic_add(loss, 0, (pg + VF_COEF * v_loss - ENT_COEF * ent) * inv_m)
    wp.atomic_add(stats, 0, ((ratio - 1.0) - logratio) * inv_m)
    wp.atomic_add(stats, 1, wp.where(wp.abs(ratio - 1.0) > CLIP, inv_m, 0.0))
    wp.atomic_add(stats, 2, pg * inv_m)
    wp.atomic_add(stats, 3, v_loss * inv_m)
    wp.atomic_add(stats, 4, ent * inv_m)


class PPO:
    def __init__(self, env, agent, rollouts: int = 24, epochs: int = 5, minibatches: int = 4,
                 lr: float = 3e-4, seed: int = 0):
        assert (rollouts * env.num_envs) % minibatches == 0
        self.env, self.agent = env, agent
        self.epochs, self.minibatches = epochs, minibatches
        self.lr = lr
        self.device = env.device
        self.seed = seed
        self.global_step = 0
        self.iteration = 0
        T, N, d = rollouts, env.num_envs, env.device
        self.T, self.N = T, N
        self.batch_size = B = T * N
        self.mb = B // minibatches

        self.obs_b = wp.zeros((T + 1, N, OBS_DIM), dtype=float, device=d)  # raw obs
        self.obs_n = wp.zeros((N, OBS_DIM), dtype=float, device=d)  # normalized scratch
        self.act_b = wp.zeros((T, N, ACT_DIM), dtype=float, device=d)
        self.logp_b = wp.zeros((T, N), dtype=float, device=d)
        self.rew_b = wp.zeros((T, N), dtype=float, device=d)
        self.done_b = wp.zeros((T, N), dtype=wp.int32, device=d)
        self.val_b = wp.zeros((T + 1, N), dtype=float, device=d)
        self.adv_b = wp.zeros((T, N), dtype=float, device=d)
        self.ret_b = wp.zeros((T, N), dtype=float, device=d)
        self.obs_flat = self.obs_b.reshape(((T + 1) * N, OBS_DIM))
        self.act_flat = self.act_b.reshape((B, ACT_DIM))
        self.logp_flat = self.logp_b.reshape((B,))
        self.adv_flat = self.adv_b.reshape((B,))
        self.ret_flat = self.ret_b.reshape((B,))
        self.val_flat = self.val_b.reshape(((T + 1) * N,))

        # requires_grad: the tile-kernel backward of Linear writes the input's adjoint unconditionally
        self.mb_obs = wp.zeros((self.mb, OBS_DIM), dtype=float, device=d, requires_grad=True)
        self.mb_act = wp.zeros((self.mb, ACT_DIM), dtype=float, device=d)
        self.mb_logp = wp.zeros(self.mb, dtype=float, device=d)
        self.mb_adv = wp.zeros(self.mb, dtype=float, device=d)
        self.mb_ret = wp.zeros(self.mb, dtype=float, device=d)
        self.mb_val = wp.zeros(self.mb, dtype=float, device=d)
        self.keys = wp.zeros(2 * B, dtype=float, device=d)  # radix sort needs 2x capacity
        self.perm = wp.zeros(2 * B, dtype=wp.int32, device=d)
        self.loss = wp.zeros(1, dtype=float, device=d, requires_grad=True)
        self.stats = wp.zeros(5, dtype=float, device=d)
        self.adv_acc = wp.zeros(2, dtype=float, device=d)

        self.ret_run = wp.zeros(N, dtype=float, device=d)
        self.ep_ret = wp.zeros(N, dtype=float, device=d)
        self.ep_len = wp.zeros(N, dtype=float, device=d)
        self.fin = wp.zeros(3, dtype=float, device=d)

        self.obs_rms = RunningMeanStd(OBS_DIM, d)
        self.ret_rms = RunningMeanStd(1, d)
        # disable_graph: Adam's launches must record into our epoch graph, not a nested one
        self.opt = optimizers.Adam(agent.parameters(), lr=lr, device=d,
                                   max_norm=MAX_GRAD_NORM, disable_graph=True)

        # Seed obs stats and obs_b[T]: each iteration carries the last obs over to slot 0.
        wp.copy(self.obs_b[T], env.obs)
        self.obs_rms.accumulate(env.obs)
        self.obs_rms.merge(N)
        self._graph_rollout = GraphedCall(self._rollout_and_advantage, d)
        self._graph_epoch = GraphedCall(self._epoch, d)

    def _launch(self, kernel, dim, inputs, outputs=None):
        wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs or [], device=self.device)

    def iterate(self) -> dict:
        self.stats.zero_()
        self._graph_rollout()
        log = self._update()
        # Fold this iteration's raw obs into the stats only now, so rollout, values, and
        # minibatch gathers all normalized with the same frozen mean/std.
        self.obs_rms.accumulate(self.obs_flat, row_start=self.N)
        self.obs_rms.merge(self.batch_size)

        self.global_step += self.batch_size
        self.iteration += 1
        fin = self.fin.numpy()
        self.fin.zero_()
        if fin[2] > 0:
            log["ep_return"] = float(fin[0] / fin[2])
            log["ep_length"] = float(fin[1] / fin[2])
        log["log_std"] = float(self.agent.log_std.data.numpy().mean())
        log["lr"] = self.lr
        log["iteration"] = self.iteration
        return log

    def _rollout_and_advantage(self):
        T, N = self.T, self.N
        wp.copy(self.obs_b[0], self.obs_b[T])
        for t in range(T):
            self.obs_rms.normalize(self.obs_b[t], self.obs_n)
            mean = self.agent.actor(self.obs_n)
            self._launch(sample_kernel, N, [mean, self.agent.log_std.data, self.seed + 1, self.env.tick],
                         [self.act_b[t], self.logp_b[t]])
            self.env.step(self.act_b[t], self.obs_b[t + 1], self.rew_b[t], self.done_b[t])
            self._launch(track_stats_kernel, N,
                         [self.rew_b[t], self.done_b[t], self.ret_run, self.ep_ret, self.ep_len,
                          self.ret_rms.acc, self.fin])
        for t in range(T + 1):  # values in (N,) slices to reuse the rollout-sized activations
            self.obs_rms.normalize(self.obs_b[t], self.obs_n)
            wp.copy(self.val_b[t], self.agent.critic(self.obs_n).reshape((N,)))
        self.ret_rms.merge(self.batch_size)
        self._launch(gae_kernel, N, [self.rew_b, self.done_b, self.val_b, self.ret_rms.inv_std,
                                     self.adv_b, self.ret_b])
        self.adv_acc.zero_()
        stride = min(self.batch_size, 4096)
        self._launch(moments_kernel, stride, [self.adv_flat, stride, self.adv_acc])
        self._launch(normalize_flat_kernel, self.batch_size, [self.adv_flat, self.adv_acc])

    def _epoch(self):
        self._launch(fill_keys_kernel, self.batch_size, [self.seed + 2, self.env.tick, self.keys, self.perm])
        self._launch(bump_kernel, 1, [self.env.tick])
        wp.utils.radix_sort_pairs(self.keys, self.perm, self.batch_size)
        for k in range(self.minibatches):
            self._launch(gather_kernel, self.mb,
                         [self.perm, k * self.mb, self.obs_rms.mean, self.obs_rms.inv_std,
                          self.obs_flat, self.act_flat, self.logp_flat, self.adv_flat,
                          self.ret_flat, self.val_flat,
                          self.mb_obs, self.mb_act, self.mb_logp, self.mb_adv, self.mb_ret, self.mb_val])
            self.loss.zero_()
            with wp.Tape() as tape:
                mean = self.agent.actor(self.mb_obs)
                value = self.agent.critic(self.mb_obs)
                self._launch(loss_kernel, self.mb,
                             [mean, value, self.agent.log_std.data, self.mb_act, self.mb_logp,
                              self.mb_adv, self.mb_ret, self.mb_val],
                             [self.loss, self.stats])
            tape.backward(self.loss)
            self.opt.step()
            tape.zero()
            self._launch(clamp_log_std_kernel, ACT_DIM, [self.agent.log_std.data])

    def _update(self) -> dict:
        # Adam.step(lr=...) fills its device lr array host-side, which a graph would bake
        # in as a constant -- write it directly between replays instead.
        self.opt._lr.fill_(self.lr)
        n_upd, kl_prev, kl_stop = 0, 0.0, False
        for _ in range(self.epochs):
            self._graph_epoch()
            n_upd += self.minibatches
            kl_acc = float(self.stats.numpy()[0])
            epoch_kl = (kl_acc - kl_prev) / self.minibatches
            kl_prev = kl_acc
            if epoch_kl > 1.5 * TARGET_KL:
                kl_stop = True
                break

        kl, clipfrac, pg, v, ent = self.stats.numpy() / max(n_upd, 1)
        if kl > 2.0 * TARGET_KL:
            self.lr = max(LR_MIN, self.lr / LR_FACTOR)
        elif kl < 0.5 * TARGET_KL:
            self.lr = min(LR_MAX, self.lr * LR_FACTOR)
        return {"policy_loss": float(pg), "value_loss": float(v), "entropy": float(ent),
                "approx_kl": float(kl), "clipfrac": float(clipfrac), "kl_stop": int(kl_stop)}
