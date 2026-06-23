"""Reinforcement learning simulation environment managing batched multi-map vehicle physics."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np
import torch
import warp as wp

from include.constants import (
    DONE_TERMINATED,
    DONE_TRUNCATED,
    DR_FRAC,
    LIDAR_FOV,
    LIDAR_RANGE,
    MAX_STEPS,
    NUM_LIDAR,
    OBS_DIM,
)
from include.map import Map
from include.warped_functions import step_kernel

if TYPE_CHECKING:
    from include.visuals import Visuals

# --- CONFIGURATION CONSTANTS ---
INITIAL_PACKING_BOX_SIZE = 10000.0
MAP_GAP_MARGIN = 1.0

# Linear Congruential Generator (LCG) Seed Constants
SEED_MULTIPLIER_BASE = 2654435761
SEED_MULTIPLIER_CALL = 83492791
SEED_MASK = 0x7FFFFFFF


@dataclass
class MapBuffers:
    """Encapsulates accelerated layout tracking buffers resident permanently on GPU VRAM."""
    dt_buf: Optional[wp.array] = None
    lut_buf: Optional[wp.array] = None
    centerline_buf: Optional[wp.array] = None
    maps_origin: Optional[wp.array] = None
    maps_res: Optional[wp.array] = None
    maps_n_cl: Optional[wp.array] = None
    maps_mh_f: Optional[wp.array] = None


@dataclass
class AgentBuffers:
    """Manages active simulation instances and state tracking blocks on Device memory."""
    cars: wp.array
    cars_int: wp.array
    car_dr: wp.array
    obs: wp.array
    critic_obs: wp.array
    rew: wp.array
    done: wp.array
    lidar_buf: wp.array
    zero_act: wp.array
    env_map_ids: wp.array


@dataclass
class TorchViews:
    """Manages Zero-Copy PyTorch overlay mirrors for direct tensor compilation mapping."""
    obs_buf: torch.Tensor
    critic_obs_buf: torch.Tensor
    rew_buf: torch.Tensor
    done_buf: torch.Tensor
    cars_buf: torch.Tensor
    cars_int_buf: torch.Tensor
    term_buf: torch.Tensor
    trunc_buf: torch.Tensor


class Environment:
    """Manages parallel vehicle simulations across distinct clustered 2D track maps.

    Coordinates in-place updates between PyTorch tensors and accelerated NVIDIA Warp
    memory blocks without re-allocating memory during active training hot-loops.
    """

    num_envs: int
    seed: int
    seed_base: int
    device: wp.Device
    maps: list[Map]
    num_maps: int
    max_active_maps: int
    floor_square_size: float
    available_maps: list[Path]
    vs: Optional["Visuals"]

    # Structured State Containers
    maps_storage: MapBuffers
    agents: AgentBuffers
    views: TorchViews

    _step_counter: torch.Tensor
    _empty_info: Dict[str, Any]
    _call: int

    def __init__(
        self,
        maps_dir: Path,
        num_envs: int,
        seed: int,
        target_device: wp.Device,
        live_viewer: bool,
        max_active_maps: int = 10,
    ) -> None:
        """Initializes the simulation cluster and structures tracking views."""
        self.num_envs = num_envs
        self.seed = seed
        self.seed_base = seed
        self._call = 0
        self.device = target_device

        self.max_active_maps = max_active_maps
        self.floor_square_size = 0.0
        self.available_maps = []
        self.vs = None

        # Container Declarations
        self.maps_storage = MapBuffers()
        self._initialize_map_library(maps_dir)
        self._allocate_persistent_buffers()
        self.trigger_map_rotation()

        if live_viewer:
            from include.visuals import Visuals
            self.vs = Visuals(self)

    def trigger_map_rotation(self) -> None:
        """Swaps the active computational track pool dynamically."""
        self._call += 1
        rng = np.random.default_rng(self.seed + self._call)

        sample_size = min(self.max_active_maps, len(self.available_maps))
        chosen_paths = rng.choice(
            self.available_maps, size=sample_size, replace=False
        )

        self.maps = [Map(p) for p in chosen_paths]
        # import concurrent.futures
        # self.maps = []
        # if sample_size > 0:
        #     with concurrent.futures.ThreadPoolExecutor() as executor:
        #         self.maps = list(executor.map(Map, chosen_paths))
        self.num_maps = len(self.maps)

        global_shifts = self._initialize_active_maps()
        self._shuffle_and_assign_maps(global_shifts)

        if self.vs is not None:
            self.vs.refresh_maps()

    def _initialize_map_library(self, maps_dir: Path) -> None:
        """Parses target path structures into predictable system libraries."""
        resolved_target = Path(maps_dir).resolve()
        if resolved_target.is_file():
            self.available_maps = [resolved_target]
        elif resolved_target.is_dir():
            self.available_maps = sorted(
                list(resolved_target.glob("*.yaml")), key=lambda p: p.name
            )
            if not self.available_maps:
                raise FileNotFoundError(
                    f"[Env Error] No .yaml configurations inside: {resolved_target}"
                )
        else:
            raise FileNotFoundError(
                f"[Env Error] Map target path is invalid: {resolved_target}"
            )

    def _allocate_persistent_buffers(self) -> None:
        """Allocates primary GPU structural tensors once to guarantee pointer safety."""
        d = self.device
        
        # Primary Device Allocations
        cars = wp.zeros((self.num_envs, 7), dtype=float, device=d)
        cars_int = wp.zeros((self.num_envs, 3), dtype=int, device=d)
        car_dr = wp.zeros((self.num_envs, 4), dtype=float, device=d)
        obs = wp.zeros((self.num_envs, OBS_DIM), dtype=float, device=d)
        critic_obs = wp.zeros((self.num_envs, OBS_DIM + 5), dtype=float, device=d)
        rew = wp.zeros(self.num_envs, dtype=float, device=d)
        done = wp.zeros(self.num_envs, dtype=int, device=d)

        angles = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR, dtype=np.float32)
        lidar_buf = wp.array(
            np.column_stack([np.cos(angles), np.sin(angles)]),
            dtype=wp.vec2,
            device=d,
        )
        zero_act = wp.zeros(self.num_envs, dtype=wp.vec2, device=d)
        env_map_ids = wp.zeros(self.num_envs, dtype=int, device=d)

        self.agents = AgentBuffers(
            cars=cars, cars_int=cars_int, car_dr=car_dr, obs=obs,
            critic_obs=critic_obs, rew=rew, done=done, lidar_buf=lidar_buf,
            zero_act=zero_act, env_map_ids=env_map_ids
        )

        # Zero-copy conversions directly overlaying standard PyTorch structures
        obs_buf = wp.to_torch(obs)
        term_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=obs_buf.device)
        trunc_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=obs_buf.device)

        self.views = TorchViews(
            obs_buf=obs_buf,
            critic_obs_buf=wp.to_torch(critic_obs),
            rew_buf=wp.to_torch(rew),
            done_buf=wp.to_torch(done),
            cars_buf=wp.to_torch(cars),
            cars_int_buf=wp.to_torch(cars_int),
            term_buf=term_buf,
            trunc_buf=trunc_buf
        )
        
        self._step_counter = self.views.cars_int_buf[:, 0]
        self._empty_info = {}

    def _initialize_active_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """Packs grid frames tightly and centers the unified system around (0,0)."""
        d = self.device

        max_shape_0 = max(m.dt.T.shape[0] for m in self.maps)
        max_shape_1 = max(m.dt.T.shape[1] for m in self.maps)
        max_n_cl = max(max_shape_0, max_shape_1)

        for m in self.maps:
            max_n_cl = max(max_n_cl, len(m.centerline))

        dt_np = np.zeros((self.num_maps, max_shape_0, max_shape_1), dtype=np.float32)
        lut_np = np.zeros((self.num_maps, max_shape_0, max_shape_1), dtype=np.int32)
        cl_np = np.zeros((self.num_maps, max_n_cl, 3), dtype=np.float32)

        n_cl_list, origins_list, res_list, mh_f_list = [], [], [], []

        sorted_map_indices = sorted(
            range(self.num_maps),
            key=lambda i: max(self.maps[i].wall_width, self.maps[i].wall_length),
            reverse=True,
        )

        free_rects = [[0.0, 0.0, INITIAL_PACKING_BOX_SIZE, INITIAL_PACKING_BOX_SIZE]]
        map_raw_offsets = {}
        current_max_x, current_max_y = 0.0, 0.0

        # --- 2D BIN PACKING ROUTINE ---
        for idx in sorted_map_indices:
            m = self.maps[idx]
            w_box = float(m.wall_width) + MAP_GAP_MARGIN
            l_box = float(m.wall_length) + MAP_GAP_MARGIN

            best_rect_idx = -1
            best_score = float("inf")
            best_x, best_y = 0.0, 0.0

            for r_idx, rect in enumerate(free_rects):
                rx, ry, rw, rh = rect
                if rw >= w_box and rh >= l_box:
                    potential_max_x = max(current_max_x, rx + w_box)
                    potential_max_y = max(current_max_y, ry + l_box)
                    score = max(potential_max_x, potential_max_y)

                    if score < best_score:
                        best_score = score
                        best_x, best_y = rx, ry
                        best_rect_idx = r_idx
                    elif score == best_score:
                        if (potential_max_x * potential_max_y) < (
                            max(current_max_x, best_x + w_box) * max(current_max_y, best_y + l_box)
                        ):
                            best_x, best_y = rx, ry
                            best_rect_idx = r_idx

            if best_rect_idx == -1:
                raise RuntimeError(f"Failed to pack map {m.path_name}.")

            placed_x, placed_y = best_x, best_y
            current_max_x = max(current_max_x, placed_x + w_box)
            current_max_y = max(current_max_y, placed_y + l_box)

            map_raw_offsets[idx] = (placed_x + (m.wall_width / 2.0), placed_y + (m.wall_length / 2.0))

            new_free_rects = []
            rect_to_remove = free_rects[best_rect_idx]

            for rect in free_rects:
                if rect == rect_to_remove:
                    rx, ry, rw, rh = rect
                    if ry + l_box < ry + rh:
                        new_free_rects.append([rx, ry + l_box, rw, rh - l_box])
                    if rx + w_box < rx + rw:
                        new_free_rects.append([rx + w_box, ry, rw - w_box, l_box])
                else:
                    rx, ry, rw, rh = rect
                    px2, py2 = placed_x + w_box, placed_y + l_box
                    if not (rx >= px2 or rx + rw <= placed_x or ry >= py2 or ry + rh <= placed_y):
                        if placed_x > rx:
                            new_free_rects.append([rx, ry, placed_x - rx, rh])
                        if px2 < rx + rw:
                            new_free_rects.append([px2, ry, (rx + rw) - px2, rh])
                        if placed_y > ry:
                            new_free_rects.append([rx, ry, rw, placed_y - ry])
                        if py2 < ry + rh:
                            new_free_rects.append([rx, py2, rw, (ry + rh) - py2])
                    else:
                        new_free_rects.append(rect)

            free_rects = [
                r for r in new_free_rects
                if r[2] > 0.0 and r[3] > 0.0 and not any(
                    r != other and other[0] <= r[0] and other[1] <= r[1] and
                    other[0] + other[2] >= r[0] + r[2] and other[1] + other[3] >= r[1] + r[3]
                    for other in new_free_rects
                )
            ]

        # --- COORDINATE CENTERING & HEIGHT EXTRACTION ---
        global_center_x = current_max_x / 2.0
        global_center_y = current_max_y / 2.0

        final_shifts_x = np.zeros(self.num_maps, dtype=np.float32)
        final_shifts_y = np.zeros(self.num_maps, dtype=np.float32)

        for idx, m in enumerate(self.maps):
            s0, s1 = m.dt.T.shape
            raw_x, raw_y = map_raw_offsets[idx]

            shift_x, shift_y = raw_x - global_center_x, raw_y - global_center_y
            final_shifts_x[idx], final_shifts_y[idx] = shift_x, shift_y

            dt_np[idx, :s0, :s1] = m.dt.T.astype(np.float32)
            lut_np[idx, :s0, :s1] = m.lut.T.astype(np.int32)

            actual_mh = s1
            for h_chk in range(s1):
                reverse_idx = s1 - 1 - h_chk
                if (lut_np[idx, 0, reverse_idx] != 0 or 
                    lut_np[idx, s0 - 1, reverse_idx] != 0 or 
                    lut_np[idx, s0 // 2, reverse_idx] != 0):
                    actual_mh = reverse_idx + 1
                    break

            mh_f_list.append(float(actual_mh) - 1.0)

            n_cl_curr = len(m.centerline)
            cl_np[idx, :n_cl_curr, 0] = m.centerline[:, 0] + shift_x
            cl_np[idx, :n_cl_curr, 1] = m.centerline[:, 1] + shift_y
            cl_np[idx, :n_cl_curr, 2] = m.angles

            n_cl_list.append(n_cl_curr)
            origins_list.append(wp.vec2(m.ox + shift_x, m.oy + shift_y))
            res_list.append(float(m.res))

        self.floor_square_size = float(max(current_max_x, current_max_y))

        # Pack fields directly into the structured Dataclass
        self.maps_storage = MapBuffers(
            dt_buf=wp.array(dt_np, dtype=float, device=d),
            lut_buf=wp.array(lut_np, dtype=int, device=d),
            centerline_buf=wp.array(cl_np, dtype=wp.vec3, device=d),
            maps_n_cl=wp.array(n_cl_list, dtype=int, device=d),
            maps_origin=wp.array(origins_list, dtype=wp.vec2, device=d),
            maps_res=wp.array(res_list, dtype=float, device=d),
            maps_mh_f=wp.array(mh_f_list, dtype=float, device=d)
        )

        return final_shifts_x, final_shifts_y

    def _shuffle_and_assign_maps(
        self, global_shifts: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Distributes vector batches uniformly across available track layouts."""
        shifts_x, shifts_y = global_shifts
        rng = np.random.default_rng(self.seed + self._call)

        assigned_maps = rng.integers(0, self.num_maps, size=self.num_envs).astype(np.int32)
        self.agents.env_map_ids.assign(assigned_maps)

        cars_np = np.zeros((self.num_envs, 7), dtype=np.float32)
        cars_int_np = np.zeros((self.num_envs, 3), dtype=np.int32)

        for i in range(self.num_envs):
            m_idx = assigned_maps[i]
            m = self.maps[m_idx]
            cl_idx = rng.integers(0, len(m.centerline))

            cars_np[i, 0] = m.centerline[cl_idx, 0] + shifts_x[m_idx]
            cars_np[i, 1] = m.centerline[cl_idx, 1] + shifts_y[m_idx]
            cars_np[i, 4] = m.angles[cl_idx]
            cars_int_np[i, 1] = cl_idx

        self.agents.cars.assign(cars_np)
        self.agents.cars_int.assign(cars_int_np)

        dr_init_np = 1.0 - DR_FRAC + 2.0 * DR_FRAC * rng.random((self.num_envs, 4), dtype=np.float32)
        self.agents.car_dr.assign(dr_init_np)

        self.views.rew_buf.zero_()
        self.views.done_buf.zero_()

    def step(
        self, action: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]
    ]:
        """Steps simulator physics forward using target actions via high-speed tracking arrays."""
        self._launch(wp.from_torch(action.detach().contiguous(), dtype=wp.vec2))
        self._sanitize()

        torch.eq(self.views.done_buf, DONE_TERMINATED, out=self.views.term_buf)
        torch.eq(self.views.done_buf, DONE_TRUNCATED, out=self.views.trunc_buf)

        return (
            self.views.obs_buf,
            self.views.critic_obs_buf,
            self.views.rew_buf,
            self.views.term_buf,
            self.views.trunc_buf,
            self._empty_info,
        )

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Resets agent positions while retaining allocated buffer integrity."""
        self._step_counter.fill_(MAX_STEPS)
        self._launch(self.agents.zero_act)
        self._sanitize()
        self._step_counter.zero_()
        self.views.rew_buf.zero_()
        self.views.done_buf.zero_()
        return self.views.obs_buf, self.views.critic_obs_buf, self._empty_info

    def save_state(self) -> Dict[str, Any]:
        """Extracts cloned state tensor dictionaries for check-pointing validation."""
        return {
            k: getattr(self.views, k).clone()
            for k in ("cars_buf", "cars_int_buf", "obs_buf", "rew_buf", "done_buf")
        } | {
            "car_dr": wp.to_torch(self.agents.car_dr).clone(),
        }

    def restore_state(self, s: Dict[str, Any]) -> None:
        """Overwrites local agent buffer views with data parameters."""
        self.views.cars_buf.copy_(s["cars_buf"])
        self.views.cars_int_buf.copy_(s["cars_int_buf"])
        wp.to_torch(self.agents.car_dr).copy_(s["car_dr"])
        self.views.obs_buf.copy_(s["obs_buf"])
        self.views.rew_buf.copy_(s["rew_buf"])
        self.views.done_buf.copy_(s["done_buf"])

    def _launch(self, act: wp.array) -> None:
        """Launches the primary physics stepping kernel across device dimensions."""
        seed: int = (
            self.seed_base * SEED_MULTIPLIER_BASE + self._call * SEED_MULTIPLIER_CALL
        ) & SEED_MASK
        
        wp.launch(
            step_kernel,
            dim=self.num_envs,
            inputs=[
                act,
                self.agents.obs,
                self.agents.critic_obs,
                self.agents.rew,
                self.agents.done,
                self.agents.cars,
                self.agents.cars_int,
                self.agents.car_dr,
                self.maps_storage.centerline_buf,
                self.maps_storage.dt_buf,
                self.maps_storage.lut_buf,
                self.maps_storage.maps_origin,
                self.maps_storage.maps_res,
                self.maps_storage.maps_n_cl,
                self.maps_storage.maps_mh_f,
                self.agents.env_map_ids,
                self.agents.lidar_buf,
                int(seed),
            ],
        )
        self._call += 1
        # Synchronize call removed. Stream operations pipeline automatically.

    def _sanitize(self) -> None:
        """Clears out invalid NaN values across observations directly on VRAM blocks."""
        torch.nan_to_num_(self.views.obs_buf, nan=0.0, posinf=LIDAR_RANGE, neginf=0.0)
        torch.nan_to_num_(self.views.critic_obs_buf, nan=0.0, posinf=LIDAR_RANGE, neginf=0.0)
        torch.nan_to_num_(self.views.cars_buf, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.views.rew_buf, nan=0.0, posinf=0.0, neginf=0.0)