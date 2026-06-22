from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import numpy as np
import torch
import warp as wp

from include.constants import *
from include.map import Map
from include.warped_functions import step_kernel

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from include.visuals import Visuals


class Environment:
    num_envs: int
    seed: int
    seed_base: int
    device: str
    map: Map
    vs: Optional['Visuals']
    
    dt_buf: wp.array
    lut_buf: wp.array
    centerline_buf: wp.array
    cars: wp.array
    cars_int: wp.array
    car_dr: wp.array
    obs: wp.array
    rew: wp.array
    done: wp.array
    lidar_buf: wp.array
    _zero_act: wp.array
    
    map_origin: wp.vec2
    
    obs_buf: torch.Tensor
    rew_buf: torch.Tensor
    done_buf: torch.Tensor
    cars_buf: torch.Tensor
    cars_int_buf: torch.Tensor
    _step_counter: torch.Tensor
    
    term_buf: torch.Tensor
    trunc_buf: torch.Tensor
    _empty_info: Dict[str, Any]
    
    n_cl: int
    look_step: int
    _call: int

    def __init__(self, maps_dir: Path, num_envs: int, seed: int, target_device: wp.Device, live_viewer: bool):
        self.num_envs = num_envs
        self.seed = seed
        self.seed_base = seed  
        self._call = 0         
        self.device = target_device
        
        self.available_maps = []
        self.current_map_idx = 0
        
        self._initialize_map_library(maps_dir)
        
        # Explicit allocations done only ONCE to prevent losing compilation pointer references
        self._allocate_persistent_buffers()
        
        self.load_map_by_index(self.current_map_idx)
        
        self.vs = None
        if live_viewer:
            from include.visuals import Visuals
            self.vs = Visuals(self, self.map)

    def _initialize_map_library(self, maps_dir: Path) -> None:
        resolved_target = Path(maps_dir).resolve()
        if resolved_target.is_file():
            self.available_maps = [resolved_target]
        elif resolved_target.is_dir():
            self.available_maps = sorted(list(resolved_target.glob("*.yaml")), key=lambda p: p.name)
            if not self.available_maps:
                raise FileNotFoundError(f"[Env Error] No .yaml configurations found inside: {resolved_target}")
        else:
            raise FileNotFoundError(f"[Env Error] Provided map path target is invalid: {resolved_target}")

    def _allocate_persistent_buffers(self) -> None:
        """Allocates underlying GPU structural arrays once to ensure memory address integrity."""
        d = self.device
        self.cars = wp.zeros((self.num_envs, 7), dtype=float, device=d)
        self.cars_int = wp.zeros((self.num_envs, 3), dtype=int, device=d)
        self.car_dr = wp.zeros((self.num_envs, 4), dtype=float, device=d)
        self.obs = wp.zeros((self.num_envs, OBS_DIM), dtype=float, device=d)
        self.rew = wp.zeros(self.num_envs, dtype=float, device=d)
        self.done = wp.zeros(self.num_envs, dtype=int, device=d)

        # Mirror permanent tensor views onto persistent memory footprints
        self.obs_buf = wp.to_torch(self.obs)
        self.rew_buf = wp.to_torch(self.rew)
        self.done_buf = wp.to_torch(self.done)
        self.cars_buf = wp.to_torch(self.cars)
        self.cars_int_buf = wp.to_torch(self.cars_int)
        self._step_counter = self.cars_int_buf[:, 0]

        self.term_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.obs_buf.device)
        self.trunc_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.obs_buf.device)
        self._empty_info = {}

        angles = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR, dtype=np.float32)
        self.lidar_buf = wp.array(np.column_stack([np.cos(angles), np.sin(angles)]), dtype=wp.vec2, device=d)
        self._zero_act = wp.zeros(self.num_envs, dtype=wp.vec2, device=d)

    def load_map_by_index(self, idx: int) -> None:
        if not (0 <= idx < len(self.available_maps)):
            raise IndexError(f"[Env Error] Target index {idx} falls outside map library boundary limits.")
            
        self.current_map_idx = idx
        current_map_path = self.available_maps[self.current_map_idx]
        
        print(f"[Environment] Activating track layout [{self.current_map_idx}]: {current_map_path.name}")
        self.load_map(current_map_path, reset_call_count=True)
        
        if hasattr(self, 'vs') and self.vs is not None:
            self.vs.switch_track_layout(self.map)

    def cycle_next_map(self, randomize: bool = False) -> None:
        if len(self.available_maps) <= 1:
            return
        if randomize:
            choices = [i for i in range(len(self.available_maps)) if i != self.current_map_idx]
            next_idx = int(np.random.choice(choices))
        else:
            next_idx = (self.current_map_idx + 1) % len(self.available_maps)
            
        self.load_map_by_index(next_idx)

    def load_map(self, map_path: Path, reset_call_count: bool = False) -> None:
        self.map = Map(map_path)
        if reset_call_count:
            self._call = 0
        self._update_map_dependent_buffers()

    def _update_map_dependent_buffers(self) -> None:
        """In-place updates internal track data arrays to prevent breaking graph pointers."""
        self.look_step = self.map.look_step
        d = self.device

        # Re-initialize track arrays safely
        self.dt_buf = wp.array(self.map.dt.T.astype(np.float32), dtype=float, device=d)
        self.lut_buf = wp.array(self.map.lut.T.astype(np.int32), dtype=int, device=d)
        self.centerline_buf = wp.array(
            np.column_stack([self.map.centerline, self.map.angles]).astype(np.float32),
            dtype=wp.vec3, device=d
        )
        self.n_cl = len(self.map.centerline)
        self.map_origin = wp.vec2(self.map.ox, self.map.oy)

        # Procedurally populate starting line poses safely in-place on the host side
        rng = np.random.default_rng(self.seed)
        idxs = rng.integers(0, self.n_cl, size=self.num_envs)
        
        cars_np = np.zeros((self.num_envs, 7), dtype=np.float32)
        cars_np[:, 0] = self.map.centerline[idxs, 0]
        cars_np[:, 1] = self.map.centerline[idxs, 1]
        cars_np[:, 4] = self.map.angles[idxs]
        
        cars_int_np = np.zeros((self.num_envs, 3), dtype=np.int32)
        cars_int_np[:, 1] = idxs
        
        dr_init_np = (1.0 - DR_FRAC + 2.0 * DR_FRAC * rng.random((self.num_envs, 4), dtype=np.float32))

        # Correct NVIDIA Warp syntax for in-place assignment from NumPy arrays
        self.cars.assign(cars_np)
        self.cars_int.assign(cars_int_np)
        self.car_dr.assign(dr_init_np)

        # Warm-up compile phase execution
        self._launch(self._zero_act)
        self._sanitize()
        self._step_counter.zero_()
        self.rew_buf.zero_()
        self.done_buf.zero_()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        self._launch(wp.from_torch(action.detach().contiguous(), dtype=wp.vec2))
        self._sanitize()
        
        # In-place logical checks on underlying tracking buffers
        torch.eq(self.done_buf, DONE_TERMINATED, out=self.term_buf)
        torch.eq(self.done_buf, DONE_TRUNCATED, out=self.trunc_buf)

        return (
            self.obs_buf,
            self.rew_buf,
            self.term_buf,
            self.trunc_buf,
            self._empty_info,
        )
    
    def reset(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self._step_counter.fill_(MAX_STEPS)
        self._launch(self._zero_act)
        self._sanitize()
        self._step_counter.zero_()
        self.rew_buf.zero_()
        self.done_buf.zero_()
        return self.obs_buf, self._empty_info

    def save_state(self) -> Dict[str, Any]:
        return {
            k: getattr(self, k).clone()
            for k in ("cars_buf", "cars_int_buf", "obs_buf", "rew_buf", "done_buf")
        } | {
            "car_dr": wp.to_torch(self.car_dr).clone(),
        }

    def restore_state(self, s: Dict[str, Any]) -> None:
        self.cars_buf.copy_(s["cars_buf"])
        self.cars_int_buf.copy_(s["cars_int_buf"])
        wp.to_torch(self.car_dr).copy_(s["car_dr"])
        self.obs_buf.copy_(s["obs_buf"])
        self.rew_buf.copy_(s["rew_buf"])
        self.done_buf.copy_(s["done_buf"])

    def _launch(self, act: wp.array) -> None:
        seed: int = (self.seed_base * 2654435761 + self._call * 83492791) & 0x7FFFFFFF
        wp.launch(
            step_kernel,
            dim=self.num_envs,
            inputs=[
                act, self.obs, self.rew, self.done, self.cars, self.cars_int, self.car_dr,
                self.map_origin, self.map.res, self.dt_buf, self.lut_buf, self.centerline_buf,
                self.n_cl, self.look_step, self.lidar_buf, int(seed),
            ],
        )
        self._call += 1
        # Synchronize streams to guarantee that Warp completes before PyTorch modifications run
        wp.synchronize_device()
    
    def _sanitize(self) -> None:
        torch.nan_to_num_(self.obs_buf, nan=0.0, posinf=LIDAR_RANGE, neginf=0.0)
        torch.nan_to_num_(self.cars_buf, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.rew_buf, nan=0.0, posinf=0.0, neginf=0.0)