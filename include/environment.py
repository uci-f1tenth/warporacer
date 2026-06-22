from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import numpy as np
import torch
import warp as wp

from include.constants import *
from include.map import Map
from include.warped_functions import step_kernel

from typing import TYPE_CHECKING # Forward declaration
if TYPE_CHECKING:
    from include.visuals import Visuals

# TODO : Enable randomize CW/CCW map directions (if doing idea below, have two versions?)
# TODO : Instead of loading one map at a time, have them all load at once but isolated?

class Environment:
    # Class attribute type annotations for structural linting and memory layout clarity
    num_envs: int
    seed: int
    seed_base: int
    device: str
    map: Map
    vs: Optional['Visuals']
    
    # Native NVIDIA Warp array storage buffers (Resident directly on GPU Device memory)
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
    
    # Pre-allocated structural properties to completely eliminate inside-loop allocations
    map_origin: wp.vec2
    
    # Zero-copy interoperability PyTorch tensor views sharing underlying Warp memory layouts
    obs_buf: torch.Tensor
    rew_buf: torch.Tensor
    done_buf: torch.Tensor
    cars_buf: torch.Tensor
    cars_int_buf: torch.Tensor
    _step_counter: torch.Tensor
    
    # Pre-allocated target optimization masks for high-throughput tensor evaluations
    term_buf: torch.Tensor
    trunc_buf: torch.Tensor
    _empty_info: Dict[str, Any]
    
    # Performance tracking scalars
    n_cl: int
    look_step: int
    _call: int

    def __init__(self, maps_dir: Path, num_envs: int, seed: int, target_device: wp.Device, live_viewer: bool):
        self.num_envs = num_envs
        self.seed = seed
        self.seed_base = seed  # Required for the LCG random seed inside _launch
        self._call = 0         # Required to track kernel step dispatch offsets
        self.device = target_device
        
        # Internally track the map management system
        self.available_maps = []
        self.current_map_idx = 0
        
        # Discover and populate your track library automatically
        self._initialize_map_library(maps_dir)
        
        # Load up your initial baseline map choice (which subsequently initializes the physics buffers)
        self.load_map_by_index(self.current_map_idx)
        
        # Handle lazy visual binding structures for optional rendering pipelines
        self.vs = None
        if live_viewer:
            from include.visuals import Visuals
            self.vs = Visuals(self, self.map)

    def _initialize_map_library(self, maps_dir: Path) -> None:
        """Determines if the target is a single asset or directory and indexes files."""
        resolved_target = Path(maps_dir).resolve()
        
        if resolved_target.is_file():
            # Single-map configuration pipeline: Lock strictly to the provided file
            self.available_maps = [resolved_target]
            self.current_map_idx = 0
        elif resolved_target.is_dir():
            # Multi-map layout directory crawl pipeline: Sort alphabetically for deterministic ordering
            self.available_maps = sorted(list(resolved_target.glob("*.yaml")), key=lambda p: p.name)
            if not self.available_maps:
                raise FileNotFoundError(f"[Env Error] No .yaml configurations found inside: {resolved_target}")
            self.current_map_idx = 0
        else:
            raise FileNotFoundError(f"[Env Error] Provided map path target is invalid: {resolved_target}")

    def load_map_by_index(self, idx: int) -> None:
        """Explicitly sets and activates a track selection based on collection index coordinates."""
        if not (0 <= idx < len(self.available_maps)):
            raise IndexError(f"[Env Error] Target index {idx} falls outside map library boundary limits.")
            
        self.current_map_idx = idx
        current_map_path = self.available_maps[self.current_map_idx]
        
        # Procedurally instantiate the internal Map data representation
        print(f"[Environment] Activating track layout [{self.current_map_idx}]: {current_map_path.name}")
        self.load_map(current_map_path, reset_call_count=True)
        
        # Cascading fallback update notice down to the visual layer graphics buffers if mounted
        if hasattr(self, 'vs') and self.vs is not None:
            self.vs.switch_track_layout(self.map)

    def cycle_next_map(self, randomize: bool = False) -> None:
        """Steps or shuffles map tracks programmatically without external directory re-scans."""
        if len(self.available_maps) <= 1:
            print("[Environment] Single map lock active. Skipping sequence shift request.")
            return

        if randomize:
            # Filter out current index locations to guarantee a distinct layout variation step
            choices = [i for i in range(len(self.available_maps)) if i != self.current_map_idx]
            next_idx = int(np.random.choice(choices))
        else:
            # Increment step sequence loops sequentially, rolling over at the end of the list
            next_idx = (self.current_map_idx + 1) % len(self.available_maps)
            
        self.load_map_by_index(next_idx)

    def load_map(self, map_path: Path, reset_call_count: bool = False) -> None:
        """Dynamically shifts environmental maps and re-allocates structural buffers smoothly."""
        self.map = Map(map_path)
        
        if reset_call_count:
            self._call = 0

        # Discard stale physics references and execute fresh object buffer instantiations
        self._init_cars()

    def _init_cars(self) -> None:
        """Handles physical hardware allocations and maps raw variables into device data layout structures."""
        self.look_step = self.map.look_step
        d: str = self.device

        # Transfer physical track grid properties and structures into persistent Warp device arrays
        self.dt_buf = wp.array(self.map.dt.T.astype(np.float32), dtype=float, device=d)
        self.lut_buf = wp.array(self.map.lut.T.astype(np.int32), dtype=int, device=d)
        self.centerline_buf = wp.array(
            np.column_stack([self.map.centerline, self.map.angles]).astype(np.float32),
            dtype=wp.vec3,
            device=d,
        )
        self.n_cl = len(self.map.centerline)

        # Pre-allocate static map properties to isolate inputs across parallelized kernel tasks
        self.map_origin = wp.vec2(self.map.ox, self.map.oy)

        # Sample uniform random initial spawn positions using an independent host generator seed
        rng: np.random.Generator = np.random.default_rng(self.seed)
        idxs: np.ndarray = rng.integers(0, self.n_cl, size=self.num_envs)
        
        # State vector topology: [x, y, x_vel, y_vel, theta, angular_vel, steer]
        cars: np.ndarray = np.zeros((self.num_envs, 7), dtype=np.float32)
        cars[:, 0] = self.map.centerline[idxs, 0]
        cars[:, 1] = self.map.centerline[idxs, 1]
        cars[:, 4] = self.map.angles[idxs]
        
        # Discrete tracking coordinates: [step_counter, target_centerline_index]
        cars_int: np.ndarray = np.zeros((self.num_envs, 2), dtype=np.int32)
        cars_int[:, 1] = idxs
        
        # Domain Randomization (DR) scaling multipliers to handle heterogeneous asset variance
        dr_init: np.ndarray = (
            1.0 - DR_FRAC + 2.0 * DR_FRAC * rng.random((self.num_envs, 4), dtype=np.float32)
        )

        # Instantiate dedicated Warp buffers strictly on the native hardware compute device
        self.cars = wp.array(cars, dtype=float, device=d)
        self.cars_int = wp.array(cars_int, dtype=int, device=d)
        self.car_dr = wp.array(dr_init, dtype=float, device=d)
        self.obs = wp.zeros((self.num_envs, OBS_DIM), dtype=float, device=d)
        self.rew = wp.zeros(self.num_envs, dtype=float, device=d)
        self.done = wp.zeros(self.num_envs, dtype=int, device=d)

        # Construct zero-copy PyTorch tensor viewpoints into identical raw memory allocations
        self.obs_buf = wp.to_torch(self.obs)
        self.rew_buf = wp.to_torch(self.rew)
        self.done_buf = wp.to_torch(self.done)
        self.cars_buf = wp.to_torch(self.cars)
        self.cars_int_buf = wp.to_torch(self.cars_int)
        self._step_counter = self.cars_int_buf[:, 0]

        # Allocate static truth masks to protect step-loop performance profiles from GC pauses
        self.term_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.obs_buf.device)
        self.trunc_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.obs_buf.device)
        self._empty_info = {}

        # Compute spatial directional unit rays for the multi-channel onboard Lidar configuration
        angles: np.ndarray = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR, dtype=np.float32)
        self.lidar_buf = wp.array(
            np.column_stack([np.cos(angles), np.sin(angles)]),
            dtype=wp.vec2,
            device=d,
        )
        self._zero_act = wp.zeros(self.num_envs, dtype=wp.vec2, device=d)

        # Dispatch warm-up sequence loop to safely force GPU kernel initialization compilation
        self._launch(self._zero_act)
        self._sanitize()
        self._step_counter.zero_()
        self.rew_buf.zero_()
        self.done_buf.zero_()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Maps torch tensor actions over to hardware vector blocks and runs a physics updates step."""
        self._launch(wp.from_torch(action.detach().contiguous(), dtype=wp.vec2))
        self._sanitize()
        
        # Populate pre-allocated outcome masks directly in hardware without causing OS allocations
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
        """Forces full environmental state overrides to bring agents back to baseline configurations."""
        self._step_counter.fill_(MAX_STEPS)
        self._launch(self._zero_act)
        self._sanitize()
        self._step_counter.zero_()
        self.rew_buf.zero_()
        self.done_buf.zero_()
        return self.obs_buf, self._empty_info

    def save_state(self) -> Dict[str, Any]:
        """Creates a standalone deep-copy snapshot dict mapping current environment tensors."""
        return {
            k: getattr(self, k).clone()
            for k in ("cars_buf", "cars_int_buf", "obs_buf", "rew_buf", "done_buf")
        } | {
            "car_dr": wp.to_torch(self.car_dr).clone(),
        }

    def restore_state(self, s: Dict[str, Any]) -> None:
        """In-place writes an upstream state checkpoint back down into hardware device tracks."""
        self.cars_buf.copy_(s["cars_buf"])
        self.cars_int_buf.copy_(s["cars_int_buf"])
        wp.to_torch(self.car_dr).copy_(s["car_dr"])
        self.obs_buf.copy_(s["obs_buf"])
        self.rew_buf.copy_(s["rew_buf"])
        self.done_buf.copy_(s["done_buf"])

    def _launch(self, act: wp.array) -> None:
        """Launches the massive GPU-parallel physics kernel using a Linear Congruential Generator step."""
        # Bitwise mask guarantees the seed stays within standard 32-bit integer limits
        seed: int = (self.seed_base * 2654435761 + self._call * 83492791) & 0x7FFFFFFF
        wp.launch(
            step_kernel,
            dim=self.num_envs,
            inputs=[
                act,
                self.obs,
                self.rew,
                self.done,
                self.cars,
                self.cars_int,
                self.car_dr,
                self.map_origin,
                self.map.res,
                self.dt_buf,
                self.lut_buf,
                self.centerline_buf,
                self.n_cl,
                self.look_step,
                self.lidar_buf,
                int(seed),
            ],
        )
        self._call += 1
    
    def _sanitize(self) -> None:
        """Performs localized repairs across buffer horizons to instantly isolate numerical explosions."""
        torch.nan_to_num_(self.obs_buf, nan=0.0, posinf=LIDAR_RANGE, neginf=0.0)
        torch.nan_to_num_(self.cars_buf, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.rew_buf, nan=0.0, posinf=0.0, neginf=0.0)