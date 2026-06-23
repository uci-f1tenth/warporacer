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
    maps: list[Map]
    num_maps: int
    switch_map_iter: int
    vs: Optional['Visuals']
    
    # Persistent 3D/2D Batched Map Buffers (Stored on GPU VRAM)
    dt_buf: wp.array          # Shape: (num_maps, max_w, max_h)
    lut_buf: wp.array         # Shape: (num_maps, max_w, max_h)
    centerline_buf: wp.array  # Shape: (num_maps, max_n_cl) of wp.vec3
    cars: wp.array
    cars_int: wp.array
    car_dr: wp.array
    obs: wp.array
    critic_obs: wp.array
    rew: wp.array
    done: wp.array
    lidar_buf: wp.array
    _zero_act: wp.array
    
    # GPU-Side Track Metadata Arrays
    maps_origin: wp.array     # Shape: (num_maps) of wp.vec2
    maps_res: wp.array        # Shape: (num_maps) of float
    maps_n_cl: wp.array       # Shape: (num_maps) of int
    maps_look_step: wp.array   # Shape: (num_maps) of int
    floor_square_size: float

    # Environment-to-Track Routing Buffer
    env_map_ids: wp.array     # Shape: (num_envs) of int
    
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

    def __init__(self, maps_dir: Path, num_envs: int, seed: int, target_device: wp.Device, live_viewer: bool, max_active_maps: int = 4):
        self.num_envs = num_envs
        self.seed = seed
        self.seed_base = seed  
        self._call = 0         
        self.device = target_device
        
        self.max_active_maps = max_active_maps
        self.floor_square_size = 0.0
        self.available_maps = []
        
        # 1. Define vs as None BEFORE triggering any rotations
        self.vs = None 
        
        self._initialize_map_library(maps_dir)
        self._allocate_persistent_buffers()
        
        # 2. Trigger the first map load (will safely pass the 'if self.vs is not None' check now)
        self.trigger_map_rotation()
        
        # 3. Initialize the actual visualizer if requested
        # (Make sure you removed the redundant 'self.vs = None' that used to be right above this)
        if live_viewer:
            from include.visuals import Visuals
            self.vs = Visuals(self)
            
    def trigger_map_rotation(self) -> None:
        """Called externally by the training loop to swap the active map pool."""
        # Step the internal seed forward so we don't pick the same maps forever
        self._call += 1 
        rng = np.random.default_rng(self.seed + self._call)
        
        # Determine how many maps to load (cannot exceed available maps)
        sample_size = min(self.max_active_maps, len(self.available_maps))
        
        # Sample unique map paths
        chosen_paths = rng.choice(self.available_maps, size=sample_size, replace=False)
        
        # Parse only the chosen subset into memory
        self.maps = [Map(p) for p in chosen_paths]
        self.num_maps = len(self.maps)
        
        # Rebuild GPU buffers for the new subset and snap agents to them
        self._initialize_active_maps()
        self._shuffle_and_assign_maps()

        # Add this to the very bottom:
        if self.vs is not None:
            self.vs.refresh_maps()

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
        self.critic_obs = wp.zeros((self.num_envs, OBS_DIM + 5), dtype=float, device=d)
        self.rew = wp.zeros(self.num_envs, dtype=float, device=d)
        self.done = wp.zeros(self.num_envs, dtype=int, device=d)

        # Mirror permanent tensor views onto persistent memory footprints
        self.obs_buf = wp.to_torch(self.obs)
        self.critic_obs_buf = wp.to_torch(self.critic_obs)
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
        self.env_map_ids = wp.zeros(self.num_envs, dtype=int, device=d)

    def _initialize_active_maps(self) -> None:
        """Pads and bakes the CURRENT ACTIVE subset of map structures into static GPU memory."""
        d = self.device
        
        # Identify bounding constraint limits across active tracks
        max_shape_0 = max(m.dt.T.shape[0] for m in self.maps)
        max_shape_1 = max(m.dt.T.shape[1] for m in self.maps)
        max_n_cl = max(len(m.centerline) for m in self.maps)
        
        # Build host-side allocation pads
        dt_np = np.zeros((self.num_maps, max_shape_0, max_shape_1), dtype=np.float32)
        lut_np = np.zeros((self.num_maps, max_shape_0, max_shape_1), dtype=np.int32)
        cl_np = np.zeros((self.num_maps, max_n_cl, 3), dtype=np.float32)
        
        n_cl_list, origins_list, res_list, look_step_list = [], [], [], []
        
        # --- HIGH-DENSITY SQUARE BOUNDING PACKING MATH (METERS) ---
        GAP_MARGIN = 1.0  # 1-meter safety buffer between tracks
        
        # Sort maps by largest dimension descending to place big constraints first
        sorted_map_indices = sorted(
            range(self.num_maps), 
            key=lambda i: max(self.maps[i].wall_width, self.maps[i].wall_length), 
            reverse=True
        )
        
        # Open rectangular tracking nodes: [x, y, width, height]
        free_rects = [[0.0, 0.0, 10000.0, 10000.0]]
        map_offsets = {}
        
        # Tracks the current global bounding box of all placed items
        current_max_x = 0.0
        current_max_y = 0.0
        
        for idx in sorted_map_indices:
            m = self.maps[idx]
            
            w_box = float(m.wall_width) + GAP_MARGIN
            l_box = float(m.wall_length) + GAP_MARGIN
            
            best_rect_idx = -1
            best_score = float('inf')  # Penalizes options that stretch out the layout
            best_x, best_y = 0.0, 0.0
            
            for r_idx, rect in enumerate(free_rects):
                rx, ry, rw, rh = rect
                if rw >= w_box and rh >= l_box:
                    # Evaluate what the new global boundaries would look like if placed here
                    potential_max_x = max(current_max_x, rx + w_box)
                    potential_max_y = max(current_max_y, ry + l_box)
                    
                    # The score is the maximum edge size. This heavily penalizes long rectangular growth.
                    score = max(potential_max_x, potential_max_y)
                    
                    # Choose the slot that keeps the overall layout bounding box smallest and squarest
                    if score < best_score:
                        best_score = score
                        best_x = rx
                        best_y = ry
                        best_rect_idx = r_idx
                    elif score == best_score:
                        # Tie-breaker: choose the position that minimizes the area footprint
                        if potential_max_x * potential_max_y < (max(current_max_x, best_x + w_box) * max(current_max_y, best_y + l_box)):
                            best_x = rx
                            best_y = ry
                            best_rect_idx = r_idx
            
            if best_rect_idx == -1:
                raise RuntimeError(f"[Env Error] Failed to pack map {m.path_name} symmetrically.")
            
            placed_x = best_x
            placed_y = best_y
            
            # Update the verified global outer boundaries
            current_max_x = max(current_max_x, placed_x + w_box)
            current_max_y = max(current_max_y, placed_y + l_box)
            
            # Map tracking center conversion logic remains synchronized
            shift_x = placed_x + (m.wall_width / 2.0)
            shift_y = placed_y + (m.wall_length / 2.0)
            map_offsets[idx] = (shift_x, shift_y)
            
            # Subdivide chosen space to keep the packing loop completely clean
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
                        
            free_rects = []
            for r in new_free_rects:
                if r[2] > 0.0 and r[3] > 0.0:
                    if not any(r != other and other[0] <= r[0] and other[1] <= r[1] and 
                               other[0]+other[2] >= r[0]+r[2] and other[1]+other[3] >= r[1]+r[3] 
                               for other in new_free_rects):
                        free_rects.append(r)
        
        # --- BAKE SHIFTED DATA INTO METADATA TENSORS ---
        for idx, m in enumerate(self.maps):
            s0, s1 = m.dt.T.shape    
            shift_x, shift_y = map_offsets[idx]
            
            dt_np[idx, :s0, :s1] = m.dt.T.astype(np.float32)
            lut_np[idx, :s0, :s1] = m.lut.T.astype(np.int32)
            
            n_cl_curr = len(m.centerline)
            cl_np[idx, :n_cl_curr, 0] = m.centerline[:, 0] + shift_x
            cl_np[idx, :n_cl_curr, 1] = m.centerline[:, 1] + shift_y
            cl_np[idx, :n_cl_curr, 2] = m.angles
            
            n_cl_list.append(n_cl_curr)
            
            origins_list.append(wp.vec2(m.ox + shift_x, m.oy + shift_y))
            res_list.append(float(m.res))
            look_step_list.append(int(m.look_step))

        # Assign the tightest outer square bounding value encompassing all tracks cleanly
        self.floor_square_size = float(max(current_max_x, current_max_y))
            
        # Permanent GPU Memory Commit
        self.dt_buf = wp.array(dt_np, dtype=float, device=d)
        self.lut_buf = wp.array(lut_np, dtype=int, device=d)
        self.centerline_buf = wp.array(cl_np, dtype=wp.vec3, device=d)
        
        self.maps_n_cl = wp.array(n_cl_list, dtype=int, device=d)
        self.maps_origin = wp.array(origins_list, dtype=wp.vec2, device=d)
        self.maps_res = wp.array(res_list, dtype=float, device=d)
        self.maps_look_step = wp.array(look_step_list, dtype=int, device=d)

    def _shuffle_and_assign_maps(self) -> None:
        """Distributes vector batches across track profiles and switches start configurations."""
        rng = np.random.default_rng(self.seed + self._call)
        
        # Uniform random distribution of map selections to environments
        assigned_maps = rng.integers(0, self.num_maps, size=self.num_envs).astype(np.int32)
        self.env_map_ids.assign(assigned_maps)
        
        cars_np = np.zeros((self.num_envs, 7), dtype=np.float32)
        cars_int_np = np.zeros((self.num_envs, 3), dtype=np.int32)
        
        for i in range(self.num_envs):
            m_idx = assigned_maps[i]
            m = self.maps[m_idx]
            cl_idx = rng.integers(0, len(m.centerline))
            
            # Reposition vehicles directly on their target track's starting nodes
            cars_np[i, 0] = m.centerline[cl_idx, 0]
            cars_np[i, 1] = m.centerline[cl_idx, 1]
            cars_np[i, 4] = m.angles[cl_idx]
            cars_int_np[i, 1] = cl_idx
            
        self.cars.assign(cars_np)
        self.cars_int.assign(cars_int_np)
        
        dr_init_np = (1.0 - DR_FRAC + 2.0 * DR_FRAC * rng.random((self.num_envs, 4), dtype=np.float32))
        self.car_dr.assign(dr_init_np)
        
        self.rew_buf.zero_()
        self.done_buf.zero_()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        self._launch(wp.from_torch(action.detach().contiguous(), dtype=wp.vec2))
        self._sanitize()
        
        torch.eq(self.done_buf, DONE_TERMINATED, out=self.term_buf)
        torch.eq(self.done_buf, DONE_TRUNCATED, out=self.trunc_buf)

        # Removed internal self.switch_map_iter check

        return (
            self.obs_buf, self.critic_obs_buf, self.rew_buf,
            self.term_buf, self.trunc_buf, self._empty_info,
        )
    
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        self._step_counter.fill_(MAX_STEPS)
        self._launch(self._zero_act)
        self._sanitize()
        self._step_counter.zero_()
        self.rew_buf.zero_()
        self.done_buf.zero_()
        return self.obs_buf, self.critic_obs_buf, self._empty_info

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
                act, self.obs, self.critic_obs, self.rew, self.done, self.cars, self.cars_int, self.car_dr,
                self.centerline_buf, self.dt_buf, self.lut_buf, self.maps_origin, self.maps_res,
                self.maps_n_cl, self.maps_look_step, self.env_map_ids, self.lidar_buf, int(seed),
            ],
        )
        self._call += 1
        wp.synchronize_device()

    def _sanitize(self) -> None:
        torch.nan_to_num_(self.obs_buf, nan=0.0, posinf=LIDAR_RANGE, neginf=0.0)
        torch.nan_to_num_(self.critic_obs_buf, nan=0.0, posinf=LIDAR_RANGE, neginf=0.0)
        torch.nan_to_num_(self.cars_buf, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.rew_buf, nan=0.0, posinf=0.0, neginf=0.0)