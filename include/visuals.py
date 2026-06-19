import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pyglet
import torch
from pyglet.window import key
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation as R  # Moved to top-level to avoid per-frame overhead

import warp as wp
import warp.render  # Explicitly require for tracking OpenGLRenderer reference
from include.constants import *
from include.imgui_manager import ImGuiManager
from include.map import Map

if TYPE_CHECKING:
    from include.environment import Environment

class Visuals:
    """
    Manages the high-performance hardware-accelerated OpenGL rendering context.
    Acts as a reactive viewport interface that tracks, renders, and captures user inputs
    for the underlying physics structures contained within the simulation environment.
    """
    def __init__(self, env: "Environment", map_layout: Map):
        self.env = env
        self.map = map_layout
    
        # Log hardware device mapping states to verify hardware acceleration context
        print(f"Warp Device: {wp.get_device().name}")
        print(f"Pyglet Device: {pyglet.gl.gl_info.get_renderer()}")
        
        # Instantiate persistent hardware-accelerated rendering window pipeline
        self.renderer = warp.render.OpenGLRenderer(
            title="Warp from Thaumcraft",
            screen_width=1280,
            screen_height=720,
            near_plane=0.1,
            far_plane=1000,
            up_axis="Z",  # Note: Internal sun directions are fixed inside this instance
            background_color=(0, 0, 0),
            draw_grid=False,
            draw_axis=False,
            draw_sky=False,
            device=wp.get_device()
        )

        # Mount HUD overlay layer management system using OpenGL callbacks
        self.imgui_manager = ImGuiManager(self.renderer, env)
        self.renderer.render_2d_callbacks.append(self.imgui_manager._render_frame)

        # Attach hardware keyboard tracking handler loops to capture manual driving inputs
        self.key_handler = key.KeyStateHandler()
        self.renderer.window.push_handlers(self.key_handler)

        # State check initialization flags for tracking persistent multi-agent array structures
        self.initialized_all_agents = False
        self.lidar_hit_points = wp.zeros(NUM_LIDAR, dtype=wp.vec3, device=self.env.device)

        # Trigger structural registration configurations for the baseline track map layout
        self.switch_track_layout(self.map)

    def switch_track_layout(self, new_map: Map) -> None:
        """Flushes and re-registers map geometries directly inside the running window."""
        self.map = new_map
        self.initialized_all_agents = False
        
        # Clear existing primitive shapes and invalidate active GPU memory bindings
        self.renderer.clear()
        
        # Regenerate fresh structural static layers onto the new layout matrix specifications
        self._setup_map()
        
        # Pre-allocate zero-index components for independent single-car validation
        self._setup_dynamic_objects()

    def interactive_render_loop(self) -> None:
        """While loop for rendering, must be last! Handles input collection and simulation steps."""
        # BUG FIX: Bind exactly to the underlying tensor device to prevent cross-device faults
        user_actions = torch.zeros((self.env.num_envs, ACT_DIM), device=self.env.obs_buf.device)

        self.last_render_time = time.perf_counter()
        
        while self.renderer.is_running():
            current_time = time.perf_counter()
            if current_time - self.last_render_time >= DT:
                self.last_render_time = current_time

                # Extract manual driving inputs from the host keyboard layer
                throttle = float(self.key_handler[key.I] - self.key_handler[key.K])
                steering = float(self.key_handler[key.J] - self.key_handler[key.L])

                user_actions[0, 0] = steering
                user_actions[0, 1] = throttle
                
                # Advance parallel physics engine components by one step
                self.env.step(user_actions)
                self.render()

        self._clear()

    def render(self) -> None:
        """Pushes current state calculations down to visual graphics primitives."""
        sim_time = self.env._call * DT 
        self.renderer.begin_frame(sim_time)
        
        # Direct routing handles individual agents or multi-agent swarm updates smoothly
        if self.env.num_envs > 1:
            self._render_all_agents()
        else:
            self._render_user_car()
            
        self._render_user_lidar()
        self.renderer.end_frame()

    def _setup_map(self) -> None:
        """Bundles procedural operations required to draw a complete track layout."""
        self._setup_map_ground()
        self._setup_map_walls()
        self._setup_map_center_line()

    def _setup_map_ground(self) -> None:
        """Calculates physical matrix scale bounds and draws an un-stretched track canvas plane."""
        physical_width = self.map.w * self.map.res
        physical_length = self.map.h * self.map.res
        
        # Calculate the true physical center of the map image
        center_x = self.map.ox + (physical_width / 2.0)
        center_y = self.map.oy + (physical_length / 2.0)

        # Use the maximum dimension to keep the ground plane a perfect square.
        # This prevents the ground texture grid/squares from stretching on rectangular maps.
        max_extent = max(physical_width, physical_length)

        self.renderer.render_plane(
            name="map_ground",
            pos=[center_x, center_y, 0.0],
            rot=np.array(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2.0)),
            width=max_extent / 2.0,
            length=max_extent / 2.0,
            color=(0.15, 0.15, 0.15)
        )
    
    def _setup_map_walls(self) -> None:
        """Extracts track boundaries using spatial image dilation to plot outer wall point clouds."""
        dilated_free = binary_dilation(self.map.free)
        boundary_mask = dilated_free & ~self.map.free 
        boundary_pixels = np.argwhere(boundary_mask)

        rows = boundary_pixels[:, 0]
        cols = boundary_pixels[:, 1]

        wall_x = self.map.ox + cols * self.map.res
        wall_y = self.map.oy + (self.map.h - 1 - rows) * self.map.res

        num_wall_points = len(boundary_pixels)
        self.wall_vertices = np.zeros((num_wall_points, 3), dtype=np.float32)
        self.wall_vertices[:, 0] = wall_x
        self.wall_vertices[:, 1] = wall_y
        self.wall_vertices[:, 2] = 0.05

        self.renderer.render_points(
            name="physics_walls",
            points=self.wall_vertices,
            colors=(1.0, 0.0, 0.0),
            radius=0.05
        )

    def _setup_map_center_line(self) -> None:
        """Draws spline guide arrays down the physical midpoint coordinates of the track layout."""
        num_points = len(self.map.centerline)
        self.track_vertices = np.zeros((num_points, 3), dtype=np.float32)
        self.track_vertices[:, 0] = self.map.centerline[:, 0]
        self.track_vertices[:, 1] = self.map.centerline[:, 1]
        self.track_vertices[:, 2] = 0.05
        
        self.renderer.render_points(
            name="center_line",
            points=self.track_vertices,
            colors=(0.0, 1.0, 0.0),
            radius=0.05
        )

    def _setup_dynamic_objects(self) -> None:
        """Allocates baseline mesh structures inside the rendering frame for primary validation."""
        car_state = self.env.cars_buf[0].cpu().numpy()
        car_x, car_y, car_psi = car_state[0], car_state[1], car_state[4]
        car_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(car_psi))
        
        self.renderer.render_box(
            name="car_0",
            pos=[car_x, car_y, 0.15], 
            rot=np.array(car_rot),
            extents=[LENGTH / 2.0, WIDTH / 2.0, 0.1],
            color=(1.0, 1.0, 0.0)
        )

    def _render_all_agents(self) -> None:
        """Handles structural allocation, color distribution, and updates for massive parallel agent swarms."""
        if not self.initialized_all_agents:
            all_car_states = self.env.cars_buf.cpu().numpy()
            
            for i in range(self.env.num_envs):
                percent = float(i / self.env.num_envs)
                car_state = all_car_states[i]
                car_x, car_y, car_psi = car_state[0], car_state[1], car_state[4]
                car_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(car_psi))

                car_name = f"car_{i}"

                # Allocate concrete bounding box parameters into active renderer layout lists
                self.renderer.render_box(
                    name=car_name,
                    pos=[car_x, car_y, 0.15], 
                    rot=np.array(car_rot),
                    extents=[LENGTH / 2.0, WIDTH / 2.0, 0.1]
                )

                # Assign distinct tracking gradients to distinguish vehicles across parallel horizons
                car_color = (1.0 - percent, 1.0 - percent, percent)
                self.renderer.update_shape_instance(
                    name=car_name,
                    color1=car_color,
                    color2=car_color
                )
            
            self.initialized_all_agents = True

        # Extract current physical coordinate structures out of shared PyTorch engine arrays
        car_states = self.env.cars_buf.cpu().numpy()
        num_cars = len(car_states)
        
        xs = car_states[:, 0]
        ys = car_states[:, 1]
        psis = car_states[:, 4]

        # Vectorize spatial orientation transformations to avoid CPU bottleneck constraints
        angles = np.zeros((num_cars, 3))
        angles[:, 2] = psis 
        
        # Convert raw yaw values into uniform quaternion blocks using lightning-fast Scipy C calls
        quats = R.from_euler('xyz', angles, degrees=False).as_quat()

        # Execute instanced property writes over the underlying object transformations
        for i in range(num_cars):
            self.renderer.update_shape_instance(
                name=f"car_{i}",
                pos=[xs[i], ys[i], 0.15],
                rot=np.array(quats[i])
            )

    def _render_user_car(self) -> None:
        """Updates spatial transformations for individual user-controlled vehicle models."""
        car_state = self.env.cars_buf[0].cpu().numpy()
        car_x, car_y, car_psi = car_state[0], car_state[1], car_state[4]
        car_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(car_psi))

        self.renderer.update_shape_instance(
            name="car_0",
            pos=[car_x, car_y, 0.15], 
            rot=np.array(car_rot)
        )

    def _render_user_lidar(self) -> None:
        """Dispatches parallel raycasting kernels to evaluate and paint visual sensor reflections."""
        obs_wp = wp.from_torch(self.env.obs_buf.contiguous(), dtype=wp.float32)
        cars_wp = wp.from_torch(self.env.cars_buf.contiguous(), dtype=wp.float32)
        
        # Run parallel laser boundary intersection processing on the designated GPU engine
        wp.launch(
            kernel=calc_single_lidar_hits_kernel,
            dim=NUM_LIDAR,
            inputs=[
                obs_wp,
                cars_wp,
                self.env.lidar_buf,
                self.lidar_hit_points
            ],
            device=self.env.device
        )
        
        # Draw computed laser hits directly into the canvas array space
        self.renderer.render_points(
            name="lidar_cloud_0",
            points=self.lidar_hit_points,
            radius=0.05,
            colors=(0.0, 1.0, 1.0)
        )

    def _clear(self) -> None:
        """Safely tears down active context handles and frees allocated interface wrappers."""
        self.imgui_manager.shutdown()
        self.renderer.clear()


@wp.kernel
def calc_single_lidar_hits_kernel(
    obs: wp.array2d[wp.float32],
    cars: wp.array2d[wp.float32],
    lidar_dirs: wp.array[wp.vec2],
    hit_points: wp.array[wp.vec3]
):
    """Calculates spatial target coordinates for multi-channel onboard laser sensors in parallel."""
    tid = wp.tid()
    ray_idx = tid
    
    car_x = cars[0, 0]
    car_y = cars[0, 1]
    car_psi = cars[0, 4]
    
    # Identify relative sensor rig anchor points matching spatial chassis configurations
    # Note: LF is implicitly captured from include.constants at kernel compile time
    lx = car_x + LF * wp.cos(car_psi)
    ly = car_y + LF * wp.sin(car_psi)
    
    dist = obs[0, 3 + ray_idx]
    
    # Map relative projection arrays out over unified global path coordinates
    local_dir = lidar_dirs[ray_idx]
    local_angle = wp.atan2(local_dir[1], local_dir[0])
    global_angle = car_psi + local_angle
    
    hit_x = lx + dist * wp.cos(global_angle)
    hit_y = ly + dist * wp.sin(global_angle)
    hit_z = 0.150
    
    # Store absolute space coordinate returns back onto tracking arrays
    hit_points[tid] = wp.vec3(hit_x, hit_y, hit_z)