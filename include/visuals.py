"""Visuals library handling hardware-accelerated OpenGL rendering and viewport logic."""

import time
from typing import TYPE_CHECKING

import numpy as np
import pyglet
from pyglet.window import key
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation as R
import torch
import warp as wp
import warp.render

from include.constants import ACT_DIM, DT, LENGTH, LF, NUM_LIDAR, WIDTH
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

    env: "Environment"
    renderer: warp.render.OpenGLRenderer
    imgui_manager: ImGuiManager
    key_handler: key.KeyStateHandler
    initialized_all_agents: bool
    lidar_hit_points: wp.array
    last_render_time: float

    def __init__(self, env: "Environment") -> None:
        """Initializes the visual context, hardware bindings, and UI overlays."""
        self.env = env

        # Log hardware device mapping states to verify hardware acceleration context
        print(f"Warp Device: {wp.get_device().name}")
        print(f"Pyglet Device: {pyglet.gl.gl_info.get_renderer()}")

        # Rendering configuration constants
        screen_width = 1280
        screen_height = 720
        near_plane = 0.1
        far_plane = 1000.0
        bg_color = (0.0, 0.0, 0.0)

        # Instantiate persistent hardware-accelerated rendering window pipeline
        self.renderer = warp.render.OpenGLRenderer(
            title="Warp from Thaumcraft",
            screen_width=screen_width,
            screen_height=screen_height,
            near_plane=near_plane,
            far_plane=far_plane,
            up_axis="Z",
            background_color=bg_color,
            draw_grid=False,
            draw_axis=False,
            draw_sky=False,
            device=env.device,
        )

        self.renderer._camera_speed = 1.0

        # Mount HUD overlay layer management system using OpenGL callbacks
        self.imgui_manager = ImGuiManager(self.renderer, env)
        self.renderer.render_2d_callbacks.append(self.imgui_manager._render_frame)

        # Attach hardware keyboard tracking handler loops to capture manual driving inputs
        self.key_handler = key.KeyStateHandler()
        self.renderer.window.push_handlers(self.key_handler)

        # State check initialization flags for tracking persistent multi-agent array structures
        self.initialized_all_agents = False
        self.lidar_hit_points = wp.zeros(NUM_LIDAR, dtype=wp.vec3, device=self.env.device)

        self.refresh_maps()

    def refresh_maps(self) -> None:
        """Flushes and re-registers all active map geometries directly inside the running window."""
        self.initialized_all_agents = False

        # Clear existing primitive shapes and invalidate active GPU memory bindings
        self.renderer.clear()

        # Regenerate fresh structural static layers using the environment's active map list
        self._setup_map()

        # Pre-allocate zero-index components for independent single-car validation
        self._setup_dynamic_objects()

    def interactive_render_loop(self) -> None:
        """While loop for rendering, must be last! Handles input collection and simulation steps."""
        user_actions = torch.zeros(
            (self.env.num_envs, ACT_DIM), device=self.env.views.obs_buf.device
        )

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
        """Renders a single unified square floor plane bounding all packed track structures."""
        floor_size = self.env.floor_square_size
        center_coord = floor_size / 2.0
        floor_color = (0.13, 0.13, 0.13)

        self.renderer.render_plane(
            name="unified_square_ground",
            pos=[0.0, 0.0, 0.0],
            rot=np.array(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2.0)),
            width=center_coord,
            length=center_coord,
            color=floor_color,
        )

        # Iterate through and render the standalone visual elements for each map
        for idx, single_map in enumerate(self.env.maps):
            self._setup_single_map_walls(idx, single_map)
            self._setup_single_map_centerline(idx, single_map)

    def _setup_single_map_walls(self, idx: int, m: Map) -> None:
        """Extracts track boundaries to plot wall point clouds using 2D grid coordinates."""
        dilated_free = binary_dilation(m.free)
        boundary_mask = dilated_free & ~m.free
        boundary_pixels = np.argwhere(boundary_mask)

        rows = boundary_pixels[:, 0]
        cols = boundary_pixels[:, 1]

        # Extract the exact unique grid offsets processed by the physics environment
        env_origin = self.env.maps_storage.maps_origin.numpy()[idx]
        shifted_ox = env_origin[0]
        shifted_oy = env_origin[1]

        # Compute physical world positions with full 2D layout alignment
        wall_x = shifted_ox + cols * m.res
        wall_y = shifted_oy + (m.h - 1 - rows) * m.res

        wall_z = 0.05
        wall_radius = 0.06
        wall_color = (0.8, 0.2, 0.2)

        num_wall_points = len(boundary_pixels)
        vertices = np.zeros((num_wall_points, 3), dtype=np.float32)
        vertices[:, 0] = wall_x
        vertices[:, 1] = wall_y
        vertices[:, 2] = wall_z

        self.renderer.render_points(
            name=f"physics_walls_{idx}",
            points=vertices,
            colors=wall_color,
            radius=wall_radius,
        )

    def _setup_single_map_centerline(self, idx: int, m: Map) -> None:
        """Draws spline guide lines down the physical midpoint coordinates of each layout."""
        cl_data = self.env.maps_storage.centerline_buf.numpy()[idx]
        n_cl = self.env.maps_storage.maps_n_cl.numpy()[idx]

        cl_z = 0.04
        cl_radius = 0.04
        cl_color = (0.2, 0.7, 0.2)

        vertices = np.zeros((n_cl, 3), dtype=np.float32)
        vertices[:, 0] = cl_data[:n_cl, 0]
        vertices[:, 1] = cl_data[:n_cl, 1]
        vertices[:, 2] = cl_z

        self.renderer.render_points(
            name=f"center_line_{idx}",
            points=vertices,
            colors=cl_color,
            radius=cl_radius,
        )

    def _setup_dynamic_objects(self) -> None:
        """Allocates baseline mesh structures inside the rendering frame for primary validation."""
        car_state = self.env.views.cars_buf[0].cpu().numpy()
        car_x, car_y, car_psi = car_state[0], car_state[1], car_state[4]
        car_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(car_psi))

        car_z = 0.15
        car_height = 0.1
        car_color = (1.0, 1.0, 0.0)

        self.renderer.render_box(
            name="car_0",
            pos=[car_x, car_y, car_z],
            rot=np.array(car_rot),
            extents=[LENGTH / 2.0, WIDTH / 2.0, car_height],
            color=car_color,
        )

    def _render_all_agents(self) -> None:
        """Handles structural allocation and updates for massive parallel agent swarms."""
        agent_count_divider = 256
        car_states = self.env.views.cars_buf.cpu().numpy()[::agent_count_divider]
        num_cars_to_render = len(car_states)
        
        car_z = 0.15
        car_height = 0.1

        if not self.initialized_all_agents:
            for i in range(num_cars_to_render):
                percent = float(i / num_cars_to_render)
                car_x, car_y, car_psi = car_states[i, 0], car_states[i, 1], car_states[i, 4]
                car_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(car_psi))

                car_name = f"car_{i}"

                # Allocate concrete bounding box parameters into active renderer layout lists
                self.renderer.render_box(
                    name=car_name,
                    pos=[car_x, car_y, car_z],
                    rot=np.array(car_rot),
                    extents=[LENGTH / 2.0, WIDTH / 2.0, car_height],
                )

                # Assign distinct tracking gradients to distinguish vehicles across parallel horizons
                car_color = (1.0 - percent, 1.0 - percent, percent)
                self.renderer.update_shape_instance(
                    name=car_name, color1=car_color, color2=car_color
                )

            self.initialized_all_agents = True

        # Extract current physical coordinate structures for the active rendering subset
        xs = car_states[:, 0]
        ys = car_states[:, 1]
        psis = car_states[:, 4]

        # Vectorize spatial orientation transformations to avoid CPU bottleneck constraints
        angles = np.zeros((num_cars_to_render, 3))
        angles[:, 2] = psis

        # Convert raw yaw values into uniform quaternion blocks
        quats = R.from_euler("xyz", angles, degrees=False).as_quat()

        # Execute instanced property writes over the underlying object transformations
        for i in range(num_cars_to_render):
            self.renderer.update_shape_instance(
                name=f"car_{i}",
                pos=[xs[i], ys[i], car_z],
                rot=np.array(quats[i]),
            )

    def _render_user_car(self) -> None:
        """Updates spatial transformations for individual user-controlled vehicle models."""
        car_state = self.env.views.cars_buf[0].cpu().numpy()
        car_x, car_y, car_psi = car_state[0], car_state[1], car_state[4]
        car_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(car_psi))
        car_z = 0.15

        self.renderer.update_shape_instance(
            name="car_0", pos=[car_x, car_y, car_z], rot=np.array(car_rot)
        )

    def _render_user_lidar(self) -> None:
        """Dispatches parallel raycasting kernels to evaluate and paint visual sensor reflections."""
        obs_wp = wp.from_torch(self.env.views.obs_buf.contiguous(), dtype=wp.float32)
        cars_wp = wp.from_torch(self.env.views.cars_buf.contiguous(), dtype=wp.float32)

        # Run parallel laser boundary intersection processing on the designated GPU engine
        wp.launch(
            kernel=calc_single_lidar_hits_kernel,
            dim=NUM_LIDAR,
            inputs=[
                obs_wp,
                cars_wp,
                self.env.agents.lidar_buf,
                self.lidar_hit_points,
            ],
            device=self.env.device,
        )

        lidar_radius = 0.05
        lidar_color = (0.0, 1.0, 1.0)

        # Draw computed laser hits directly into the canvas array space
        self.renderer.render_points(
            name="lidar_cloud_0",
            points=self.lidar_hit_points,
            radius=lidar_radius,
            colors=lidar_color,
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
    hit_points: wp.array[wp.vec3],
):
    """Calculates spatial target coordinates for multi-channel onboard laser sensors in parallel."""
    tid = wp.tid()
    ray_idx = tid

    car_x = cars[0, 0]
    car_y = cars[0, 1]
    car_psi = cars[0, 4]

    # Identify relative sensor rig anchor points matching spatial chassis configurations
    lx = car_x + LF * wp.cos(car_psi)
    ly = car_y + LF * wp.sin(car_psi)

    dist = obs[0, 3 + ray_idx]

    # Map relative projection arrays out over unified global path coordinates
    local_dir = lidar_dirs[ray_idx]
    local_angle = wp.atan2(local_dir[1], local_dir[0])
    global_angle = car_psi + local_angle

    hit_x = lx + dist * wp.cos(global_angle)
    hit_y = ly + dist * wp.sin(global_angle)
    
    # Static rendering offset for beam visualization
    hit_z = 0.150

    # Store absolute space coordinate returns back onto tracking arrays
    hit_points[tid] = wp.vec3(hit_x, hit_y, hit_z)