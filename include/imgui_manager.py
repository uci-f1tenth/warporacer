from typing import TYPE_CHECKING  # Forward declaration
from pathlib import Path

import warp as wp
import numpy as np
from imgui_bundle import imgui
from imgui_bundle.python_backends import pyglet_backend

if TYPE_CHECKING:
    from include.environment import Environment

class ImGuiManager:
    """
    Manages the ImGui overlay context attached to the primary Warp OpenGL renderer.
    Handles mouse/keyboard input routing, DPI scaling, and real-time dashboard rendering.
    """
    # Note: ImGUI coordinates map from Top-Left, while Pyglet matrices track from Bottom-Left.
    # High-DPI monitors require careful explicit scaling adjustments to preserve point precision.
    def __init__(self, renderer: wp.render.OpenGLRenderer, env: "Environment"):
        imgui.create_context()
        self.renderer = renderer
        self.env = env

        # Bootstrap the Pyglet window backend but block its default event intercepts
        self.impl = pyglet_backend.create_renderer(self.renderer.window, attach_callbacks=False)
        self.impl.on_mouse_motion = self.on_mouse_motion
        self.impl.on_mouse_drag = self.on_mouse_drag
        self.impl._attach_callbacks(self.renderer.window)

        # Append customized key event filters directly over the running window handler stack
        self.renderer.window.push_handlers(on_key_press=self._on_key_press)

    def on_mouse_motion(self, x, y, dx, dy):
        """Converts window pointer coordinate steps while matching current display DPI configurations."""
        ratio = self.renderer.window.get_pixel_ratio()
        self.impl.io.add_mouse_pos_event(x / ratio, self.impl.io.display_size.y - (y / ratio))
    
    def on_mouse_drag(self, x, y, dx, dy, button, modifiers):
        """Routes click-and-drag parameters to preserve drag interaction mapping inside window elements."""
        self.impl._on_mouse_button(button, True)
        self.on_mouse_motion(x, y, dx, dy)
        return self.impl.io.want_capture_mouse
    
    def _on_key_press(self, symbol, modifiers):
        """Protects active window text items by stealing keyboard focus from simulation controllers."""
        return self.impl.io.want_capture_keyboard

    def _render_frame(self):
        """Assembles, updates bounds, and passes current interface drawing lists down to the canvas loop."""
        io = imgui.get_io()
        ratio = self.renderer.window.get_pixel_ratio()
        io.display_size = self.renderer.screen_width / ratio, self.renderer.screen_height / ratio

        self.impl.process_inputs()
        imgui.new_frame()

        # Build HUD data blocks
        self._draw_ui()

        imgui.render()
        self.impl.render(imgui.get_draw_data())

    def _draw_ui(self):
        """Constructs the primary dashboard window and routes layout to specific category tabs."""
        # Lock the window to the top left with a slight margin
        imgui.set_next_window_pos(imgui.ImVec2(20, 20), imgui.Cond_.first_use_ever)
        
        # Auto-resize ensures the window snaps to fit the active tab's contents
        window_flags = imgui.WindowFlags_.always_auto_resize
        
        imgui.begin("Simulation Dashboard", flags=window_flags)

        # Initialize the Tab Bar System
        if imgui.begin_tab_bar("MainTabBar"):
            
            # Tab 1: Map & Environment Controls
            if imgui.begin_tab_item("Environment")[0]:
                self._draw_environment_tab()
                imgui.end_tab_item()
                
            # Tab 2: Live Agent Telemetry
            if imgui.begin_tab_item("Telemetry")[0]:
                self._draw_telemetry_tab()
                imgui.end_tab_item()
                
            imgui.end_tab_bar()

        imgui.end()

    def _draw_environment_tab(self):
        """Handles track selection, map hot-swapping, and global environment statistics."""
        imgui.text("--- Global Status ---")
        imgui.text(f"Active Vehicles: {self.env.num_envs}")
        imgui.text(f"Simulation Step: {self.env._call}")
        imgui.text(f"Current Track Index: {self.env.current_map_idx}")
        imgui.text(f"Active Track File: {self.env.map.path_name}")
        
        imgui.separator()
        imgui.spacing()

        # Simple Random Loop Cycle Button trigger
        if imgui.button("Random Map Swap"):
            # Environment updates itself AND updates visuals synchronously behind the scenes
            self.env.cycle_next_map(randomize=True)
            
        imgui.spacing()
        imgui.separator()
        
        imgui.text("Direct Track Selection:")
        imgui.spacing()
        
        # Procedurally print explicit direct selections for every map discovered on device
        for idx, map_path in enumerate(self.env.available_maps):
            # Visually highlight the currently active track selection item
            is_active = (idx == self.env.current_map_idx)
            if is_active:
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.2, 1.0, 0.2, 1.0)) # Bright Green tint
                
            if imgui.selectable(f"Track [{idx}]: {map_path.name}", is_active)[0]:
                if not is_active:
                    self.env.load_map_by_index(idx)
                    
            if is_active:
                imgui.pop_style_color()

    def _draw_telemetry_tab(self):
        """Displays real-time kinematic data and sensor arrays for a target agent."""
        imgui.text("--- Agent 0 Telemetry ---")
        
        # Safety check: Ensure buffers exist before attempting CPU transfer
        if hasattr(self.env, 'cars_buf') and self.env.cars_buf is not None:
            
            # NOTE: We only pull agent 0 to the CPU to prevent massive frame drops.
            # Pulling thousands of agents to the CPU every frame for UI will destroy performance.
            car_state = self.env.cars_buf[0].cpu().numpy()
            car_reward = self.env.rew_buf[0].cpu().numpy()
            
            car_x = car_state[0]
            car_y = car_state[1]
            car_steer = car_state[2]
            car_vel = car_state[3]
            car_yaw = car_state[4]

            # Display Kinematics
            imgui.text(f"Position X: {car_x:.3f} m")
            imgui.text(f"Position Y: {car_y:.3f} m")
            imgui.text(f"Heading (Yaw): {np.degrees(car_yaw):.2f} deg")
            imgui.text(f"Velocity: {car_vel:.3f} m/s")
            imgui.text(f"Steering Angle: {np.degrees(car_steer):.2f} deg")
            imgui.text(f"Reward: {car_reward:.3f}")
            
            imgui.separator()
            
            # Display localized sensor/lidar data if available
            if hasattr(self.env, 'obs_buf') and self.env.obs_buf is not None:
                obs_data = self.env.obs_buf[0].cpu().numpy()
                # Assuming LiDAR distance data starts at index 3 based on standard F1TENTH mappings
                front_lidar = obs_data[len(obs_data)//2 + 3] if len(obs_data) > 6 else 0.0
                imgui.text(f"Center LiDAR Ray: {front_lidar:.3f} m")
        else:
            imgui.text_colored("Waiting for agent initialization...", imgui.ImVec4(1.0, 1.0, 0.0, 1.0))

    def shutdown(self):
        """Safely breaks down active UI contexts and releases open backend window pointers."""
        self.impl.shutdown()