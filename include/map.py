import heapq
from collections import deque
from pathlib import Path

import numpy as np
import cv2
from cv2 import IMREAD_GRAYSCALE, imread
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from yaml import safe_load

from include.constants import *


class Map:
    """
    Multiton Map class handling map loading, binarization, boundary detection, 
    centerline extraction, and lookup table generation.
    """
    
    # Class-level cache dictionary to prevent redundant processing of identical maps
    _cache = {}

    def __new__(cls, path: Path):
        """Intercepts instance creation to implement the Multiton/Flyweight caching pattern."""
        abs_path = path.resolve()
        
        if abs_path not in cls._cache:
            # Allocate memory for a new instance if not cached
            instance = super().__new__(cls)
            instance._initialized = False
            cls._cache[abs_path] = instance
            
        return cls._cache[abs_path]

    def __init__(self, path: Path):
        """Initializes the map logic. Bypasses heavy lifting if pulled from cache."""
        # Skip processing if this instance was successfully fetched from the cache
        if getattr(self, "_initialized", False):
            return

        print(f"Processing and caching new map: {path.name}...")

        # 1. Load map metadata and raw image array
        self.path_name = path.name
        self.meta = safe_load(path.read_text())
        self.img_path = path.parent / self.meta["image"]

        self.raw = imread(str(self.img_path), IMREAD_GRAYSCALE)
        if self.raw is None:
            raise FileNotFoundError(f"Could not load image at {self.img_path}")
        
        # 2. Programmatically seal the image border with black pixels (0)
        # Prevents "leaks" in distance transforms and visually closes tightly cropped tracks
        self.raw[0, :] = 0    
        self.raw[-1, :] = 0   
        self.raw[:, 0] = 0    
        self.raw[:, -1] = 0   
        
        # 3. Create boolean map of drivable space and calculate distances to nearest walls
        self.free = self.raw >= OCC_THRESH
        self.dt = distance_transform_edt(self.free)

        # 4. Extract dimensional data and coordinate origin from metadata
        self.ox, self.oy, _ = self.meta["origin"]
        self.h, self.w = self.raw.shape
        self.res = float(self.meta["resolution"])

        # 5. Execute processing pipeline
        self._calculate_wall_bounds()
        self._extract_wall_segments()
        self._compute_centerline()
        self._build_lut()

        # Mark as initialized to enable instant cache-returns on future calls
        self._initialized = True

    def _calculate_wall_bounds(self):
        """Calculates physical dimensions of the track and mathematical centers the world origin."""
        track_pts = np.argwhere(self.free)
        
        # Fallback to full image bounds if no free space is found (prevent crashes)
        if len(track_pts) == 0:
            min_r, min_c, max_r, max_c = 0, 0, self.h - 1, self.w - 1
        else:
            min_r, min_c = track_pts.min(axis=0)
            max_r, max_c = track_pts.max(axis=0)

        # Physical track bounds in meters
        self.wall_width = float((max_c - min_c) * self.res)
        self.wall_length = float((max_r - min_r) * self.res)

        # Map pixel centers to world-space coordinates
        center_c = (min_c + max_c) / 2.0
        center_r = (min_r + max_r) / 2.0
        orig_center_x = self.ox + center_c * self.res
        orig_center_y = self.oy + (self.h - 1 - center_r) * self.res

        # Shift the raw origin so the track's mathematical center perfectly hits (0, 0)
        self.ox -= orig_center_x
        self.oy -= orig_center_y

        self.center_x = 0.0
        self.center_y = 0.0
        self.max_extent = float(max(self.wall_width, self.wall_length)) + 2.0

    def _extract_wall_segments(self):
        """Extracts drivable boundaries as 2D vector line segments."""
        # 1. Convert boolean free space to uint8 for OpenCV compatibility
        free_img = self.free.astype(np.uint8) * 255

        # 2. Find contours (the boundaries between free space and walls)
        # RETR_LIST gets outer walls and inner obstacles.
        contours, _ = cv2.findContours(free_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        segments = []
        
        for contour in contours:
            # 3. Simplify the jagged pixel edges into smooth vector lines
            # epsilon is the maximum distance from contour to approximated contour.
            # 1.0 pixel is usually a perfect balance of accuracy vs segment reduction.
            approx = cv2.approxPolyDP(contour, epsilon=1.0, closed=True)
            
            # Reshape OpenCV's output to a standard (N, 2) array of [col, row]
            pts = approx.reshape(-1, 2) 
            
            if len(pts) < 3:
                continue # Ignore microscopic noise artifacts
                
            # 4. Convert pixel coordinates (col, row) to world coordinates (X, Y)
            world_x = self.ox + pts[:, 0] * self.res
            world_y = self.oy + (self.h - 1 - pts[:, 1]) * self.res
            
            world_pts = np.column_stack([world_x, world_y])
            
            # 5. Create point-to-point line segments closing the loop
            for i in range(len(world_pts)):
                p1 = world_pts[i]
                p2 = world_pts[(i + 1) % len(world_pts)]
                
                # Append as [x1, y1, x2, y2]
                segments.append([p1[0], p1[1], p2[0], p2[1]])
                
        # Store as a flat, highly-optimized float32 array for the GPU
        self.wall_segments = np.array(segments, dtype=np.float32)
        print(f" -> Extracted {len(self.wall_segments)} wall segments for {self.path_name}")

    def _compute_centerline(self):
        """Extracts the optimal racing circuit centerline using skeletonization and Dijkstra's algorithm."""
        
        # 1. Collapse drivable space into a 1-pixel wide skeleton
        skel = skeletonize(self.free)
        pts = np.argwhere(skel)
        
        if len(pts) == 0:
            raise RuntimeError(f"[Map Error] Skeleton is empty on {self.img_path.name}.")

        # Fast lookup tables for pixel coordinates to node IDs and vice-versa
        node_to_px = {i: tuple(pt) for i, pt in enumerate(pts)}
        px_to_node = {tuple(pt): i for i, pt in enumerate(pts)}

        # 2. Build graph adjacency list checking 8-way connectivity
        adj = {i: set() for i in node_to_px}
        for i, (r, c) in node_to_px.items():
            for dr, dc in ADJ:
                nr, nc = r + dr, c + dc
                if (nr, nc) in px_to_node:
                    adj[i].add(px_to_node[(nr, nc)])

        # 3. Heal disjointed skeleton segments (jumps < 12 pixels are reconnected)
        kdtree = KDTree(pts)
        max_gap_pixels = 12.0  
        endpoints = [i for i, neighbors in adj.items() if len(neighbors) <= 1]
        
        for u in endpoints:
            u_px = node_to_px[u]
            indices = kdtree.query_ball_point(u_px, r=max_gap_pixels)
            for v in indices:
                if v == u or v in adj[u]: 
                    continue
                v_px = node_to_px[v]
                if np.hypot(u_px[0] - v_px[0], u_px[1] - v_px[1]) > 1.5:
                    adj[u].add(v)
                    adj[v].add(u)

        # 4. Iteratively prune useless dead-end branches
        degrees = {i: len(neighbors) for i, neighbors in adj.items()}
        q = deque([i for i, d in degrees.items() if d == 1])

        while q:
            u = q.popleft()
            for v in adj[u]:
                if u in adj[v]:
                    adj[v].remove(u)
                    degrees[v] -= 1
                    if degrees[v] == 1: q.append(v)
            adj[u].clear()
            degrees[u] = 0

        valid_nodes = set(i for i, d in degrees.items() if d >= 2)
        if not valid_nodes:
            raise RuntimeError(f"[Map Error] No closed loops found on {self.img_path.name}.")

        # 5. Extract the largest connected component (main track)
        visited = set()
        components = []
        for node in valid_nodes:
            if node not in visited:
                comp = []
                cq = deque([node])
                visited.add(node)
                while cq:
                    curr = cq.popleft()
                    comp.append(curr)
                    for nbr in adj[curr]:
                        if nbr in valid_nodes and nbr not in visited:
                            visited.add(nbr)
                            cq.append(nbr)
                components.append(comp)

        main_component = set(max(components, key=len))

        # Start search at the widest point on the track
        start_node = max(main_component, key=lambda n: self.dt[node_to_px[n]])
        min_clearance_px = max(2.0, 0.15 / self.res)

        def dijkstra_soft(src, target=None, penalty_nodes=None):
            """Internal pathfinder prioritizing wider spaces over strictly shorter distances."""
            # FIX: Mutable default argument eliminated to prevent memory leaking between calls
            if penalty_nodes is None:
                penalty_nodes = set()
                
            pq = [(0.0, src)]
            costs = {src: 0.0}
            parent = {src: None}
            
            while pq:
                curr_cost, u = heapq.heappop(pq)
                if target is not None and u == target: break
                
                # Prune outdated higher-cost paths
                if curr_cost > costs.get(u, float('inf')): 
                    continue
                
                for v in adj[u]:
                    if v not in main_component: 
                        continue
                    
                    u_px, v_px = node_to_px[u], node_to_px[v]
                    clearance = self.dt[v_px]
                    
                    if clearance < min_clearance_px: 
                        continue
                    
                    edge_dist = np.hypot(u_px[0] - v_px[0], u_px[1] - v_px[1])
                    # Weight function aggressively penalizes getting close to walls
                    weight = edge_dist + (8.0 / (clearance + 1e-3))
                    
                    if v in penalty_nodes: 
                        weight += 5000.0
                    
                    new_cost = curr_cost + weight
                    if new_cost < costs.get(v, float('inf')):
                        costs[v] = new_cost
                        parent[v] = u
                        heapq.heappush(pq, (new_cost, v))
            return costs, parent

        # Route to furthest point on map
        forward_costs, _ = dijkstra_soft(start_node)
        far_nodes = sorted([n for n in forward_costs if n in main_component], 
                           key=lambda n: forward_costs[n], reverse=True)
        
        if not far_nodes: 
            raise RuntimeError("[Map Error] Routing disconnected.")
        target_node = far_nodes[0]

        # Extract Outbound Path
        _, p1_tree = dijkstra_soft(start_node, target_node)
        path1, curr = [], target_node
        while curr is not None:
            path1.append(curr)
            curr = p1_tree[curr]
        path1.reverse()

        # Extract Inbound Path (Penalize outbound nodes so it takes the other side of the track)
        internal_nodes = set(path1[1:-1])
        _, p2_tree = dijkstra_soft(target_node, start_node, penalty_nodes=internal_nodes)
        
        path2, curr = [], start_node
        while curr is not None:
            path2.append(curr)
            curr = p2_tree[curr]
        path2.reverse()
        
        best_circuit = path1 + path2[1:-1]

        # ---------------------------------------------------------
        # OPTIMIZED PRUNER: Removes dead-end excursions in O(N) time
        # ---------------------------------------------------------
        simplified_circuit = []
        node_indices = {}  # Tracks the list index of nodes to avoid O(N) lookups
        max_detour_len = len(best_circuit) * 0.25 

        for node in best_circuit:
            if node in node_indices:
                idx = node_indices[node]
                # If the loop/detour is short, slice it out
                if len(simplified_circuit) - idx < max_detour_len:
                    # Clean up the dictionary for nodes we are about to delete
                    for popped_node in simplified_circuit[idx + 1:]:
                        del node_indices[popped_node]
                    simplified_circuit = simplified_circuit[:idx + 1]
                else:
                    # Massive loop (main circuit overlap). Keep it.
                    simplified_circuit.append(node)
                    node_indices[node] = len(simplified_circuit) - 1
            else:
                simplified_circuit.append(node)
                node_indices[node] = len(simplified_circuit) - 1

        best_circuit = simplified_circuit
        # ---------------------------------------------------------

        # Convert back to physical world coordinates and orient sequence
        best_path_px = np.array([node_to_px[n] for n in best_circuit])

        origin_px = np.array([self.h - 1 + self.oy / self.res, -self.ox / self.res])
        start_idx = np.argmin(((best_path_px - origin_px) ** 2).sum(axis=1))
        best_path_rolled = np.roll(best_path_px, -start_idx, axis=0)

        world = np.column_stack(
            [
                self.ox + best_path_rolled[:, 1] * self.res,
                self.oy + (self.h - 1 - best_path_rolled[:, 0]) * self.res,
            ]
        )

        # Smooth raw geometry
        self.centerline = savgol_filter(world, SMOOTH_WINDOW, 3, axis=0, mode="wrap")
        
        # Precompute yaw angles
        diffs = np.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        
        # Calculate dynamic index lookahead based on average pixel spacing
        avg_sp = float(np.linalg.norm(diffs, axis=1).mean())
        self.look_step = max(1, int(round(1.0 / avg_sp)))

    def _build_lut(self):
        """Generates a spatial lookup table to query nearest centerline index instantly."""
        # Map physical centerline coordinates back to image pixel space
        cl_px = np.column_stack(
            [
                self.h - 1 - (self.centerline[:, 1] - self.oy) / self.res,
                (self.centerline[:, 0] - self.ox) / self.res,
            ]
        )
        
        # Use KDTree to generate an image-sized grid mapping every map pixel to a track index
        tree = KDTree(cl_px)
        rows, cols = np.mgrid[: self.h, : self.w]
        
        self.lut = tree.query(
            np.column_stack([rows.ravel(), cols.ravel()]), workers=-1
        )[1].reshape(rows.shape)