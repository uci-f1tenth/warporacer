import heapq
from collections import deque
from pathlib import Path

import numpy as np
from cv2 import IMREAD_GRAYSCALE, imread
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from yaml import safe_load

from include.constants import *


class Map:
    # Create a class-level cache dictionary
    _cache = {}

    def __new__(cls, path: Path):
        # 2. Use the absolute path as the unique cache key
        abs_path = path.resolve()
        
        if abs_path not in cls._cache:
            # If not in cache, allocate memory for a new instance
            instance = super().__new__(cls)
            instance._initialized = False
            cls._cache[abs_path] = instance
            
        # Return the cached instance
        return cls._cache[abs_path]

    def __init__(self, path: Path):
        # 3. If this instance was pulled from the cache, skip the heavy math
        if getattr(self, "_initialized", False):
            return

        print(f"Processing and caching new map: {path.name}...")

        self.meta = safe_load(path.read_text())
        self.img_path = path.parent / self.meta["image"]

        self.raw = imread(str(self.img_path), IMREAD_GRAYSCALE)
        if self.raw is None:
            raise FileNotFoundError(f"Could not load image at {self.img_path}")
        
        # Paint a literal black border directly onto the raw image
        self.raw[0, :] = 0    
        self.raw[-1, :] = 0   
        self.raw[:, 0] = 0    
        self.raw[:, -1] = 0   
        
        self.free = self.raw >= OCC_THRESH
        self.dt = distance_transform_edt(self.free)

        self.ox, self.oy, _ = self.meta["origin"]
        self.h, self.w = self.raw.shape
        self.res = float(self.meta["resolution"])

        self._calculate_wall_bounds()
        self._compute_centerline()
        self._build_lut()

        # 4. Mark as initialized so future calls bypass this block
        self._initialized = True

    def _calculate_wall_bounds(self):
        """Measures the drivable free space and zero-centers the map origin."""
        # FIX: Target the free track space, not the occupied void.
        track_pts = np.argwhere(self.free)
        if len(track_pts) == 0:
            min_r, min_c, max_r, max_c = 0, 0, self.h - 1, self.w - 1
        else:
            min_r, min_c = track_pts.min(axis=0)
            max_r, max_c = track_pts.max(axis=0)

        # Calculate the physical dimensions of the track
        self.wall_width = float((max_c - min_c) * self.res)
        self.wall_length = float((max_r - min_r) * self.res)

        # Calculate where the center of the track currently is in world-space
        center_c = (min_c + max_c) / 2.0
        center_r = (min_r + max_r) / 2.0
        orig_center_x = self.ox + center_c * self.res
        orig_center_y = self.oy + (self.h - 1 - center_r) * self.res

        # FIX: Shift the map's underlying origin so the track's center perfectly aligns with (0,0)
        self.ox -= orig_center_x
        self.oy -= orig_center_y

        # The center is now exactly 0,0. Add a 2-meter padding so the floor exceeds the walls slightly.
        self.center_x = 0.0
        self.center_y = 0.0
        self.max_extent = float(max(self.wall_width, self.wall_length)) + 2.0

    def _compute_centerline(self):
        skel = skeletonize(self.free)
        pts = np.argwhere(skel)
        
        if len(pts) == 0:
            raise RuntimeError(f"[Map Error] Skeleton is empty on {self.img_path.name}.")

        node_to_px = {i: tuple(pt) for i, pt in enumerate(pts)}
        px_to_node = {tuple(pt): i for i, pt in enumerate(pts)}

        adj = {i: set() for i in node_to_px}
        for i, (r, c) in node_to_px.items():
            for dr, dc in ADJ:
                nr, nc = r + dr, c + dc
                if (nr, nc) in px_to_node:
                    adj[i].add(px_to_node[(nr, nc)])

        kdtree = KDTree(pts)
        max_gap_pixels = 12.0  
        
        endpoints = [i for i, neighbors in adj.items() if len(neighbors) <= 1]
        for u in endpoints:
            u_px = node_to_px[u]
            indices = kdtree.query_ball_point(u_px, r=max_gap_pixels)
            for idx in indices:
                v = idx
                if v == u or v in adj[u]: continue
                v_px = node_to_px[v]
                if np.hypot(u_px[0] - v_px[0], u_px[1] - v_px[1]) > 1.5:
                    adj[u].add(v)
                    adj[v].add(u)

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

        start_node = max(main_component, key=lambda n: self.dt[node_to_px[n]])
        min_clearance_px = max(2.0, 0.15 / self.res)

        def dijkstra_soft(src, target=None, penalty_nodes=set()):
            pq = [(0.0, src)]
            costs = {src: 0.0}
            parent = {src: None}
            
            while pq:
                curr_cost, u = heapq.heappop(pq)
                if target is not None and u == target: break
                if curr_cost > costs.get(u, float('inf')): continue
                
                for v in adj[u]:
                    if v not in main_component: continue
                    u_px, v_px = node_to_px[u], node_to_px[v]
                    clearance = self.dt[v_px]
                    
                    if clearance < min_clearance_px: continue
                    
                    edge_dist = np.hypot(u_px[0] - v_px[0], u_px[1] - v_px[1])
                    weight = edge_dist + (8.0 / (clearance + 1e-3))
                    if v in penalty_nodes: weight += 5000.0
                    
                    new_cost = curr_cost + weight
                    if new_cost < costs.get(v, float('inf')):
                        costs[v] = new_cost
                        parent[v] = u
                        heapq.heappush(pq, (new_cost, v))
            return costs, parent

        forward_costs, _ = dijkstra_soft(start_node)
        far_nodes = sorted([n for n in forward_costs if n in main_component], 
                           key=lambda n: forward_costs[n], reverse=True)
        
        if not far_nodes: raise RuntimeError("[Map Error] Routing disconnected.")
        target_node = far_nodes[0]

        _, p1_tree = dijkstra_soft(start_node, target_node)
        path1, curr = [], target_node
        while curr is not None:
            path1.append(curr)
            curr = p1_tree[curr]
        path1.reverse()

        internal_nodes = set(path1[1:-1])
        _, p2_tree = dijkstra_soft(target_node, start_node, penalty_nodes=internal_nodes)
        
        path2, curr = [], start_node
        while curr is not None:
            path2.append(curr)
            curr = p2_tree[curr]
        path2.reverse()
        
        best_circuit = path1 + path2[1:-1]

        # ---------------------------------------------------------
        # SAFE PRUNER: Only removes dead-end excursions (small loops)
        # ---------------------------------------------------------
        simplified_circuit = []
        max_detour_len = len(best_circuit) * 0.25  # A detour is <25% of track length

        for node in best_circuit:
            if node in simplified_circuit:
                idx = simplified_circuit.index(node)
                # If the loop we found is small, it's a side-alley. Slice it out.
                if len(simplified_circuit) - idx < max_detour_len:
                    simplified_circuit = simplified_circuit[:idx+1]
                else:
                    # It's a massive loop (the main track crossing or returning). Keep it.
                    simplified_circuit.append(node)
            else:
                simplified_circuit.append(node)

        best_circuit = simplified_circuit

        # ---------------------------------------------------------

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

        self.centerline = savgol_filter(world, SMOOTH_WINDOW, 3, axis=0, mode="wrap")
        
        diffs = np.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        
        avg_sp = float(np.linalg.norm(diffs, axis=1).mean())
        self.look_step = max(1, int(round(1.0 / avg_sp)))

    def _build_lut(self):
        cl_px = np.column_stack(
            [
                self.h - 1 - (self.centerline[:, 1] - self.oy) / self.res,
                (self.centerline[:, 0] - self.ox) / self.res,
            ]
        )
        
        tree = KDTree(cl_px)
        rows, cols = np.mgrid[: self.h, : self.w]
        
        self.lut = tree.query(
            np.column_stack([rows.ravel(), cols.ravel()]), workers=-1
        )[1].reshape(rows.shape)