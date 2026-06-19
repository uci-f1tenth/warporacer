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
    """
    Parses an occupancy grid map from a YAML/Image pair, extracts the drivable centerline, 
    and builds a spatial lookup table (LUT) mapping pixels to the nearest centerline coordinate.
    """
    def __init__(self, path: Path):
        self.meta = safe_load(path.read_text())
        self.img_path = path.parent / self.meta["image"]

        self.raw = imread(str(self.img_path), IMREAD_GRAYSCALE)
        if self.raw is None:
            raise FileNotFoundError(f"Could not load image at {self.img_path}")
        
        # Binarize raw map data directly to preserve thin walls completely
        self.free = self.raw >= OCC_THRESH
        self.dt = distance_transform_edt(self.free)

        self.ox, self.oy, _ = self.meta["origin"]
        self.h, self.w = self.raw.shape
        self.res = float(self.meta["resolution"])

        self._compute_centerline()
        self._build_lut()

    def _compute_centerline(self):
        """
        Extracts the skeleton, heals local structural gaps using a spatial KDTree, 
        prunes dead ends, and computes a safe cost-weighted loop circuit using soft-penalties.
        """
        skel = skeletonize(self.free)
        pts = np.argwhere(skel)
        
        if len(pts) == 0:
            raise RuntimeError(f"[Map Error] Skeleton is empty on {self.img_path.name}.")

        # ---------------------------------------------------------
        # PHASE 1: Graph Assembly & KDTree Proximity Gap Healing
        # ---------------------------------------------------------
        node_to_px = {i: tuple(pt) for i, pt in enumerate(pts)}
        px_to_node = {tuple(pt): i for i, pt in enumerate(pts)}

        # Build initial 8-connected grid adjacency
        adj = {i: set() for i in node_to_px}
        for i, (r, c) in node_to_px.items():
            for dr, dc in ADJ:
                nr, nc = r + dr, c + dc
                if (nr, nc) in px_to_node:
                    adj[i].add(px_to_node[(nr, nc)])

        # Heal track gaps: Find dead ends and link them to nearby skeleton segments
        kdtree = KDTree(pts)
        max_gap_pixels = 12.0  # Bridges gaps up to ~12 pixels wide caused by raw LiDAR/noise
        
        endpoints = [i for i, neighbors in adj.items() if len(neighbors) <= 1]
        for u in endpoints:
            u_px = node_to_px[u]
            indices = kdtree.query_ball_point(u_px, r=max_gap_pixels)
            for idx in indices:
                v = idx
                if v == u or v in adj[u]:
                    continue
                v_px = node_to_px[v]
                dist = np.hypot(u_px[0] - v_px[0], u_px[1] - v_px[1])
                if dist > 1.5:  # Avoid immediate grid neighbors
                    adj[u].add(v)
                    adj[v].add(u)

        # ---------------------------------------------------------
        # PHASE 2: Fast Topological Pruning & Component Isolation
        # ---------------------------------------------------------
        degrees = {i: len(neighbors) for i, neighbors in adj.items()}
        q = deque([i for i, d in degrees.items() if d == 1])

        # Iteratively melt away true dead-end branches (Levine's hallways)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if u in adj[v]:
                    adj[v].remove(u)
                    degrees[v] -= 1
                    if degrees[v] == 1:
                        q.append(v)
            adj[u].clear()
            degrees[u] = 0

        valid_nodes = set(i for i, d in degrees.items() if d >= 2)
        if not valid_nodes:
            raise RuntimeError(f"[Map Error] No closed loop tracks found on {self.img_path.name}.")

        # Group remaining loops and keep only the largest connected track component
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

        # ---------------------------------------------------------
        # PHASE 3: Soft-Penalty Circuit Reconstructor
        # ---------------------------------------------------------
        # Pick a safe source node located in the widest open track space
        start_node = max(main_component, key=lambda n: self.dt[node_to_px[n]])
        
        # Enforce car width: half-width of F1TENTH car is ~0.15m. 
        # Node must have at least this clearance to be traversable.
        min_clearance_px = max(2.0, 0.15 / self.res)

        def dijkstra_soft(src, target=None, penalty_nodes=set()):
            pq = [(0.0, src)]
            costs = {src: 0.0}
            parent = {src: None}
            
            while pq:
                curr_cost, u = heapq.heappop(pq)
                if target is not None and u == target:
                    break
                if curr_cost > costs.get(u, float('inf')):
                    continue
                for v in adj[u]:
                    if v not in main_component:
                        continue
                    
                    u_px, v_px = node_to_px[u], node_to_px[v]
                    clearance = self.dt[v_px]
                    
                    # Guardrail: Hard block if the physical space cannot fit the car width
                    if clearance < min_clearance_px:
                        continue
                    
                    edge_dist = np.hypot(u_px[0] - v_px[0], u_px[1] - v_px[1])
                    
                    # Base weight favors maximum wall distance (Stata's Pillars)
                    weight = edge_dist + (8.0 / (clearance + 1e-3))
                    
                    # Soft Penalty: heavily penalize reusing forward path nodes unless absolutely forced
                    if v in penalty_nodes:
                        weight += 5000.0
                    
                    new_cost = curr_cost + weight
                    if new_cost < costs.get(v, float('inf')):
                        costs[v] = new_cost
                        parent[v] = u
                        heapq.heappush(pq, (new_cost, v))
            return costs, parent

        # Find the node topologically furthest away from the start
        forward_costs, _ = dijkstra_soft(start_node)
        far_nodes = sorted([n for n in forward_costs if n in main_component], 
                           key=lambda n: forward_costs[n], reverse=True)
        
        if not far_nodes:
            raise RuntimeError(f"[Map Error] Map routing disconnected due to car width safety constraints.")
            
        target_node = far_nodes[0]

        # Path 1: Forward from Start to Target
        _, p1_tree = dijkstra_soft(start_node, target_node)
        path1 = []
        curr = target_node
        while curr is not None:
            path1.append(curr)
            curr = p1_tree[curr]
        path1.reverse()

        # Path 2: Return from Target to Start with Soft Penalties applied to Path 1
        internal_nodes = set(path1[1:-1])
        _, p2_tree = dijkstra_soft(target_node, start_node, penalty_nodes=internal_nodes)
        
        if start_node not in p2_tree:
            raise RuntimeError(f"[Map Error] Failed to stitch loop circuit on {self.img_path.name}.")
            
        path2 = []
        curr = start_node
        while curr is not None:
            path2.append(curr)
            curr = p2_tree[curr]
        path2.reverse()
        
        best_circuit = path1 + path2[1:-1]

        # ---------------------------------------------------------
        # PHASE 4: Transform to Real-World Coordinates & Smooth
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