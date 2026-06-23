"""Track map library class managing image parsing, skeletons, and coordinate lookups."""

from collections import deque
import heapq
from pathlib import Path
from typing import Any, ClassVar, Dict, Tuple

import cv2
from cv2 import IMREAD_GRAYSCALE, imread
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from yaml import safe_load

from include.constants import ADJ, OCC_THRESH, SMOOTH_WINDOW


class Map:
    """Multiton Map class handling map loading, binarization, and skeleton geometry."""

    _cache: ClassVar[Dict[Path, "Map"]] = {}
    _initialized: bool

    path_name: str
    meta: Dict[str, Any]
    img_path: Path
    raw: np.ndarray
    free: np.ndarray
    dt: np.ndarray
    lut: np.ndarray

    ox: float
    oy: float
    h: int
    w: int
    res: float
    shape: Tuple[int, int]

    wall_width: float
    wall_length: float
    center_x: float
    center_y: float
    max_extent: float

    centerline: np.ndarray
    angles: np.ndarray

    def __new__(cls, path: Path) -> "Map":
        """Intercepts instance creation to implement Flyweight/Multiton mapping cache."""
        abs_path = path.resolve()

        if abs_path not in cls._cache:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._cache[abs_path] = instance

        return cls._cache[abs_path]

    @profile
    def __init__(self, path: Path) -> None:
        """Initializes structural map matrices from cached configuration lines."""
        if getattr(self, "_initialized", False):
            return

        print(f"Processing and caching new map: {path.name}...")

        # 1. Load map metadata and raw image array
        self.path_name = path.name
        self.meta = safe_load(path.read_text())
        self.img_path = path.parent / self.meta["image"]

        raw_img = imread(str(self.img_path), IMREAD_GRAYSCALE)
        if raw_img is None:
            raise FileNotFoundError(f"Could not load image at {self.img_path}")
        self.raw = raw_img

        # 2. Programmatically seal image boundaries to enforce safety borders
        self.raw[0, :] = 0
        self.raw[-1, :] = 0
        self.raw[:, 0] = 0
        self.raw[:, -1] = 0

        # 3. Create boolean map of drivable space using metadata thresholds
        self.free = self.raw >= OCC_THRESH
        self.dt = cv2.distanceTransform(
            self.free.astype(np.uint8), 
            cv2.DIST_L2, 
            cv2.DIST_MASK_PRECISE
        )

        # 4. Extract dimensional data and coordinate origin from metadata
        self.ox, self.oy, _ = self.meta["origin"]
        self.h, self.w = self.raw.shape
        self.res = float(self.meta["resolution"])
        self.shape = (self.h, self.w)

        # 5. Execute processing pipeline
        self._calculate_wall_bounds()
        self._compute_centerline()
        self._build_lut()

        self._initialized = True

    @profile
    def _calculate_wall_bounds(self) -> None:
        """Calculates physical dimensions of the track and balances world spaces."""
        track_pts = np.argwhere(self.free)

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

        # Shift raw origins so that track center lands perfectly on (0, 0)
        self.ox -= orig_center_x
        self.oy -= orig_center_y

        self.center_x = 0.0
        self.center_y = 0.0

        extent_padding = 2.0
        self.max_extent = (
            float(max(self.wall_width, self.wall_length)) + extent_padding
        )

    @profile
    def _compute_centerline(self) -> None:
        """Extracts continuous optimal circuit centerlines using high-performance graph heuristics."""
        
        # 1. High-Performance Morphological Skeletonization
        skel = skeletonize(self.free)
        
        pts = np.argwhere(skel)
        num_nodes = len(pts)
        if num_nodes == 0:
            raise RuntimeError(f"Skeleton empty on {self.img_path.name}.")
            
        # 2. O(1) Grid Lookup Mapping
        node_grid = np.full((self.h, self.w), -1, dtype=np.int32)
        node_grid[pts[:, 0], pts[:, 1]] = np.arange(num_nodes)
        clearances = self.dt[pts[:, 0], pts[:, 1]]
        
        # 3. Vectorized Coordinate Array Adjacency Gathering
        # Pre-allocate tracking collections to gather flat matrix coordinates
        all_u, all_v, all_dists = [], [], []
        
        for dr, dc in ADJ:
            sr = pts[:, 0] + dr
            sc = pts[:, 1] + dc
            
            valid = (sr >= 0) & (sr < self.h) & (sc >= 0) & (sc < self.w)
            valid_idx = np.where(valid)[0]
            
            v_nodes = node_grid[sr[valid_idx], sc[valid_idx]]
            hit_mask = v_nodes != -1
            
            u_nodes = valid_idx[hit_mask]
            v_nodes = v_nodes[hit_mask]
            
            if len(u_nodes) == 0:
                continue
                
            dist = 1.41421356 if (dr != 0 and dc != 0) else 1.0
            
            # Directly extend flat lists with chunks of arrays—bypassing Python dict loops entirely
            all_u.append(u_nodes)
            all_v.append(v_nodes)
            all_dists.append(np.full(len(u_nodes), dist, dtype=np.float64))

        # Flatten out neighbor-shift edges
        if all_u:
            all_u = np.concatenate(all_u)
            all_v = np.concatenate(all_v)
            all_dists = np.concatenate(all_dists)
        else:
            all_u = np.empty(0, dtype=np.int32)
            all_v = np.empty(0, dtype=np.int32)
            all_dists = np.empty(0, dtype=np.float64)

        # 4. Heal disjointed structural skeleton components via flat metrics
        # Calculate degree of nodes directly from raw directional mappings
        degrees = np.bincount(all_u, minlength=num_nodes)
        endpoints = np.where(degrees <= 1)[0]
        
        heal_u, heal_v, heal_dists = [], [], []
        
        if len(endpoints) > 0:
            kdtree = KDTree(pts)
            max_gap_pixels = 12.0
            
            # Grouped fast-lookup index arrays to quickly prevent duplicates
            existing_edges = set(zip(all_u, all_v))
            
            for u in endpoints:
                u_px = pts[u]
                indices = kdtree.query_ball_point(u_px, r=max_gap_pixels)
                for v in indices:
                    if v == u or (u, v) in existing_edges:
                        continue
                    v_px = pts[v]
                    dist = float(np.hypot(u_px[0] - v_px[0], u_px[1] - v_px[1]))
                    if dist > 1.5:
                        heal_u.extend([u, v])
                        heal_v.extend([v, u])
                        heal_dists.extend([dist, dist])
                        
            if heal_u:
                all_u = np.concatenate([all_u, heal_u])
                all_v = np.concatenate([all_v, heal_v])
                all_dists = np.concatenate([all_dists, heal_dists])
                degrees = np.bincount(all_u, minlength=num_nodes)

        # 5. Prune dead-end leaf branches (Clean topological loops)
        # Rebuild a temporary lightweight dynamic look-up for quick topological graph reductions
        adj_dict = {i: set() for i in range(num_nodes)}
        for u, v in zip(all_u, all_v):
            adj_dict[u].add(v)
            
        q = deque(np.where(degrees == 1)[0])
        while q:
            u = q.popleft()
            for v in list(adj_dict[u]):
                if u in adj_dict[v]:
                    adj_dict[v].remove(u)
                    degrees[v] -= 1
                    if degrees[v] == 1:
                        q.append(v)
            adj_dict[u].clear()
            degrees[u] = 0

        valid_nodes_mask = degrees >= 2
        valid_indices = np.where(valid_nodes_mask)[0]
        if len(valid_indices) == 0:
            raise RuntimeError(f"No closed loops found on {self.img_path.name}.")

        # 6. Pure Vectorized Connected Component Extraction
        # Filter the flat array elements directly with NumPy logical index operations
        valid_edges_mask = valid_nodes_mask[all_u] & valid_nodes_mask[all_v]
        p_row = all_u[valid_edges_mask]
        p_col = all_v[valid_edges_mask]
        p_data = all_dists[valid_edges_mask]

        pruned_graph = csr_matrix((np.ones(len(p_row)), (p_row, p_col)), shape=(num_nodes, num_nodes))
        _, labels = connected_components(csgraph=pruned_graph, directed=False, return_labels=True)
        
        valid_labels = labels[valid_indices]
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        largest_comp_label = unique_labels[np.argmax(counts)]
        
        in_main_component = (labels == largest_comp_label) & valid_nodes_mask
        main_comp_indices = np.where(in_main_component)[0]
        start_node = main_comp_indices[np.argmax(clearances[main_comp_indices])]

        # 7. Fast Vectorized Sparse Matrix Generation for Dijkstra
        physical_clearance_limit = 0.15
        min_clearance_px = max(2.0, physical_clearance_limit / self.res)
        penalty_scale = 8.0

        # Apply vectorized constraints over raw global edge listings in one operation
        route_mask = in_main_component[all_u] & in_main_component[all_v] & (clearances[all_v] >= min_clearance_px)
        d_row = all_u[route_mask]
        d_col = all_v[route_mask]
        
        # Vectorized computation of custom distance clearances weights
        d_weight = all_dists[route_mask] + (penalty_scale / (clearances[d_col] + 1e-3))

        if len(d_weight) == 0:
            raise RuntimeError("Routing disconnected: No valid edges after clearance filtering.")

        # 8. Route forward to furthest point
        dijkstra_graph = csr_matrix((d_weight, (d_row, d_col)), shape=(num_nodes, num_nodes))
        dist_matrix, predecessors = dijkstra(
            csgraph=dijkstra_graph, directed=True, indices=start_node, return_predecessors=True
        )
        
        valid_dists = dist_matrix[main_comp_indices]
        valid_dists[np.isinf(valid_dists)] = -1
        target_node = main_comp_indices[np.argmax(valid_dists)]
        
        if target_node == start_node or dist_matrix[target_node] <= 0:
            raise RuntimeError("Routing disconnected during exploration.")

        def reconstruct_path(start, target, preds):
            path, curr = [], target
            while curr != -9999 and curr != start:
                path.append(curr)
                curr = preds[curr]
            if curr == start:
                path.append(start)
            path.reverse()
            return path

        path1 = reconstruct_path(start_node, target_node, predecessors)

        # 9. Inbound Path Trace Routing with Collision Avoidance
        internal_nodes = np.array(path1[1:-1], dtype=np.int32)
        collision_penalty = 5000.0
        
        d_weight_penalized = np.array(d_weight)
        penalized_edges = np.isin(d_col, internal_nodes)
        d_weight_penalized[penalized_edges] += collision_penalty
        
        dijkstra_graph_penalized = csr_matrix((d_weight_penalized, (d_row, d_col)), shape=(num_nodes, num_nodes))
        _, predecessors_2 = dijkstra(
            csgraph=dijkstra_graph_penalized, directed=True, indices=target_node, return_predecessors=True
        )
        
        path2 = reconstruct_path(target_node, start_node, predecessors_2)
        best_circuit = path1 + path2[1:-1]

        # 10. Fast O(N) Detour Removal
        simplified_circuit = []
        node_indices = {}
        max_detour_ratio = 0.25
        max_detour_len = len(best_circuit) * max_detour_ratio
        
        for node in best_circuit:
            if node in node_indices:
                idx = node_indices[node]
                if len(simplified_circuit) - idx < max_detour_len:
                    for popped_node in simplified_circuit[idx + 1:]:
                        del node_indices[popped_node]
                    simplified_circuit = simplified_circuit[: idx + 1]
                else:
                    simplified_circuit.append(node)
                    node_indices[node] = len(simplified_circuit) - 1
            else:
                simplified_circuit.append(node)
                node_indices[node] = len(simplified_circuit) - 1
                
        best_circuit = simplified_circuit

        # 11. Spatial Coordinate Mapping & Final Filtering
        best_path_px = pts[best_circuit]
        origin_px = np.array([self.h - 1 + self.oy / self.res, -self.ox / self.res])
        start_idx = np.argmin(((best_path_px - origin_px) ** 2).sum(axis=1))
        best_path_rolled = np.roll(best_path_px, -start_idx, axis=0)
        
        world = np.column_stack(
            [
                self.ox + best_path_rolled[:, 1] * self.res,
                self.oy + (self.h - 1 - best_path_rolled[:, 0]) * self.res,
            ]
        )
        
        poly_order = 3
        self.centerline = savgol_filter(world, SMOOTH_WINDOW, poly_order, axis=0, mode="wrap")
        
        diffs = np.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.angles = np.arctan2(diffs[:, 1], diffs[:, 0])

    @profile
    def _build_lut(self) -> None:
        """Generates coordinate grids to sample closest index parameters."""
        cl_px = np.column_stack(
            [
                self.h - 1 - (self.centerline[:, 1] - self.oy) / self.res,
                (self.centerline[:, 0] - self.ox) / self.res,
            ]
        )

        tree = KDTree(cl_px)
        
        # 1. Extract only the (y, x) coordinates of drivable pixels
        y_coords, x_coords = np.nonzero(self.free)
        
        # 2. Stack them into an (N, 2) array for the KDTree
        query_points = np.column_stack((y_coords, x_coords))

        # 3. Query the tree ONLY for drivable space (cutting workload by ~85-95%)
        nearest_indices = tree.query(query_points, workers=-1)[1]

        # 4. Initialize the LUT with -1 (meaning "wall/invalid/out-of-bounds")
        self.lut = np.full((self.h, self.w), -1, dtype=np.int32)
        
        # 5. Map the computed indices back to their exact pixel locations on the grid
        self.lut[y_coords, x_coords] = nearest_indices