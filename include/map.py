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
        
        # Phase 1: Morphological Skeletonization and Adjacency Processing
        pts, num_nodes, clearances, all_u, all_v, all_dists = self._extract_initial_skeleton_graph()
        
        # Phase 2: Heal Disjointed Structural Components via KDTree Vectorization
        all_u, all_v, all_dists = self._heal_skeleton_gaps(pts, num_nodes, all_u, all_v, all_dists)
        
        # Phase 3: Prune Dead-Ends and Segment Core Topological Loops
        start_node, in_main_component = self._prune_and_segment_main_loop(
            num_nodes, clearances, all_u, all_v, all_dists
        )
        
        # Phase 4: Path Routing, Detour Removal, and Coordinate Spline Transformation
        self._route_and_smooth_circuit(
            pts, num_nodes, clearances, start_node, in_main_component, all_u, all_v, all_dists
        )

    def _extract_initial_skeleton_graph(self):
        """Phase 1: Generates skeleton node pixels and extracts direct pixel neighbor relations."""
        skel = skeletonize(self.free, method="zhang")
        pts = np.argwhere(skel)
        num_nodes = len(pts)
        if num_nodes == 0:
            raise RuntimeError(f"Skeleton empty on {self.img_path.name}.")
            
        node_grid = np.full((self.h, self.w), -1, dtype=np.int32)
        node_grid[pts[:, 0], pts[:, 1]] = np.arange(num_nodes)
        clearances = self.dt[pts[:, 0], pts[:, 1]]
        
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
            all_u.append(u_nodes)
            all_v.append(v_nodes)
            all_dists.append(np.full(len(u_nodes), dist, dtype=np.float64))

        if all_u:
            all_u = np.concatenate(all_u)
            all_v = np.concatenate(all_v)
            all_dists = np.concatenate(all_dists)
        else:
            all_u = np.empty(0, dtype=np.int32)
            all_v = np.empty(0, dtype=np.int32)
            all_dists = np.empty(0, dtype=np.float64)
            
        return pts, num_nodes, clearances, all_u, all_v, all_dists

    @profile
    def _heal_skeleton_gaps(self, pts, num_nodes, all_u, all_v, all_dists):
        """Phase 2: Performs vectorized bridge building across micro-gaps for disjointed endpoints."""
        degrees = np.bincount(all_u, minlength=num_nodes)
        endpoints = np.where(degrees <= 1)[0]

        if len(endpoints) > 0:
            kdtree = KDTree(pts)
            max_gap_pixels = 12.0
            existing_edges = set(zip(all_u, all_v))
            
            pairs_u, pairs_v = [], []
            for u in endpoints:
                u_px = pts[u]
                indices = kdtree.query_ball_point(u_px, r=max_gap_pixels)
                for v in indices:
                    if v == u or (u, v) in existing_edges:
                        continue
                    pairs_u.append(u)
                    pairs_v.append(v)
                    
            if pairs_u:
                arr_u = np.array(pairs_u, dtype=np.int32)
                arr_v = np.array(pairs_v, dtype=np.int32)
                
                u_coords = pts[arr_u]
                v_coords = pts[arr_v]
                dists = np.linalg.norm(u_coords - v_coords, axis=1)
                
                valid_mask = dists > 1.5
                filtered_u = arr_u[valid_mask]
                filtered_v = arr_v[valid_mask]
                filtered_dists = dists[valid_mask]
                
                if len(filtered_dists) > 0:
                    heal_u = np.empty(2 * len(filtered_u), dtype=np.int32)
                    heal_u[0::2] = filtered_u
                    heal_u[1::2] = filtered_v
                    
                    heal_v = np.empty(2 * len(filtered_v), dtype=np.int32)
                    heal_v[0::2] = filtered_v
                    heal_v[1::2] = filtered_u
                    
                    heal_dists = np.empty(2 * len(filtered_dists), dtype=np.float64)
                    heal_dists[0::2] = filtered_dists
                    heal_dists[1::2] = filtered_dists
                    
                    all_u = np.concatenate([all_u, heal_u])
                    all_v = np.concatenate([all_v, heal_v])
                    all_dists = np.concatenate([all_dists, heal_dists])
                    
        return all_u, all_v, all_dists

    @profile
    def _prune_and_segment_main_loop(self, num_nodes, clearances, all_u, all_v, all_dists):
        """Phase 3: Strips spurious dead-ends vectorially and isolates the dominant loop component."""
        active_edges_mask = np.ones(len(all_u), dtype=bool)

        while True:
            degrees = np.bincount(all_u[active_edges_mask], minlength=num_nodes)
            leaves = np.where(degrees == 1)[0]
            if len(leaves) == 0:
                break
            invalid_mask = np.isin(all_u, leaves) | np.isin(all_v, leaves)
            active_edges_mask &= ~invalid_mask

        all_u_p = all_u[active_edges_mask]
        all_v_p = all_v[active_edges_mask]

        valid_nodes_mask = degrees >= 2
        valid_indices = np.where(valid_nodes_mask)[0]
        if len(valid_indices) == 0:
            raise RuntimeError(f"No closed loops found on {self.img_path.name}.")

        valid_edges_mask = valid_nodes_mask[all_u_p] & valid_nodes_mask[all_v_p]
        p_row = all_u_p[valid_edges_mask]
        p_col = all_v_p[valid_edges_mask]

        pruned_graph = csr_matrix((np.ones(len(p_row)), (p_row, p_col)), shape=(num_nodes, num_nodes))
        _, labels = connected_components(csgraph=pruned_graph, directed=False, return_labels=True)
        
        valid_labels = labels[valid_indices]
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        largest_comp_label = unique_labels[np.argmax(counts)]
        
        in_main_component = (labels == largest_comp_label) & valid_nodes_mask
        main_comp_indices = np.where(in_main_component)[0]
        start_node = main_comp_indices[np.argmax(clearances[main_comp_indices])]
        
        return start_node, in_main_component

    @profile
    def _route_and_smooth_circuit(self, pts, num_nodes, clearances, start_node, in_main_component, all_u, all_v, all_dists):
        """Phase 4: Computes bidirectional penalization routing, clears loops, and processes smooth coordinates."""
        physical_clearance_limit = 0.15
        min_clearance_px = max(2.0, physical_clearance_limit / self.res)
        penalty_scale = 8.0

        route_mask = in_main_component[all_u] & in_main_component[all_v] & (clearances[all_v] >= min_clearance_px)
        d_row = all_u[route_mask]
        d_col = all_v[route_mask]
        d_weight = all_dists[route_mask] + (penalty_scale / (clearances[d_col] + 1e-3))

        if len(d_weight) == 0:
            raise RuntimeError("Routing disconnected: No valid edges after clearance filtering.")

        dijkstra_graph = csr_matrix((d_weight, (d_row, d_col)), shape=(num_nodes, num_nodes))
        dist_matrix, predecessors = dijkstra(csgraph=dijkstra_graph, directed=True, indices=start_node, return_predecessors=True)
        
        main_comp_indices = np.where(in_main_component)[0]
        valid_dists = dist_matrix[main_comp_indices]
        valid_dists[np.isinf(valid_dists)] = -1
        target_node = main_comp_indices[np.argmax(valid_dists)]
        
        if target_node == start_node or dist_matrix[target_node] <= 0:
            raise RuntimeError("Routing disconnected during exploration.")

        path1 = self._reconstruct_path(start_node, target_node, predecessors)

        # Inbound Path Trace Routing with Collision Avoidance
        internal_nodes = np.array(path1[1:-1], dtype=np.int32)
        collision_penalty = 5000.0
        
        d_weight_penalized = np.array(d_weight)
        penalized_edges = np.isin(d_col, internal_nodes)
        d_weight_penalized[penalized_edges] += collision_penalty
        
        dijkstra_graph_penalized = csr_matrix((d_weight_penalized, (d_row, d_col)), shape=(num_nodes, num_nodes))
        _, predecessors_2 = dijkstra(csgraph=dijkstra_graph_penalized, directed=True, indices=target_node, return_predecessors=True)
        
        path2 = self._reconstruct_path(target_node, start_node, predecessors_2)
        best_circuit = path1 + path2[1:-1]

        # --- HYPER-OPTIMIZED TRACKING & DETOUR REMOVAL ---
        # Replace dictionary lookups with a fast, pre-allocated flat tracking array
        node_positions = np.full(num_nodes, -1, dtype=np.int32)
        max_detour_len = len(best_circuit) * 0.25
        
        simplified_circuit = []
        append_node = simplified_circuit.append
        
        for node in best_circuit:
            idx = node_positions[node]
            if idx != -1:
                if len(simplified_circuit) - idx < max_detour_len:
                    # Wipe values from tracking array for the dropped slice in one go
                    dropped_nodes = simplified_circuit[idx + 1:]
                    node_positions[dropped_nodes] = -1
                    del simplified_circuit[idx + 1:]
                    continue
            
            append_node(node)
            node_positions[node] = len(simplified_circuit) - 1
                
        best_circuit = simplified_circuit

        # Spatial Coordinate Mapping & Final Filtering
        best_path_px = pts[best_circuit]
        origin_px = np.array([self.h - 1 + self.oy / self.res, -self.ox / self.res])
        start_idx = np.argmin(((best_path_px - origin_px) ** 2).sum(axis=1))
        best_path_rolled = np.roll(best_path_px, -start_idx, axis=0)
        
        world = np.column_stack([
            self.ox + best_path_rolled[:, 1] * self.res,
            self.oy + (self.h - 1 - best_path_rolled[:, 0]) * self.res,
        ])
        
        poly_order = 3
        self.centerline = savgol_filter(world, SMOOTH_WINDOW, poly_order, axis=0, mode="wrap")
        
        diffs = np.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.angles = np.arctan2(diffs[:, 1], diffs[:, 0])

    @staticmethod
    def _reconstruct_path(start: int, target: int, preds: np.ndarray) -> list:
        """Fastest pure-Python unrolling of the predecessor tree."""
        # Convert the numpy array to a native Python list *once* before looping.
        # Indexing a Python list inside a raw loop is dramatically faster than 
        # scalar indexing a NumPy array.
        preds_list = preds.tolist()
        
        path = []
        append = path.append
        curr = target
        
        while curr != -9999 and curr != start:
            append(curr)
            curr = preds_list[curr]
            
        if curr == start:
            append(start)
            
        path.reverse()
        return path

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