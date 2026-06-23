"""Track map library class managing image parsing, skeletons, and coordinate lookups."""

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

    def _calculate_wall_bounds(self) -> None:
        """Calculates physical dimensions of the track and balances world spaces."""
        track_pts = np.argwhere(self.free)

        if len(track_pts) == 0:
            min_r, min_c, max_r, max_c = 0, 0, self.h - 1, self.w - 1
        else:
            min_r, min_c = track_pts.min(axis=0)
            max_r, max_c = track_pts.max(axis=0)

        self.wall_width = float((max_c - min_c) * self.res)
        self.wall_length = float((max_r - min_r) * self.res)

        # Shift raw origins so that track center lands perfectly on (0, 0)
        orig_center_x = self.ox + (min_c + max_c) / 2.0 * self.res
        orig_center_y = self.oy + (self.h - 1 - (min_r + max_r) / 2.0) * self.res
        self.ox -= orig_center_x
        self.oy -= orig_center_y

        self.center_x = 0.0
        self.center_y = 0.0
        self.max_extent = float(max(self.wall_width, self.wall_length)) + 2.0

    def _compute_centerline(self) -> None:
        """Extracts continuous optimal circuit centerlines using high-performance graph heuristics."""
        pts, num_nodes, clearances, all_u, all_v, all_dists = self._extract_initial_skeleton_graph()
        all_u, all_v, all_dists = self._heal_skeleton_gaps(pts, num_nodes, all_u, all_v, all_dists)
        start_node, in_main_component = self._prune_and_segment_main_loop(num_nodes, clearances, all_u, all_v, all_dists)
        self._route_and_smooth_circuit(pts, num_nodes, clearances, start_node, in_main_component, all_u, all_v, all_dists)

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
            # Because borders are sealed to 0, skeleton nodes can never be on the image edge.
            # Neighbors are always within array bounds. Outer bounds checking removed.
            v_nodes = node_grid[pts[:, 0] + dr, pts[:, 1] + dc]
            hit_mask = v_nodes != -1
            u_nodes = np.where(hit_mask)[0]
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

    def _heal_skeleton_gaps(self, pts, num_nodes, all_u, all_v, all_dists):
        """Phase 2: Performs vectorized bridge building across micro-gaps for disjointed endpoints."""
        degrees = np.bincount(all_u, minlength=num_nodes)
        endpoints = np.where(degrees <= 1)[0]

        if len(endpoints) > 0:
            kdtree = KDTree(pts)
            existing_edges = set(zip(all_u, all_v))
            
            pairs_u, pairs_v = [], []
            for u in endpoints:
                indices = kdtree.query_ball_point(pts[u], r=12.0)
                for v in indices:
                    if v != u and (u, v) not in existing_edges:
                        pairs_u.append(u)
                        pairs_v.append(v)
                        
            if pairs_u:
                arr_u = np.array(pairs_u, dtype=np.int32)
                arr_v = np.array(pairs_v, dtype=np.int32)
                dists = np.linalg.norm(pts[arr_u] - pts[arr_v], axis=1)
                
                valid_mask = dists > 1.5
                filtered_u = arr_u[valid_mask]
                filtered_v = arr_v[valid_mask]
                filtered_dists = dists[valid_mask]
                
                if len(filtered_dists) > 0:
                    n_heal = len(filtered_u)
                    heal_u = np.empty(2 * n_heal, dtype=np.int32)
                    heal_v = np.empty(2 * n_heal, dtype=np.int32)
                    heal_dists = np.empty(2 * n_heal, dtype=np.float64)

                    heal_u[0::2], heal_u[1::2] = filtered_u, filtered_v
                    heal_v[0::2], heal_v[1::2] = filtered_v, filtered_u
                    heal_dists[0::2], heal_dists[1::2] = filtered_dists, filtered_dists
                    
                    all_u = np.concatenate([all_u, heal_u])
                    all_v = np.concatenate([all_v, heal_v])
                    all_dists = np.concatenate([all_dists, heal_dists])
                    
        return all_u, all_v, all_dists

    def _prune_and_segment_main_loop(self, num_nodes, clearances, all_u, all_v, all_dists):
        """Phase 3: Strips spurious dead-ends vectorially and isolates the dominant loop component."""
        active_edges_mask = np.ones(len(all_u), dtype=bool)

        while True:
            degrees = np.bincount(all_u[active_edges_mask], minlength=num_nodes)
            leaves = np.where(degrees == 1)[0]
            if len(leaves) == 0:
                break
            active_edges_mask &= ~(np.isin(all_u, leaves) | np.isin(all_v, leaves))

        all_u_p = all_u[active_edges_mask]
        all_v_p = all_v[active_edges_mask]

        valid_nodes_mask = degrees >= 2
        valid_indices = np.where(valid_nodes_mask)[0]
        if len(valid_indices) == 0:
            raise RuntimeError(f"No closed loops found on {self.img_path.name}.")

        valid_edges_mask = valid_nodes_mask[all_u_p] & valid_nodes_mask[all_v_p]
        pruned_graph = csr_matrix((np.ones(len(valid_edges_mask)), (all_u_p[valid_edges_mask], all_v_p[valid_edges_mask])), shape=(num_nodes, num_nodes))
        _, labels = connected_components(csgraph=pruned_graph, directed=False, return_labels=True)
        
        unique_labels, counts = np.unique(labels[valid_indices], return_counts=True)
        in_main_component = (labels == unique_labels[np.argmax(counts)]) & valid_nodes_mask
        start_node = np.where(in_main_component)[0][np.argmax(clearances[in_main_component])]
        
        return start_node, in_main_component

    def _route_and_smooth_circuit(self, pts, num_nodes, clearances, start_node, in_main_component, all_u, all_v, all_dists):
        """Phase 4: Computes bidirectional penalization routing, clears loops, and processes smooth coordinates."""
        min_clearance_px = max(2.0, 0.15 / self.res)

        route_mask = in_main_component[all_u] & in_main_component[all_v] & (clearances[all_v] >= min_clearance_px)
        d_row = all_u[route_mask]
        d_col = all_v[route_mask]
        d_weight = all_dists[route_mask] + (8.0 / (clearances[d_col] + 1e-3))

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
        d_weight_penalized = np.array(d_weight)
        d_weight_penalized[np.isin(d_col, np.array(path1[1:-1], dtype=np.int32))] += 5000.0
        
        dijkstra_graph_penalized = csr_matrix((d_weight_penalized, (d_row, d_col)), shape=(num_nodes, num_nodes))
        _, predecessors_2 = dijkstra(csgraph=dijkstra_graph_penalized, directed=True, indices=target_node, return_predecessors=True)
        
        best_circuit = path1 + self._reconstruct_path(target_node, start_node, predecessors_2)[1:-1]

        # --- DETOUR REMOVAL ---
        node_positions = np.full(num_nodes, -1, dtype=np.int32)
        max_detour_len = len(best_circuit) * 0.25
        simplified_circuit = []
        append_node = simplified_circuit.append
        
        for node in best_circuit:
            idx = node_positions[node]
            if idx != -1 and (len(simplified_circuit) - idx < max_detour_len):
                node_positions[simplified_circuit[idx + 1:]] = -1
                del simplified_circuit[idx + 1:]
                continue
            append_node(node)
            node_positions[node] = len(simplified_circuit) - 1
                
        # Spatial Coordinate Mapping
        best_path_px = pts[simplified_circuit]
        origin_px = np.array([self.h - 1 + self.oy / self.res, -self.ox / self.res])
        start_idx = np.argmin(((best_path_px - origin_px) ** 2).sum(axis=1))
        best_path_rolled = np.roll(best_path_px, -start_idx, axis=0)
        
        world = np.column_stack([
            self.ox + best_path_rolled[:, 1] * self.res,
            self.oy + (self.h - 1 - best_path_rolled[:, 0]) * self.res,
        ])
        
        poly_order = 3
        n_points = len(world)
        step = max(1, n_points // 1000) 
        
        # Unified downsampled vs raw smoothing pathways into a single execution block
        is_downsampled = step > 1 and (n_points // step) > poly_order
        working_track = world[::step] if is_downsampled else world
        n_working = len(working_track)
        
        safe_window = min(SMOOTH_WINDOW // step if is_downsampled else SMOOTH_WINDOW, n_working - 1) | 1
        if safe_window <= poly_order:
            safe_window = (poly_order + 1) | 1
            
        padded = np.vstack([working_track[-safe_window:], working_track, working_track[:safe_window]])
        smoothed_padded = savgol_filter(padded, safe_window, poly_order, axis=0, mode="nearest")
        smoothed_track = smoothed_padded[safe_window:-safe_window]
        
        if is_downsampled:
            self.centerline = np.empty((n_points, 2), dtype=world.dtype)
            x_orig = np.arange(n_points)
            x_down = np.arange(0, n_points, step)[:n_working]
            self.centerline[:, 0] = np.interp(x_orig, x_down, smoothed_track[:, 0])
            self.centerline[:, 1] = np.interp(x_orig, x_down, smoothed_track[:, 1])
        else:
            self.centerline = smoothed_track
                
        # Final fast vector transformations
        diffs = np.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.angles = np.arctan2(diffs[:, 1], diffs[:, 0])

    @staticmethod
    def _reconstruct_path(start: int, target: int, preds: np.ndarray) -> list:
        """Fastest pure-Python unrolling of the predecessor tree."""
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

    def _build_lut(self) -> None:
        """Generates coordinate grids to sample closest index parameters."""
        cl_px = np.column_stack([
            self.h - 1 - (self.centerline[:, 1] - self.oy) / self.res,
            (self.centerline[:, 0] - self.ox) / self.res,
        ])
        y_coords, x_coords = np.nonzero(self.free)
        nearest_indices = KDTree(cl_px).query(np.column_stack((y_coords, x_coords)), workers=-1)[1]

        self.lut = np.full((self.h, self.w), -1, dtype=np.int32)
        self.lut[y_coords, x_coords] = nearest_indices