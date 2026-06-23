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
        """Extracts continuous optimal circuit centerlines using graph heuristics."""
        
        # 1. Faster Skeletonization (Try OpenCV C++ Thinning, fallback to skimage)
        try:
            free_uint8 = (self.free * 255).astype(np.uint8)
            skel = cv2.ximgproc.thinning(free_uint8, thinningType=cv2.ximgproc.THINNING_GUOHALL) > 0
        except (AttributeError, ImportError):
            from skimage.morphology import skeletonize
            skel = skeletonize(self.free)

        pts = np.argwhere(skel)
        num_nodes = len(pts)
        if num_nodes == 0:
            raise RuntimeError(f"Skeleton empty on {self.img_path.name}.")

        # 2. O(1) Grid Lookup and Vectorized Clearance
        node_grid = np.full((self.h, self.w), -1, dtype=np.int32)
        node_grid[pts[:, 0], pts[:, 1]] = np.arange(num_nodes)
        clearances = self.dt[pts[:, 0], pts[:, 1]]

        # 3. Build Initial Adjacency List (List of Dicts is faster than Dict of Dicts)
        adj = [{} for _ in range(num_nodes)]
        degrees = np.zeros(num_nodes, dtype=np.int32)
        diagonal_weight = 1.41421356

        for i in range(num_nodes):
            r, c = pts[i]
            for dr, dc in ADJ:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.h and 0 <= nc < self.w:
                    nbr = node_grid[nr, nc]
                    if nbr != -1:
                        is_diag = (dr != 0 and dc != 0)
                        dist = diagonal_weight if is_diag else 1.0
                        adj[i][nbr] = dist
                        degrees[i] += 1

        # 4. Heal disjointed structural skeleton components
        endpoints = np.where(degrees <= 1)[0]
        if len(endpoints) > 0:
            kdtree = KDTree(pts)
            max_gap_pixels = 12.0
            for u in endpoints:
                u_px = pts[u]
                indices = kdtree.query_ball_point(u_px, r=max_gap_pixels)
                for v in indices:
                    if v == u or v in adj[u]:
                        continue
                    v_px = pts[v]
                    dist = float(np.hypot(u_px[0] - v_px[0], u_px[1] - v_px[1]))
                    if dist > 1.5:
                        adj[u][v] = dist
                        adj[v][u] = dist
                        degrees[u] += 1
                        degrees[v] += 1

        # 5. Prune dead-end leaf branch arrays
        q = deque(np.where(degrees == 1)[0])
        while q:
            u = q.popleft()
            for v in list(adj[u].keys()):
                if u in adj[v]:
                    del adj[v][u]
                    degrees[v] -= 1
                    if degrees[v] == 1:
                        q.append(v)
            adj[u].clear()
            degrees[u] = 0

        valid_nodes_mask = degrees >= 2
        valid_indices = np.where(valid_nodes_mask)[0]
        if len(valid_indices) == 0:
            raise RuntimeError(f"No closed loops found on {self.img_path.name}.")

        # 6. Extract largest connected component using SciPy CSR Matrix
        p_row, p_col = [], []
        for u in valid_indices:
            for v in adj[u].keys():
                if valid_nodes_mask[v]:
                    p_row.append(u)
                    p_col.append(v)

        pruned_graph = csr_matrix((np.ones(len(p_row)), (p_row, p_col)), shape=(num_nodes, num_nodes))
        _, labels = connected_components(csgraph=pruned_graph, directed=False, return_labels=True)

        valid_labels = labels[valid_indices]
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        largest_comp_label = unique_labels[np.argmax(counts)]

        in_main_component = (labels == largest_comp_label) & valid_nodes_mask
        main_comp_indices = np.where(in_main_component)[0]
        start_node = main_comp_indices[np.argmax(clearances[main_comp_indices])]

        # 7. Build Weighted Sparse Matrix for Dijkstra Fast Routing
        physical_clearance_limit = 0.15
        min_clearance_px = max(2.0, physical_clearance_limit / self.res)
        penalty_scale = 8.0

        d_row, d_col, d_weight = [], [], []
        for u in main_comp_indices:
            for v, dist in adj[u].items():
                if in_main_component[v]:
                    clr = clearances[v]
                    if clr >= min_clearance_px:
                        weight = dist + (penalty_scale / (clr + 1e-3))
                        d_row.append(u)
                        d_col.append(v)
                        d_weight.append(weight)

        # 8. Route forward to furthest point
        dijkstra_graph = csr_matrix((d_weight, (d_row, d_col)), shape=(num_nodes, num_nodes))
        dist_matrix, predecessors = dijkstra(
            csgraph=dijkstra_graph, directed=True, indices=start_node, return_predecessors=True
        )

        valid_dists = dist_matrix[main_comp_indices]
        valid_dists[np.isinf(valid_dists)] = -1  # Ignore unreachable nodes
        target_node = main_comp_indices[np.argmax(valid_dists)]

        if target_node == start_node or dist_matrix[target_node] <= 0:
            raise RuntimeError("Routing disconnected during exploration.")

        def reconstruct_path(start, target, preds):
            path, curr = [], target
            while curr != -9999 and curr != start:  # -9999 is SciPy's null predecessor
                path.append(curr)
                curr = preds[curr]
            if curr == start:
                path.append(start)
            path.reverse()
            return path

        path1 = reconstruct_path(start_node, target_node, predecessors)

        # 9. Route inbound path avoiding the initial trace (applying collision penalties)
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

        # 10. Fast O(N) detour loop removal check
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

        # 11. Coordinate mapping & smoothing
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
        self.centerline = savgol_filter(
            world, SMOOTH_WINDOW, poly_order, axis=0, mode="wrap"
        )

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