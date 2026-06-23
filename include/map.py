"""Track map library class managing image parsing, skeletons, and coordinate lookups."""

from collections import deque
import heapq
from pathlib import Path
from typing import Any, ClassVar, Dict, Tuple

from cv2 import IMREAD_GRAYSCALE, imread
import numpy as np
from scipy.ndimage import distance_transform_edt
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
        self.dt = distance_transform_edt(self.free)

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

    def _compute_centerline(self) -> None:
        """Extracts continuous optimal circuit centerlines using graph heuristics."""
        skel = skeletonize(self.free)
        pts = np.argwhere(skel)

        if len(pts) == 0:
            raise RuntimeError(f"Skeleton empty on {self.img_path.name}.")

        node_to_px = {i: tuple(pt) for i, pt in enumerate(pts)}
        px_to_node = {tuple(pt): i for i, pt in enumerate(pts)}
        node_clearance = {i: float(self.dt[node_to_px[i]]) for i in node_to_px}

        # Build graph adjacencies using structural dictionary maps
        adj: Dict[int, Dict[int, float]] = {i: {} for i in node_to_px}
        diagonal_weight = 1.41421356

        for i, (r, c) in node_to_px.items():
            for dr, dc in ADJ:
                nr, nc = r + dr, c + dc
                if (nr, nc) in px_to_node:
                    is_diag = dr != 0 and dc != 0
                    adj[i][px_to_node[(nr, nc)]] = (
                        diagonal_weight if is_diag else 1.0
                    )

        # Heal disjointed structural skeleton components
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
                dist = float(np.hypot(u_px[0] - v_px[0], u_px[1] - v_px[1]))
                if dist > 1.5:
                    adj[u][v] = dist
                    adj[v][u] = dist

        # Prune dead-end leaf branch arrays
        degrees = {i: len(neighbors) for i, neighbors in adj.items()}
        q = deque([i for i, d in degrees.items() if d == 1])

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

        valid_nodes = set(i for i, d in degrees.items() if d >= 2)
        if not valid_nodes:
            raise RuntimeError(f"No closed loops found on {self.img_path.name}.")

        # Extract largest connected topological circuit cluster
        visited = set()
        components = []
        for node in valid_nodes:
            if node not in visited:
                comp = []
                cq: deque[int] = deque([node])
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
        start_node = max(main_component, key=lambda n: node_clearance[n])

        physical_clearance_limit = 0.15
        min_clearance_px = max(2.0, physical_clearance_limit / self.res)

        def dijkstra_soft(
            src: int,
            target: int = None,
            penalty_nodes: set = None,
        ) -> Tuple[Dict[int, float], Dict[int, Any]]:
            """Finds minimal clearance paths using soft wall penalties."""
            if penalty_nodes is None:
                penalty_nodes = set()

            pq = [(0.0, src)]
            costs = {src: 0.0}
            parent = {src: None}

            penalty_scale = 8.0
            collision_penalty = 5000.0

            while pq:
                curr_cost, u = heapq.heappop(pq)
                if target is not None and u == target:
                    break
                if curr_cost > costs.get(u, float("inf")):
                    continue

                for v, edge_dist in adj[u].items():
                    if v not in main_component:
                        continue

                    clearance = node_clearance[v]
                    if clearance < min_clearance_px:
                        continue

                    weight = edge_dist + (penalty_scale / (clearance + 1e-3))
                    if v in penalty_nodes:
                        weight += collision_penalty

                    new_cost = curr_cost + weight
                    if new_cost < costs.get(v, float("inf")):
                        costs[v] = new_cost
                        parent[v] = u
                        heapq.heappush(pq, (new_cost, v))

            return costs, parent

        # Route to the furthest available spatial component point
        forward_costs, _ = dijkstra_soft(start_node)
        far_nodes = sorted(
            [n for n in forward_costs if n in main_component],
            key=lambda n: forward_costs[n],
            reverse=True,
        )

        if not far_nodes:
            raise RuntimeError("Routing disconnected during exploration.")
        target_node = far_nodes[0]

        # Extract forward path trace
        _, p1_tree = dijkstra_soft(start_node, target_node)
        path1, curr = [], target_node
        while curr is not None:
            path1.append(curr)
            curr = p1_tree[curr]
        path1.reverse()

        # Extract inbound path trace around obstacles
        internal_nodes = set(path1[1:-1])
        _, p2_tree = dijkstra_soft(
            target_node, start_node, penalty_nodes=internal_nodes
        )

        path2, curr = [], start_node
        while curr is not None:
            path2.append(curr)
            curr = p2_tree[curr]
        path2.reverse()

        best_circuit = path1 + path2[1:-1]

        # Fast O(N) detour loop removal check
        simplified_circuit = []
        node_indices = {}
        max_detour_ratio = 0.25
        max_detour_len = len(best_circuit) * max_detour_ratio

        for node in best_circuit:
            if node in node_indices:
                idx = node_indices[node]
                if len(simplified_circuit) - idx < max_detour_len:
                    for popped_node in simplified_circuit[idx + 1 :]:
                        del node_indices[popped_node]
                    simplified_circuit = simplified_circuit[: idx + 1]
                else:
                    simplified_circuit.append(node)
                    node_indices[node] = len(simplified_circuit) - 1
            else:
                simplified_circuit.append(node)
                node_indices[node] = len(simplified_circuit) - 1

        best_circuit = simplified_circuit

        # Map pixel coordinates directly back into world coordinate vectors
        best_path_px = np.array([node_to_px[n] for n in best_circuit])
        origin_px = np.array(
            [self.h - 1 + self.oy / self.res, -self.ox / self.res]
        )
        start_idx = np.argmin(((best_path_px - origin_px) ** 2).sum(axis=1))
        best_path_rolled = np.roll(best_path_px, -start_idx, axis=0)

        world = np.column_stack(
            [
                self.ox + best_path_rolled[:, 1] * self.res,
                self.oy + (self.h - 1 - best_path_rolled[:, 0]) * self.res,
            ]
        )

        # Apply digital filter to smooth centerline path geometry variations
        poly_order = 3
        self.centerline = savgol_filter(
            world, SMOOTH_WINDOW, poly_order, axis=0, mode="wrap"
        )

        # Precompute sequential track heading tracking yaw lines
        diffs = np.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.angles = np.arctan2(diffs[:, 1], diffs[:, 0])

    def _build_lut(self) -> None:
        """Generates coordinate grids to sample closest index parameters."""
        cl_px = np.column_stack(
            [
                self.h - 1 - (self.centerline[:, 1] - self.oy) / self.res,
                (self.centerline[:, 0] - self.ox) / self.res,
            ]
        )

        tree = KDTree(cl_px)
        query_points = np.indices((self.h, self.w)).reshape(2, -1).T

        nearest_indices = tree.query(query_points, workers=-1)[1]
        self.lut = nearest_indices.reshape(self.h, self.w)