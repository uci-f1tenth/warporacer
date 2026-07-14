"""Track geometry from a ROS-style map: occupancy image -> wall distances, centerline, waypoint LUT."""

from collections import deque
from pathlib import Path

import numpy as np
from cv2 import IMREAD_GRAYSCALE, imread
from scipy.ndimage import convolve, distance_transform_edt, label
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from yaml import safe_load

OCC_THRESH = 230  # image value at/above which a pixel is drivable
SMOOTH_WINDOW = 51  # savgol window (in waypoints) for centerline smoothing
ADJ = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


class Track:
    """image (h,w uint8), edt (px to nearest wall), centerline (M,2 world m), angles (M,), lut (h,w -> waypoint)."""

    def __init__(self, yaml_path: Path):
        yaml_path = Path(yaml_path)
        meta = safe_load(yaml_path.read_text())
        self.image = imread(str(yaml_path.parent / meta["image"]), IMREAD_GRAYSCALE)
        if self.image is None:
            raise FileNotFoundError(yaml_path.parent / meta["image"])
        self.res = float(meta["resolution"])
        self.ox, self.oy = float(meta["origin"][0]), float(meta["origin"][1])
        self.h, self.w = self.image.shape
        free = self.image >= OCC_THRESH
        self.edt = distance_transform_edt(free)
        self._build_centerline(free)
        self._build_lut()

    def world_to_px(self, x, y):
        return (x - self.ox) / self.res, self.h - 1 - (y - self.oy) / self.res

    def _build_centerline(self, free):
        skel = _largest_component(_prune_spurs(skeletonize(free)))
        col0, row0 = self.world_to_px(0.0, 0.0)
        loop = _longest_loop(skel, (row0, col0))
        xy = np.column_stack(
            [self.ox + loop[:, 1] * self.res, self.oy + (self.h - 1 - loop[:, 0]) * self.res]
        )
        self.centerline = savgol_filter(xy, SMOOTH_WINDOW, 3, axis=0, mode="wrap")
        d = np.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.angles = np.arctan2(d[:, 1], d[:, 0])

    def _build_lut(self):
        cols, rows = self.world_to_px(self.centerline[:, 0], self.centerline[:, 1])
        tree = KDTree(np.column_stack([rows, cols]))
        grid = np.column_stack([g.ravel() for g in np.mgrid[: self.h, : self.w]])
        self.lut = tree.query(grid, workers=-1)[1].reshape(self.h, self.w).astype(np.int32)


def _neighbors(skel, p):
    h, w = skel.shape
    return [
        (p[0] + dr, p[1] + dc)
        for dr, dc in ADJ
        if 0 <= p[0] + dr < h and 0 <= p[1] + dc < w and skel[p[0] + dr, p[1] + dc]
    ]


def _prune_spurs(skel):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    while True:
        tips = skel & (convolve(skel.astype(np.uint8), kernel, mode="constant") <= 1)
        if not tips.any():
            return skel
        skel = skel & ~tips


def _largest_component(skel):
    labels, n = label(skel, structure=np.ones((3, 3), int))
    if n == 0:
        raise RuntimeError("empty skeleton after pruning; is the track a closed loop?")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def _bfs_path(skel, start, src, dst):
    """Shortest 8-connected path src -> dst avoiding start, or None."""
    parent = {src: src}
    queue = deque([src])
    while queue and dst not in parent:
        u = queue.popleft()
        for v in _neighbors(skel, u):
            if v not in parent and v != start:
                parent[v] = u
                queue.append(v)
    if dst not in parent:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(parent[path[-1]])
    return path


def _longest_loop(skel, origin_rowcol):
    """Longest closed pixel loop through skeleton points near the world origin.

    Trying several seeds and every pair of a seed's neighbors keeps this robust
    when a seed lands on a junction or articulation point.
    """
    pts = np.argwhere(skel)
    order = np.argsort(((pts - origin_rowcol) ** 2).sum(1))
    best = None
    for k in order[:32]:
        start = tuple(int(v) for v in pts[k])
        nbrs = _neighbors(skel, start)
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                path = _bfs_path(skel, start, nbrs[i], nbrs[j])
                if path and (best is None or len(path) + 1 > len(best)):
                    best = [start] + path[::-1]
    if best is None:
        raise RuntimeError("could not extract a closed centerline loop")
    return np.array(best)
