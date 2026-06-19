from collections import deque
from pathlib import Path

import numpy as np
from cv2 import IMREAD_GRAYSCALE, imread
from scipy.ndimage import binary_closing, distance_transform_edt, convolve
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
        
        self.free = self.raw >= OCC_THRESH
        
        # Morphological closing heals 1-2 pixel gaps in noisy F1TENTH maps
        self.free = binary_closing(self.free, iterations=2)
        
        self.dt = distance_transform_edt(self.free)

        self.ox, self.oy, _ = self.meta["origin"]
        self.h, self.w = self.raw.shape
        self.res = float(self.meta["resolution"])

        self._compute_centerline()
        self._build_lut()

    @staticmethod
    def _neighbors(skel, r, c, h, w):
        return [
            (r + dr, c + dc)
            for dr, dc in ADJ
            if 0 <= r + dr < h and 0 <= c + dc < w and skel[r + dr, c + dc]
        ]

    def _compute_centerline(self):
        skel = skeletonize(self.free)
        
        # ---------------------------------------------------------------------
        # FIX: Prune dead-end hallways ("spurs") from the skeleton.
        # Maps like Levine have branches that trap the BFS. We iteratively erase 
        # pixels with exactly 1 neighbor until only continuous loops remain.
        # ---------------------------------------------------------------------
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
        while True:
            # Count 8-connected neighbors for every pixel in the skeleton
            neighbor_count = convolve(skel.astype(int), kernel, mode='constant', cval=0)
            
            # Identify pixels that are part of the skeleton but only have 1 connection
            dead_ends = skel & (neighbor_count == 1)
            
            if not dead_ends.any():
                break # All dead ends have successfully retracted to junctions
                
            # Erase the tips of the branches
            skel[dead_ends] = False
        # ---------------------------------------------------------------------

        h, w = skel.shape
        pts = np.argwhere(skel)
        
        if len(pts) == 0:
            raise RuntimeError(f"[Map Error] Skeleton pruning erased map {self.img_path.name}. Is there a closed loop?")
        
        origin_px = np.array([self.h - 1 + self.oy / self.res, -self.ox / self.res])
        start = tuple(int(x) for x in pts[np.argmin(((pts - origin_px) ** 2).sum(1))])
        
        nbrs = self._neighbors(skel, start[0], start[1], h, w)
        if len(nbrs) < 2:
            raise RuntimeError(f"Skeleton seed {start} has {len(nbrs)} neighbours. Is the track closed?")
        
        src, target = nbrs[0], nbrs[1]
        parent = {src: src}
        q = deque([src])
        
        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbors(skel, r, c, h, w):
                n = (nr, nc)
                if n in parent or n == start:
                    continue
                
                parent[n] = (r, c)
                
                if n == target:
                    q.clear()
                    break
                q.append(n)

        if target not in parent:
            raise RuntimeError(
                f"[Map Error] Failed to close the centerline loop on {self.img_path.name}. "
                f"The map may be open-ended, or the drivable area has gaps too large "
                f"for the skeletonizer to cross. Check the map image in an editor."
            )

        path = [start]
        n = target
        while n != src:
            path.append(n)
            n = parent[n]

        path.append(src)
        path.reverse()

        rc = np.array(path)
        world = np.column_stack(
            [
                self.ox + rc[:, 1] * self.res,
                self.oy + (self.h - 1 - rc[:, 0]) * self.res,
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