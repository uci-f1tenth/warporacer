"""Debug visualization for the centerline used by training.

Mirrors the algorithm in `Map._compute_centerline` (skeletonize → BFS loop →
savgol-smooth) so you can see exactly what the training would consume on a
given map. Useful when the centerline looks broken (BFS fails to close, the
start seed lands on a spur, the smoothing oversteers a sharp corner, etc.).

Usage:
    python warporacer/viz_centerline.py warporacer/maps/my_map.yaml
"""

from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cv2 import IMREAD_GRAYSCALE, imread
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
from typer import run
from yaml import safe_load

OCC_THRESH = 230
SMOOTH_WINDOW = 51
ADJ = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def neighbors(skel, r, c, h, w):
    return [
        (r + dr, c + dc)
        for dr, dc in ADJ
        if 0 <= r + dr < h and 0 <= c + dc < w and skel[r + dr, c + dc]
    ]


def trace_loop(skel, start):
    h, w = skel.shape
    nbrs = neighbors(skel, start[0], start[1], h, w)
    if len(nbrs) < 2:
        return None, None, nbrs
    src, target = nbrs[0], nbrs[1]
    parent = {src: src}
    q = deque([src])
    closed = False
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors(skel, r, c, h, w):
            n = (nr, nc)
            if n in parent or n == start:
                continue
            parent[n] = (r, c)
            if n == target:
                closed = True
                q.clear()
                break
            q.append(n)
    if not closed:
        return None, src, nbrs
    path = [start]
    n = target
    while n != src:
        path.append(n)
        n = parent[n]
    path.append(src)
    path.reverse()
    return np.array(path), src, nbrs


def main(map_yaml: Path, save: Path = None, crop: str = ""):
    meta = safe_load(map_yaml.read_text())
    img_path = map_yaml.parent / meta["image"]
    raw = imread(str(img_path), IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(img_path)

    res = float(meta["resolution"])
    ox, oy, _ = meta["origin"]
    h, w = raw.shape

    free = raw >= OCC_THRESH
    skel = skeletonize(free)
    pts = np.argwhere(skel)

    print(f"map: {img_path.name}  shape={raw.shape}  res={res}  origin=({ox}, {oy})")
    print(f"free pixels: {int(free.sum())} / {free.size}  "
          f"skeleton pixels: {len(pts)}")

    origin_px = np.array([h - 1 + oy / res, -ox / res])
    start = tuple(int(v) for v in pts[np.argmin(((pts - origin_px) ** 2).sum(1))])
    print(f"world origin in pixels: row={origin_px[0]:.1f} col={origin_px[1]:.1f}")
    print(f"start seed (nearest skeleton pixel): row={start[0]} col={start[1]}")
    nb = neighbors(skel, start[0], start[1], h, w)
    print(f"skeleton degree at start: {len(nb)}  ({nb})")

    path_rc, src, _ = trace_loop(skel, start)

    fig, ax = plt.subplots(figsize=(12, 12 * h / max(w, 1)))
    ax.imshow(raw, cmap="gray", origin="upper")
    sk_rows, sk_cols = np.where(skel)
    ax.scatter(sk_cols, sk_rows, s=1, c="#ffcc00", label="skeleton", alpha=0.6)

    ax.scatter([origin_px[1]], [origin_px[0]], s=160, marker="x", c="cyan",
               linewidths=2.5, label="world origin (0, 0)")

    if path_rc is None:
        print("BFS did NOT close the loop — skeleton is broken or has spurs.")
        if src is not None:
            ax.scatter([src[1]], [src[0]], s=80, marker="^", c="orange",
                       label="BFS source (start neighbour)")
    else:
        print(f"BFS closed loop:    raw path length = {len(path_rc)} pts")
        world = np.column_stack(
            [ox + path_rc[:, 1] * res, oy + (h - 1 - path_rc[:, 0]) * res]
        )
        if len(world) >= SMOOTH_WINDOW:
            smoothed = savgol_filter(world, SMOOTH_WINDOW, 3, axis=0, mode="wrap")
        else:
            print(f"WARNING: path shorter than SMOOTH_WINDOW={SMOOTH_WINDOW}, "
                  "skipping savgol")
            smoothed = world

        sm_px_col = (smoothed[:, 0] - ox) / res
        sm_px_row = h - 1 - (smoothed[:, 1] - oy) / res
        ax.plot(path_rc[:, 1], path_rc[:, 0], "-", lw=1.0, c="#3aa3ff",
                label="raw BFS path", alpha=0.6)
        ax.plot(sm_px_col, sm_px_row, "-", lw=2.0, c="#ff3a6b",
                label="smoothed centerline")

        # Crop preview: draw the box and highlight the arc inside it. These
        # are the same image-fraction coords main.py's --crop consumes.
        if crop:
            fx0, fy0, fx1, fy1 = (float(v) for v in crop.split(","))
            fx0, fx1 = sorted((fx0, fx1))
            fy0, fy1 = sorted((fy0, fy1))
            cmin, cmax, rmin, rmax = fx0 * w, fx1 * w, fy0 * h, fy1 * h
            in_box = ((sm_px_col >= cmin) & (sm_px_col <= cmax)
                      & (sm_px_row >= rmin) & (sm_px_row <= rmax))
            ax.add_patch(plt.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin,
                         fill=False, edgecolor="white", lw=2.0, ls="--",
                         label=f"crop {crop}"))
            ax.scatter(sm_px_col[in_box], sm_px_row[in_box], s=10, c="lime",
                       label=f"arc in crop ({int(in_box.sum())} pts)")
            print(f"crop {crop}: {int(in_box.sum())}/{len(in_box)} "
                  f"centerline pts inside ({in_box.mean():.0%} of loop)")

        # Direction arrow near the start, pointing along travel.
        step = max(1, len(smoothed) // 50)
        dx = sm_px_col[step] - sm_px_col[0]
        dy = sm_px_row[step] - sm_px_row[0]
        ax.annotate("", xy=(sm_px_col[0] + dx, sm_px_row[0] + dy),
                    xytext=(sm_px_col[0], sm_px_row[0]),
                    arrowprops=dict(arrowstyle="->", color="lime", lw=2.5))

        diffs = np.diff(smoothed, axis=0, append=smoothed[:1])
        spacings = np.linalg.norm(diffs, axis=1)
        print(f"smoothed centerline: {len(smoothed)} pts, "
              f"avg spacing={spacings.mean():.3f} m, "
              f"min={spacings.min():.3f} m, max={spacings.max():.3f} m")
        seam = np.linalg.norm(smoothed[0] - smoothed[-1])
        print(f"seam gap (last → first): {seam:.3f} m")

    ax.scatter([start[1]], [start[0]], s=140, marker="o",
               edgecolor="red", facecolor="none", linewidths=2.5,
               label="start seed")

    ax.set_title(f"{img_path.name} — centerline debug")
    ax.legend(loc="upper right")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    plt.tight_layout()

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=150)
        print(f"saved figure to {save}")
    else:
        plt.show()


if __name__ == "__main__":
    run(main)
