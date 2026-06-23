"""Interactive crop picker for `main.py --crop`.

Opens the map image (with the extracted centerline overlaid) and lets you drag
a rectangle. On release it prints a ready-to-paste fraction string and live-
highlights the exact centerline arc that box would select for training — the
same arc logic `Map._compute_spawn_arc` uses, so what you see is what you get.

Usage:
    uv run crop_tool.py maps/new.yaml
    uv run crop_tool.py maps/new.yaml --no-centerline   # image only, faster

Then copy the printed string into training, e.g.:
    uv run main.py maps/new.yaml --init-from logs/agent_final.pt \\
        --crop 0.000,0.000,0.500,0.500 --iterations 150 --lr 1e-4
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cv2 import IMREAD_GRAYSCALE, imread
from matplotlib.widgets import RectangleSelector
from typer import run
from yaml import safe_load


def main(map_yaml: Path, centerline: bool = True):
    meta = safe_load(map_yaml.read_text())
    img_path = map_yaml.parent / meta["image"]
    raw = imread(str(img_path), IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(img_path)
    h, w = raw.shape

    # Build the real centerline so the highlighted arc matches training exactly.
    cl = None  # (cols_px, rows_px, n_cl, longest_circular_run)
    if centerline:
        try:
            from main import Map

            print("building centerline (one-time, may take a few seconds)...")
            m = Map(map_yaml)
            cols = (m.centerline[:, 0] - m.ox) / m.res
            rows = m.h - 1 - (m.centerline[:, 1] - m.oy) / m.res
            cl = (cols, rows, len(cols), Map._longest_circular_run)
        except Exception as e:
            print(f"[centerline] skipped ({e}); showing image only")

    fig, ax = plt.subplots(figsize=(12, 12 * h / max(w, 1)))
    ax.imshow(raw, cmap="gray", origin="upper")
    if cl is not None:
        ax.plot(cl[0], cl[1], "-", lw=1.5, c="#ff3a6b", label="centerline")
    arc = ax.scatter([], [], s=12, c="lime", zorder=5, label="arc in crop")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")

    hint = "drag a box  →  --crop string prints in the terminal"
    ax.set_title(hint)
    print(f"\nimage: {img_path.name}  ({w}×{h} px)\n{hint}\n")

    def on_select(eclick, erelease):
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        fx0, fx1 = np.clip([x0 / w, x1 / w], 0.0, 1.0)
        fy0, fy1 = np.clip([y0 / h, y1 / h], 0.0, 1.0)
        crop = f"{fx0:.3f},{fy0:.3f},{fx1:.3f},{fy1:.3f}"
        msg = f"--crop {crop}"

        if cl is not None:
            cols, rows, n, longest_run = cl
            in_box = (
                (cols >= fx0 * w) & (cols <= fx1 * w)
                & (rows >= fy0 * h) & (rows <= fy1 * h)
            )
            if in_box.any():
                lo, length = longest_run(in_box)
                idx = (lo + np.arange(length)) % n
                arc.set_offsets(np.column_stack([cols[idx], rows[idx]]))
                msg += f"   arc {length}/{n} = {length / n:.0%} of loop"
            else:
                arc.set_offsets(np.empty((0, 2)))
                msg += "   ⚠ no centerline inside box"
        else:
            msg += f"   px {int(x0)},{int(y0)} → {int(x1)},{int(y1)}"

        ax.set_title(msg)
        fig.canvas.draw_idle()
        print(msg)

    # Keep a reference so the widget isn't garbage-collected.
    selector = RectangleSelector(  # noqa: F841
        ax, on_select, useblit=False, interactive=True, button=[1]
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run(main)
