import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typer import run
from yaml import safe_load

def visualize_raceline(map_yaml: str):
    path = Path(map_yaml)
    
    # 1. Load the Map Metadata and Image
    try:
        meta = safe_load(path.read_text())
        img_path = path.parent / meta["image"]
        track_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        h, w = track_img.shape
        res = float(meta["resolution"])
        ox, oy, _ = meta["origin"]
    except Exception as e:
        print(f"Failed to load map yaml or image: {e}")
        return

    # 2. Load the Optimized Raceline
    csv_path = Path("racelines") / f"{path.stem}_raceline.csv"
    if not csv_path.exists():
        print(f"Error: Could not find '{csv_path}'. Did you run compute_raceline.py first?")
        return
        
    df = pd.read_csv(csv_path)
    x = df['x'].values
    y = df['y'].values
    v_target = df['v_target'].values

    # 3. Convert World Coordinates (meters) to Pixel Coordinates
    # Matches the w2p transform logic used in your Warp environment
    px = (x - ox) / res
    py = h - 1 - (y - oy) / res

    # 4. Plotting
    plt.figure(figsize=(6, 6), dpi=50)
    
    # Display the track as a dark background
    plt.imshow(track_img, cmap='gray')
    
    # Scatter the raceline points, colored by target velocity
    scatter = plt.scatter(px, py, c=v_target, cmap='turbo', s=10, zorder=2)
    
    # Add a colorbar and labels
    cbar = plt.colorbar(scatter, fraction=0.046, pad=0.04)
    cbar.set_label('Target Velocity (m/s)', rotation=270, labelpad=15)
    
    plt.title(f"Optimal Raceline & Velocity Profile: {path.stem.upper()}")
    plt.axis('off')
    
    # Save the plot to the racelines directory
    out_img = Path("racelines") / f"{path.stem}_visualization.png"
    plt.savefig(out_img, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {out_img}")
    
    # Display in the notebook/UI
    plt.show()

if __name__ == "__main__":
    run(visualize_raceline)