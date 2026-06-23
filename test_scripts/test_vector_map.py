import sys
import matplotlib.pyplot as plt
from pathlib import Path
# Resolve the absolute path to the project root (one folder up from this script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Now Python can see the include folder perfectly!
from include.map import Map

def test_map_vectorization(map_yaml_path: str):
    path = Path(map_yaml_path)
    if not path.exists():
        print(f"Error: Path '{map_yaml_path}' does not exist.")
        return

    # Initialize the map (this runs your pipeline and wall extraction)
    track = Map(path)
    
    plt.figure(figsize=(10, 10))
    
    # 1. Plot the extracted wall segments (Red)
    segments = track.wall_segments
    print(f"Plotting {len(segments)} vectorized wall segments...")
    
    for i, seg in enumerate(segments):
        x1, y1, x2, y2 = seg
        # Only label the first segment to keep the plot legend clean
        label = "Wall Segments" if i == 0 else ""
        plt.plot([x1, x2], [y1, y2], color='red', linewidth=1.5, label=label)
        
    # 2. Plot the precomputed centerline for spatial reference (Blue)
    if hasattr(track, 'centerline') and track.centerline is not None:
        plt.plot(track.centerline[:, 0], track.centerline[:, 1], 
                 color='blue', linestyle='--', linewidth=1.2, label="Centerline")

    # Aesthetics
    plt.title(f"Vectorized Map Verification: {track.path_name}", fontsize=12, fontweight='bold')
    plt.xlabel("World X (meters)")
    plt.ylabel("World Y (meters)")
    plt.axis('equal')  # Critical for 1:1 pixel aspect ratio scaling
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc="upper right")
    
    print("Opening viewer window. Close the window to exit script.")
    plt.show()

if __name__ == "__main__":
    # Point this to any YAML file in your maps directory to test it
    test_map_vectorization("./maps/stata_basement.yaml")