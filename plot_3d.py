import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
try:
    import nrrd
except ImportError:
    pass

def load_mask(file_path: str) -> np.ndarray:
    path = Path(file_path)
    if path.suffix == '.nrrd':
        data, _ = nrrd.read(str(path))
    else:
        data = nib.load(str(path)).get_fdata()
    return np.round(data).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Plot 3D segmentation mask")
    parser.add_argument("--anatomy", required=False, type=str, help="Path to anatomy mask (e.g. TotalSegmentator output)")
    parser.add_argument("--vessels", required=False, type=str, help="Path to coronary vessels mask")
    parser.add_argument("--output", default="3d_segmentation.png", type=str, help="Output PNG path")
    parser.add_argument("--step_size", type=int, default=2, help="Step size for marching cubes (higher = faster to render, but lower res)")
    parser.add_argument("--views", type=int, default=1, help="Number of views to render (1 or 4)")
    args = parser.parse_args()

    if not args.anatomy and not args.vessels:
        print("Error: Must provide at least one of --anatomy or --vessels")
        return

    data = None

    if args.anatomy:
        print(f"Loading anatomy from {args.anatomy}...")
        data = load_mask(args.anatomy)
        
    if args.vessels:
        print(f"Loading vessels from {args.vessels}...")
        vessels_vol = load_mask(args.vessels)
        if data is None:
            data = np.zeros_like(vessels_vol)
        
        # TotalSegmentator uses classes 1-7. We assign class 9 to the coronary vessels
        # so they don't overwrite the fat (class 8) or myocardium (class 1 in TS).
        data[vessels_vol > 0] = 9
            
    unique_classes = [c for c in np.unique(data) if c != 0]
    if not unique_classes:
        print("Warning: No foreground classes found in the mask(s)!")
        return

    # Updated to match TotalSegmentator 'heartchambers_highres' classes + our fat/vessels
    CLASS_INFO = {
        1: {"name": "Myocardium",      "color": [0.9, 0.5, 0.5], "alpha": 0.4},
        2: {"name": "Left Atrium",     "color": [0.1, 0.1, 0.9], "alpha": 0.4},
        3: {"name": "Left Ventricle",  "color": [0.1, 0.9, 0.1], "alpha": 0.4},
        4: {"name": "Right Atrium",    "color": [0.1, 0.9, 0.9], "alpha": 0.4},
        5: {"name": "Right Ventricle", "color": [0.9, 0.9, 0.1], "alpha": 0.4},
        6: {"name": "Aorta",           "color": [0.5, 0.1, 0.9], "alpha": 0.4},
        7: {"name": "Pulmonary Artery","color": [0.9, 0.5, 0.9], "alpha": 0.4},
        8: {"name": "Heart & Fat",     "color": [0.9, 0.9, 0.6], "alpha": 0.15}, # Translucent pale yellow
        9: {"name": "Coronary Artery", "color": [0.9, 0.1, 0.1], "alpha": 1.0},  # Bright Red, solid
    }

    print("Extracting 3D surface meshes (this might take a moment)...")
    mesh_data = []
    legend_patches = []
    for class_idx in unique_classes:
        mask = (data == class_idx).astype(np.uint8)
        verts, faces, normals, values = measure.marching_cubes(mask, level=0.5, step_size=args.step_size)
        mesh_data.append((class_idx, verts, faces))
        
        info = CLASS_INFO.get(class_idx, {"name": f"Class {class_idx}", "color": np.random.rand(3)})
        legend_patches.append(mpatches.Patch(color=info["color"], label=info["name"]))

    print("Rendering 3D plot(s)...")
    if args.views == 4:
        fig = plt.figure(figsize=(20, 20))
        axes = [
            fig.add_subplot(221, projection='3d'),
            fig.add_subplot(222, projection='3d'),
            fig.add_subplot(223, projection='3d'),
            fig.add_subplot(224, projection='3d')
        ]
        view_angles = [(20, -60), (0, -90), (0, 0), (90, 0)]
        titles = ['Isometric View', 'Anterior (Front) View', 'Lateral (Side) View', 'Superior (Top) View']
    else:
        fig = plt.figure(figsize=(10, 10))
        axes = [fig.add_subplot(111, projection='3d')]
        view_angles = [(None, None)]
        titles = ['3D Multi-Class Segmentation']

    for i, (ax, (elev, azim)) in enumerate(zip(axes, view_angles)):
        for class_idx, verts, faces in mesh_data:
            info = CLASS_INFO.get(class_idx, {"color": np.random.rand(3)})
            alpha_val = info.get("alpha", 0.4)
            
            # Poly3DCollection must be created fresh for each axis
            mesh_col = Poly3DCollection(verts[faces], alpha=alpha_val)
            mesh_col.set_facecolor(info["color"])
            ax.add_collection3d(mesh_col)
            
        ax.set_xlim(0, data.shape[0])
        ax.set_ylim(0, data.shape[1])
        ax.set_zlim(0, data.shape[2])
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title(titles[i])
        
        if elev is not None and azim is not None:
            ax.view_init(elev=elev, azim=azim)

    # Add legend to the first axis
    axes[0].legend(handles=legend_patches, loc='upper right')

    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Successfully saved 3D plot to {args.output}")

if __name__ == "__main__":
    main()