"""End-to-End Inference Pipeline for ASOCA

Stages:
1. General Heart & Fat: Extracts overall heart and epicardial fat, combined.
2. Multi-Label Anatomy: TotalSegmentator High-Res for chambers, aorta, pulmonary artery.
3. Heart Cropping: Computes a bounding box directly from the high-res chambers.
4. Coronary Vessels: Ensembles nnU-Net and TotalSegmentator.
5. Output: Saves every anatomical structure as a separate binary NIfTI file.

Usage:
    export nnUNet_results=/workspace/Project/nnUNet_results
    python run_inference_pipeline.py \
        --input ./ASOCA/ASOCA/Normal/CTCA/Normal_1.nrrd \
        --output ./Normal_1_multiclass.nii.gz \
        --dataset_id 100
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.ndimage as ndimage
import nibabel as nib
try:
    import nrrd
except ImportError:
    raise ImportError("pynrrd required: pip install pynrrd")

try:
    from totalsegmentator.python_api import totalsegmentator
except ImportError:
    totalsegmentator = None

# Final output class mapping
FINAL_CLASSES = {
    "myocardium": 1,
    "left_atrium": 2,       
    "left_ventricle": 3,
    "right_atrium": 4,
    "right_ventricle": 5,
    "aorta": 6,
    "pulmonary_artery": 7,
    "heart_and_fat": 8,
    "coronary_artery": 9,
}


def compute_bbox(mask: np.ndarray, margin: int = 20):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + margin + 1, np.array(mask.shape))
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))


def save_nifti(data: np.ndarray, spacing: tuple, path: str):
    affine = np.diag([*spacing, 1.0])
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)


def main():
    parser = argparse.ArgumentParser(description="Run ASOCA Multi-Class Inference Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input NRRD or NIfTI")
    parser.add_argument("--output", type=str, required=True, help="Path to output NIfTI mask")
    parser.add_argument("--dataset_id", type=int, default=100, help="nnU-Net Dataset ID (e.g., 100)")
    parser.add_argument("--fold", type=str, default="0", help="nnU-Net fold to use (0-4, or 'all')")
    parser.add_argument("--margin", type=int, default=20, help="Margin around the heart for cropping")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pth", help="nnU-Net checkpoint to use")
    args = parser.parse_args()

    original_env = os.environ.copy()
    if totalsegmentator is None:
        raise ImportError("TotalSegmentator is required for the ensemble. Run: pip install TotalSegmentator")

    if "nnUNet_results" not in original_env:
        raise EnvironmentError("nnUNet_results environment variable is not set. Please set it before running.")

    input_path = Path(args.input)
    
    # ---------------------------------------------------------
    # 1. Load Original Image
    # ---------------------------------------------------------
    print(f"\n[1/5] Loading {input_path.name}...")
    if input_path.suffix == ".nrrd":
        img_data, img_header = nrrd.read(str(input_path))
        img_data = img_data.astype(np.float32)
        space_dirs = img_header.get("space directions", np.eye(3))
        spacing = tuple(abs(space_dirs[i][i]) for i in range(3))
    else:
        img = nib.load(str(input_path))
        img_data = img.get_fdata().astype(np.float32)
        spacing = img.header.get_zooms()[:3]
        
    print(f"      Shape: {img_data.shape}, Spacing: {spacing}")
    
    affine = np.diag([*spacing, 1.0])
    nib_img = nib.Nifti1Image(img_data, affine)

    # ---------------------------------------------------------
    # 2. Extract General Heart & Epicardial Fat
    # ---------------------------------------------------------
    print("\n[2/5] Extracting General Heart & Epicardial Fat...")
    ts_total_nifti = totalsegmentator(
        input=nib_img,
        output=None,
        task="total",
        roi_subset=["heart"],
        ml=True,
        fast=True
    )
    heart_mask = (ts_total_nifti.get_fdata() > 0).astype(np.uint8)

    iterations = int(round(10.0 / spacing[0]))
    dilated_heart = ndimage.binary_dilation(heart_mask, iterations=iterations)
    heart_shell = dilated_heart & ~heart_mask.astype(bool)
    
    # Threshold for fat (-190 to -30 HU)
    fat_mask = heart_shell & (img_data >= -190) & (img_data <= -30)
    combined_heart_and_fat = heart_mask | fat_mask

    # ---------------------------------------------------------
    # 3. Extract High-Res Chambers (TotalSegmentator)
    # ---------------------------------------------------------
    print("\n[3/5] Extracting High-Res Chambers (TotalSegmentator)...")
    ts_chambers_nifti = totalsegmentator(
        input=nib_img,
        output=None,
        task="heartchambers_highres",
        ml=True
    )
    chambers_pred = np.round(ts_chambers_nifti.get_fdata()).astype(np.uint8)

    # Compute Bounding Box from the high-res chambers
    bbox = compute_bbox(chambers_pred, margin=args.margin)
    if bbox is None:
        raise ValueError("TotalSegmentator failed to find the heart chambers. Cannot proceed with nnU-Net cropping.")
    
    img_crop = img_data[bbox]
    print(f"      Heart BBox extracted. Cropped shape: {img_crop.shape}")

    # ---------------------------------------------------------
    # 4. Coronary Vessels (nnU-Net & TotalSegmentator)
    # ---------------------------------------------------------
    print("\n[4/5] Extracting Coronary Arteries (nnU-Net & TotalSegmentator)...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        in_dir = temp_dir_path / "inputs"
        out_dir = temp_dir_path / "outputs"
        in_dir.mkdir()
        out_dir.mkdir()

        # nnU-Net expects specific naming conventions (e.g., case_0000.nii.gz)
        temp_in_file = in_dir / "target_0000.nii.gz"
        temp_out_file = out_dir / "target.nii.gz"
        
        save_nifti(img_crop, spacing, str(temp_in_file))
        
        # Execute nnU-Net via subprocess
        cmd = [
            "nnUNetv2_predict",
            "-i", str(in_dir),
            "-o", str(out_dir),
            "-d", str(args.dataset_id),
            "-c", "3d_fullres",
            "-f", args.fold,
            "-chk", args.checkpoint
        ]
        
        print(f"      Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=original_env)
        
        # Load the nnU-Net prediction back into memory
        nnunet_crop_pred = nib.load(str(temp_out_file)).get_fdata().astype(np.uint8)

    ts_vessels_nifti = totalsegmentator(
        input=nib_img,
        output=None,
        task="coronary_arteries",
        ml=True
    )
    ts_vessels_pred = np.round(ts_vessels_nifti.get_fdata()).astype(np.uint8)

    # ---------------------------------------------------------
    # 5. Layering, Ensembling, and Saving Individual Masks
    # ---------------------------------------------------------
    print("\n[5/5] Layering, Ensembling, and Saving Individual Masks...")
    anatomy_mask = np.zeros_like(img_data, dtype=np.uint8)

    # Layer 1: Combined Heart & Fat
    anatomy_mask[combined_heart_and_fat > 0] = FINAL_CLASSES["heart_and_fat"]

    # Layer 2: High-Res Chambers and Arteries (1-7)
    valid_chamber_classes = [1, 2, 3, 4, 5, 6, 7]
    for c in valid_chamber_classes:
        anatomy_mask[chambers_pred == c] = c

    # Layer 3: Vessels Ensemble
    nnunet_full_pred = np.zeros_like(img_data, dtype=np.uint8)
    nnunet_full_pred[bbox] = nnunet_crop_pred
    ensemble_pred = (nnunet_full_pred > 0) | (ts_vessels_pred > 0)

    # Determine output paths
    out_path = Path(args.output)
    if out_path.name.endswith(".nii.gz"):
        base_name = out_path.name[:-7]
        ext = ".nii.gz"
    else:
        base_name = out_path.stem
        ext = out_path.suffix

    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("SUCCESS! Individual masks saved to:")

    # Save each class as a separate binary mask
    for class_name, class_idx in FINAL_CLASSES.items():
        if class_name == "coronary_artery":
            mask_data = (ensemble_pred > 0).astype(np.uint8)
        else:
            mask_data = (anatomy_mask == class_idx).astype(np.uint8)
            
        save_path = out_dir / f"{base_name}_{class_name}{ext}"
        save_nifti(mask_data, spacing, str(save_path))
        print(f"  {class_name}: {save_path}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()