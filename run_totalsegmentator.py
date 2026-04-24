"""Run TotalSegmentator V2 on ASOCA dataset to extract heart structures.

Usage for a single file:
    python run_totalsegmentator.py --input ./ASOCA/ASOCA/Normal/CTCA/Normal_1.nrrd --output ./Normal_1_heart.nii.gz

Usage for a directory:
    python run_totalsegmentator.py --input ./ASOCA/ASOCA --output ./ASOCA_TS_Outputs
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    import nrrd
except ImportError:
    raise ImportError("pynrrd is required: pip install pynrrd")

try:
    from totalsegmentator.python_api import totalsegmentator
except ImportError:
    raise ImportError("TotalSegmentator is required: pip install TotalSegmentator")


def load_image_as_nifti(path: Path) -> nib.Nifti1Image:
    """Loads an NRRD or NIfTI file and returns a nibabel Nifti1Image."""
    if path.suffix == ".nrrd":
        data, header = nrrd.read(str(path))
        # Convert NRRD space directions to affine matrix
        space_dirs = header.get("space directions", np.eye(3))
        spacing = tuple(abs(space_dirs[i][i]) for i in range(3))
        affine = np.diag([*spacing, 1.0])
        return nib.Nifti1Image(data, affine)
    else:
        return nib.load(str(path))


def main():
    parser = argparse.ArgumentParser(description="TotalSegmentator Heart Extraction")
    parser.add_argument("--input", type=str, required=True, help="Input NRRD file or ASOCA directory")
    parser.add_argument("--output", type=str, required=True, help="Output NIfTI file or directory")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode (3mm isotropic, much faster)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    # The specific cardiac ROIs we want to extract
    heart_rois = [
        "heart_myocardium",
        "heart_atrium_left",
        "heart_ventricle_left",
        "heart_atrium_right",
        "heart_ventricle_right",
        "aorta",
        "pulmonary_artery"
    ]

    if in_path.is_file():
        print(f"Processing single file: {in_path}")
        img = load_image_as_nifti(in_path)
        
        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        totalsegmentator(
            input=img,
            output=str(out_path),
            task="heartchambers_highres",
            ml=True,  # Save all classes in a single multi-label NIfTI
            fast=args.fast
        )
        print(f"SUCCESS! Saved to {out_path}")
        return

    # If a directory is provided, discover all NRRDs and process them
    out_path.mkdir(parents=True, exist_ok=True)
    nrrd_files = list(in_path.rglob("*.nrrd"))
    
    # Filter strictly for CTCA (image) files so we don't segment the ground truth masks
    ctca_files = [f for f in nrrd_files if "CTCA" in f.parts or "image" in f.name.lower()]
    if not ctca_files:
        ctca_files = nrrd_files  # Fallback
        
    print(f"Found {len(ctca_files)} image files to process in {in_path}")
    
    for i, file_path in enumerate(ctca_files, 1):
        print(f"\n[{i}/{len(ctca_files)}] Processing {file_path.name}...")
        img = load_image_as_nifti(file_path)
        
        # Construct output filename
        out_name = file_path.name.replace(".nrrd", "_TS_heart.nii.gz")
        out_file = out_path / out_name
        
        totalsegmentator(
            input=img,
            output=str(out_file),
            task="heartchambers_highres",
            ml=True,
            fast=args.fast
        )
        print(f"Saved -> {out_file}")

if __name__ == "__main__":
    main()