#!/usr/bin/env python3
"""Test all features of the application."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import numpy as np
from pathlib import Path
from dicom_utils import load_dicom_series_to_hu, create_metal_mask_from_rtstruct
from metal_detection import (detect_metal_volume, create_affine_from_dicom_meta, 
                            save_mask_as_nifti, detect_metal_adaptive)
from contour_operations import (create_bright_artifact_mask, create_dark_artifact_mask,
                               create_bone_mask, save_all_contours_as_nifti, refine_mask)
from visualization import (create_overlay_image, create_histogram, fig_to_base64, 
                          create_multi_slice_view, visualize_star_profiles,
                          plot_threshold_evolution)

def test_all_features(patient_name="HIP1 Patient"):
    """Test all major features."""
    print(f"\nTesting all features for {patient_name}")
    print("="*60)
    
    # Load data
    data_dir = Path("data")
    patient_path = data_dir / patient_name
    ct_dirs = [d for d in patient_path.iterdir() if d.is_dir() and "CT" in d.name]
    ct_dir = ct_dirs[0]
    
    ct_volume, ct_metadata = load_dicom_series_to_hu(str(ct_dir))
    print(f"✓ Data loaded: {ct_volume.shape}")
    
    # Test affine creation
    affine = create_affine_from_dicom_meta(ct_metadata)
    print(f"✓ Affine matrix created: {affine.shape}")
    
    # Test metal detection
    result = detect_metal_volume(ct_volume, ct_metadata['spacing'], margin_cm=3.0)
    metal_mask = result['mask']
    roi_bounds = result['roi_bounds']
    print(f"✓ Metal detection: {np.sum(metal_mask):,} voxels")
    
    # Test adaptive thresholding on single slice
    current_slice = ct_volume.shape[0] // 2
    roi_bounds_2d = {
        'y_min': roi_bounds['y_min'],
        'y_max': roi_bounds['y_max'],
        'x_min': roi_bounds['x_min'],
        'x_max': roi_bounds['x_max']
    }
    
    slice_result = detect_metal_adaptive(
        ct_volume[current_slice],
        roi_bounds_2d,
        ct_metadata['spacing'][1:]
    )
    print(f"✓ Adaptive thresholding: thresholds={slice_result.get('thresholds', 'None')}")
    
    # Test artifact segmentation
    bright_mask = create_bright_artifact_mask(ct_volume, metal_mask, roi_bounds, 800, 3000)
    dark_mask = create_dark_artifact_mask(ct_volume, metal_mask, roi_bounds, -200)
    bone_mask = create_bone_mask(ct_volume, metal_mask, bright_mask, dark_mask, roi_bounds, 150, 1500)
    
    masks = {
        'metal': refine_mask(metal_mask),
        'bright_artifacts': refine_mask(bright_mask),
        'dark_artifacts': refine_mask(dark_mask),
        'bone': refine_mask(bone_mask)
    }
    print(f"✓ Segmentation complete")
    
    # Test single slice visualization
    slice_masks = {name: mask[current_slice] for name, mask in masks.items()}
    roi_tuple = (roi_bounds['y_min'], roi_bounds['y_max'], roi_bounds['x_min'], roi_bounds['x_max'])
    fig = create_overlay_image(ct_volume[current_slice], slice_masks, roi_tuple, current_slice)
    print(f"✓ Single slice visualization")
    
    # Test histogram creation
    hu_values = ct_volume[current_slice][slice_masks['metal']]
    if hu_values.size > 0:
        fig = create_histogram(hu_values, "Metal", "red")
        if fig:
            print(f"✓ Histogram creation")
    
    # Test multi-slice view
    slice_indices = np.linspace(roi_bounds['z_min'], roi_bounds['z_max']-1, 8, dtype=int)
    fig = create_multi_slice_view(ct_volume, masks, slice_indices, roi_bounds)
    print(f"✓ Multi-slice view")
    
    # Test star profile visualization
    if slice_result.get('profiles') and np.any(slice_masks['metal']):
        y_coords, x_coords = np.where(slice_masks['metal'])
        center_y = int(np.mean(y_coords))
        center_x = int(np.mean(x_coords))
        
        fig = visualize_star_profiles(
            ct_volume[current_slice],
            slice_result['profiles'],
            (center_y, center_x),
            roi_bounds_2d,
            slice_result.get('thresholds')
        )
        print(f"✓ Star profile visualization")
    
    # Test threshold evolution plot
    if result.get('slice_thresholds'):
        fig = plot_threshold_evolution(result['slice_thresholds'])
        print(f"✓ Threshold evolution plot")
    
    # Test NIFTI export
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    output_prefix = str(output_dir / "test")
    
    save_all_contours_as_nifti(masks, affine, output_prefix)
    print(f"✓ NIFTI export")
    
    # Check if files were created
    nifti_files = list(output_dir.glob("*.nii.gz"))
    print(f"  Created {len(nifti_files)} NIFTI files")
    
    print("\n✓ All features tested successfully!")
    return True


if __name__ == "__main__":
    try:
        test_all_features("HIP1 Patient")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()