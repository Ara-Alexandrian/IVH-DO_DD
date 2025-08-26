#!/usr/bin/env python3
"""Final comprehensive test on all HIP patients."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import numpy as np
from pathlib import Path
from dicom_utils import load_dicom_series_to_hu
from metal_detection import detect_metal_volume, create_affine_from_dicom_meta
from contour_operations import (create_bright_artifact_mask, create_dark_artifact_mask,
                               create_bone_mask, refine_mask)
from visualization import create_overlay_image, create_multi_slice_view

def test_patient_comprehensive(patient_name):
    """Comprehensive test for a patient."""
    print(f"\n{'='*60}")
    print(f"Testing: {patient_name}")
    print('='*60)
    
    # Load data
    data_dir = Path("data")
    patient_path = data_dir / patient_name
    ct_dirs = [d for d in patient_path.iterdir() if d.is_dir() and "CT" in d.name]
    
    if not ct_dirs:
        print(f"ERROR: No CT directory found")
        return False
        
    ct_dir = ct_dirs[0]
    
    try:
        # Load DICOM
        ct_volume, ct_metadata = load_dicom_series_to_hu(str(ct_dir))
        print(f"✓ Loaded volume: {ct_volume.shape}, spacing: {ct_metadata['spacing']}")
        print(f"  HU range: [{ct_volume.min():.0f}, {ct_volume.max():.0f}]")
        
        # Create affine
        affine = create_affine_from_dicom_meta(ct_metadata)
        print(f"✓ Created affine matrix")
        
        # Detect metal
        result = detect_metal_volume(ct_volume, ct_metadata['spacing'], margin_cm=3.0)
        
        if result['mask'] is None:
            print("⚠ No metal detected - skipping segmentation")
            return True
            
        metal_mask = result['mask']
        roi_bounds = result['roi_bounds']
        print(f"✓ Metal detection: {np.sum(metal_mask):,} voxels")
        print(f"  Center: {result['center_coords']}")
        print(f"  ROI: Z[{roi_bounds['z_min']}-{roi_bounds['z_max']}], "
              f"Y[{roi_bounds['y_min']}-{roi_bounds['y_max']}], "
              f"X[{roi_bounds['x_min']}-{roi_bounds['x_max']}]")
        
        # Verify ROI bounds are valid
        for axis in ['x', 'y', 'z']:
            if roi_bounds[f'{axis}_min'] >= roi_bounds[f'{axis}_max']:
                print(f"ERROR: Invalid ROI bounds for {axis} axis")
                return False
        
        # Segment artifacts
        bright_mask = create_bright_artifact_mask(ct_volume, metal_mask, roi_bounds, 800, 3000)
        dark_mask = create_dark_artifact_mask(ct_volume, metal_mask, roi_bounds, -200)
        bone_mask = create_bone_mask(ct_volume, metal_mask, bright_mask, dark_mask, roi_bounds, 150, 1500)
        
        # Refine masks
        masks = {
            'metal': refine_mask(metal_mask),
            'bright_artifacts': refine_mask(bright_mask),
            'dark_artifacts': refine_mask(dark_mask),
            'bone': refine_mask(bone_mask)
        }
        
        print(f"✓ Segmentation results:")
        for name, mask in masks.items():
            voxels = np.sum(mask)
            print(f"  {name}: {voxels:,} voxels")
        
        # Test visualizations
        current_slice = ct_volume.shape[0] // 2
        
        # Single slice
        slice_masks = {name: mask[current_slice] for name, mask in masks.items()}
        roi_tuple = (roi_bounds['y_min'], roi_bounds['y_max'], roi_bounds['x_min'], roi_bounds['x_max'])
        
        fig1 = create_overlay_image(ct_volume[current_slice], slice_masks, roi_tuple, current_slice)
        print(f"✓ Single slice visualization created")
        
        # Multi-slice
        n_slices = min(8, roi_bounds['z_max'] - roi_bounds['z_min'])
        slice_indices = np.linspace(roi_bounds['z_min'], roi_bounds['z_max']-1, n_slices, dtype=int)
        fig2 = create_multi_slice_view(ct_volume, masks, slice_indices, roi_bounds)
        print(f"✓ Multi-slice visualization created")
        
        # Check thresholds
        if result.get('slice_thresholds'):
            valid_thresholds = [t for t in result['slice_thresholds'] if t['thresholds'] is not None]
            print(f"✓ Adaptive thresholds: {len(valid_thresholds)} slices with valid thresholds")
        
        print(f"\n✅ All tests passed for {patient_name}")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all patients."""
    print("CT Metal Artifact Characterization - Final Test Suite")
    print("Testing all functionality on all HIP patients")
    
    patients = ["HIP1 Patient", "HIP2 Patient", "HIP3 Patient", "HIP4 Patient"]
    results = []
    
    for patient in patients:
        success = test_patient_comprehensive(patient)
        results.append((patient, success))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for patient, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{patient}: {status}")
    
    all_passed = all(success for _, success in results)
    print("\n" + ("✅ ALL TESTS PASSED!" if all_passed else "❌ SOME TESTS FAILED"))


if __name__ == "__main__":
    main()