#!/usr/bin/env python3
"""
Test script to debug segmentation algorithm
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

import numpy as np
from dicom_utils import load_dicom_series_to_hu
from core.metal_detection import MetalDetector, MetalDetectionMethod
from contour_operations import create_russian_doll_segmentation

def test_segmentation():
    """Test the segmentation pipeline on HIP1 patient"""
    
    # Load DICOM data
    print("Loading DICOM data...")
    dicom_dir = "/config/workspace/github/IVH-DO_DD/data/HIP1 Patient/DOE^JOHN_ZZ.DD.TEST1_CT_2025-04-22_111408_RTP..Simulation..Complex-.three.or.more.treatment.areas.Adaptive_RTP.PRIMARY_n169__00000"
    
    try:
        ct_volume, metadata = load_dicom_series_to_hu(dicom_dir)
        print(f"Loaded CT volume: shape={ct_volume.shape}, spacing={metadata['spacing']}")
        print(f"HU range: [{np.min(ct_volume):.0f}, {np.max(ct_volume):.0f}]")
    except Exception as e:
        print(f"Error loading DICOM: {e}")
        return
    
    # Fix spacing to be positive
    spacing = np.abs(metadata['spacing'])
    
    # Step 1: Detect metal
    print("\n1. Detecting metal...")
    detector = MetalDetector(MetalDetectionMethod.ADAPTIVE_3D)
    result = detector.detect(ct_volume, spacing)
    
    if result['mask'] is None:
        print("No metal detected!")
        return
    
    metal_mask = result['mask']
    roi_bounds = result['roi_bounds']
    print(f"Metal detected: {np.sum(metal_mask)} voxels")
    print(f"ROI bounds: z=[{roi_bounds['z_min']}, {roi_bounds['z_max']}], "
          f"y=[{roi_bounds['y_min']}, {roi_bounds['y_max']}], "
          f"x=[{roi_bounds['x_min']}, {roi_bounds['x_max']}]")
    
    # Step 2: Segment artifacts using Russian Doll
    print("\n2. Segmenting artifacts...")
    
    # Use reasonable threshold ranges
    segmentation_result = create_russian_doll_segmentation(
        ct_volume,
        metal_mask,
        spacing,  # Use fixed spacing
        roi_bounds,
        dark_threshold_low=-500,  # Don't include air
        dark_threshold_high=-150,
        bone_threshold_low=150,
        bone_threshold_high=1500,
        bright_threshold_low=800,
        bright_threshold_high=2000,
        bright_artifact_max_distance_cm=10.0,
        use_fast_mode=True,
        use_enhanced_mode=False
    )
    
    print("\n3. Results:")
    if segmentation_result:
        for mask_name in ['dark_artifacts', 'bone', 'bright_artifacts']:
            if mask_name in segmentation_result:
                mask = segmentation_result[mask_name]
                if mask is not None:
                    voxel_count = np.sum(mask)
                    print(f"  {mask_name}: {voxel_count} voxels")
                    
                    # Check if mask overlaps with ROI
                    roi_mask = np.zeros_like(ct_volume, dtype=bool)
                    roi_mask[roi_bounds['z_min']:roi_bounds['z_max'],
                            roi_bounds['y_min']:roi_bounds['y_max'],
                            roi_bounds['x_min']:roi_bounds['x_max']] = True
                    roi_voxels = np.sum(mask & roi_mask)
                    print(f"    - Voxels in ROI: {roi_voxels}")
                else:
                    print(f"  {mask_name}: None")
            else:
                print(f"  {mask_name}: Not in result")
    else:
        print("Segmentation returned None!")
    
    # Check what HU values are actually in the ROI
    print("\n4. HU distribution in ROI:")
    roi_data = ct_volume[roi_bounds['z_min']:roi_bounds['z_max'],
                         roi_bounds['y_min']:roi_bounds['y_max'],
                         roi_bounds['x_min']:roi_bounds['x_max']]
    
    # Exclude metal from statistics
    metal_roi = metal_mask[roi_bounds['z_min']:roi_bounds['z_max'],
                           roi_bounds['y_min']:roi_bounds['y_max'],
                           roi_bounds['x_min']:roi_bounds['x_max']]
    non_metal_roi = roi_data[~metal_roi]
    
    print(f"  Non-metal voxels in ROI: {len(non_metal_roi)}")
    if len(non_metal_roi) > 0:
        print(f"  HU range (non-metal): [{np.min(non_metal_roi):.0f}, {np.max(non_metal_roi):.0f}]")
        print(f"  HU percentiles (non-metal):")
        for p in [1, 5, 25, 50, 75, 95, 99]:
            print(f"    {p}%: {np.percentile(non_metal_roi, p):.0f} HU")
        
        # Count voxels in different ranges
        dark_count = np.sum((non_metal_roi >= -500) & (non_metal_roi <= -150))
        bone_count = np.sum((non_metal_roi >= 150) & (non_metal_roi <= 1500))
        bright_count = np.sum((non_metal_roi >= 800) & (non_metal_roi <= 2000))
        
        print(f"\n  Voxels in threshold ranges (non-metal):")
        print(f"    Dark [-500, -150]: {dark_count}")
        print(f"    Bone [150, 1500]: {bone_count}")
        print(f"    Bright [800, 2000]: {bright_count}")

if __name__ == "__main__":
    test_segmentation()