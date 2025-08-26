#!/usr/bin/env python3
"""
Quick test script for Russian doll segmentation
"""
import sys
sys.path.append('app')

import numpy as np
from pathlib import Path
from dicom_utils import load_dicom_series_to_hu
from metal_detection_v3 import detect_metal_adaptive_3d
from contour_operations import create_russian_doll_segmentation
import matplotlib.pyplot as plt

def test_russian_doll():
    # Load test data
    data_dir = Path("data")
    patient_dirs = list(data_dir.glob("HIP*"))
    
    # Filter out text files
    patient_dirs = [d for d in patient_dirs if d.is_dir()]
    
    if not patient_dirs:
        print("No patient data found")
        return
    
    # Use first patient
    patient_dir = patient_dirs[0]
    print(f"Testing with patient: {patient_dir.name}")
    
    # Find CT directory - look for directories with DICOM files
    ct_dirs = []
    for subdir in patient_dir.iterdir():
        if subdir.is_dir():
            dcm_files = list(subdir.glob("*.dcm"))
            if len(dcm_files) > 10:  # Likely a CT scan
                ct_dirs.append(subdir)
    
    if not ct_dirs:
        print("No CT directory found")
        return
    
    ct_dir = ct_dirs[0]
    print(f"Loading CT data from: {ct_dir}")
    
    # Load CT volume
    ct_volume, ct_metadata = load_dicom_series_to_hu(str(ct_dir))
    if ct_volume is None:
        print("Failed to load CT data")
        return
    
    print(f"Loaded CT volume: {ct_volume.shape}")
    
    # Step 1: Detect metal
    print("\nStep 1: Detecting metal...")
    metal_result = detect_metal_adaptive_3d(
        ct_volume,
        ct_metadata['spacing'],
        fw_percentage=75,
        margin_cm=2.0,
        intensity_percentile=99.5
    )
    
    metal_mask = metal_result['mask']
    metal_voxels = np.sum(metal_mask)
    print(f"Found {metal_voxels:,} metal voxels")
    
    if metal_voxels == 0:
        print("No metal detected, cannot proceed with segmentation")
        return
    
    # Step 2: Run Russian doll segmentation
    print("\nStep 2: Running Russian doll segmentation...")
    segmentation_result = create_russian_doll_segmentation(
        ct_volume,
        metal_mask,
        ct_metadata['spacing'],
        roi_bounds=metal_result['roi_bounds'],
        dark_threshold_high=-150,
        bone_threshold_low=300,
        bone_threshold_high=1500,
        bright_artifact_max_distance_cm=10.0
    )
    
    # Print results
    print("\nSegmentation Results:")
    for mask_name, mask in segmentation_result.items():
        if isinstance(mask, np.ndarray) and mask_name not in ['confidence_map', 'distance_map']:
            voxel_count = np.sum(mask)
            print(f"  {mask_name}: {voxel_count:,} voxels")
    
    # Check discrimination confidence
    if 'confidence_map' in segmentation_result:
        confidence = segmentation_result['confidence_map']
        confident_voxels = confidence > 0
        if np.any(confident_voxels):
            avg_conf = np.mean(confidence[confident_voxels])
            print(f"\nAverage discrimination confidence: {avg_conf:.2%}")
    
    # Visualize a sample slice
    roi_bounds = metal_result['roi_bounds']
    mid_slice = (roi_bounds['z_min'] + roi_bounds['z_max']) // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original CT
    axes[0, 0].imshow(ct_volume[mid_slice], cmap='gray', vmin=-150, vmax=250)
    axes[0, 0].set_title('Original CT')
    axes[0, 0].axis('off')
    
    # Metal
    axes[0, 1].imshow(metal_mask[mid_slice], cmap='Reds')
    axes[0, 1].set_title('Metal')
    axes[0, 1].axis('off')
    
    # Dark artifacts
    if 'dark_artifacts' in segmentation_result:
        axes[0, 2].imshow(segmentation_result['dark_artifacts'][mid_slice], cmap='Purples')
        axes[0, 2].set_title('Dark Artifacts')
        axes[0, 2].axis('off')
    
    # Bone
    if 'bone' in segmentation_result:
        axes[1, 0].imshow(segmentation_result['bone'][mid_slice], cmap='Blues')
        axes[1, 0].set_title('Bone (Discriminated)')
        axes[1, 0].axis('off')
    
    # Bright artifacts
    if 'bright_artifacts' in segmentation_result:
        axes[1, 1].imshow(segmentation_result['bright_artifacts'][mid_slice], cmap='Oranges')
        axes[1, 1].set_title('Bright Artifacts')
        axes[1, 1].axis('off')
    
    # Confidence map
    if 'confidence_map' in segmentation_result:
        conf_slice = segmentation_result['confidence_map'][mid_slice]
        im = axes[1, 2].imshow(conf_slice, cmap='viridis', vmin=0, vmax=1)
        axes[1, 2].set_title('Discrimination Confidence')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('misc/russian_doll_test.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to misc/russian_doll_test.png")
    plt.close()

if __name__ == "__main__":
    test_russian_doll()