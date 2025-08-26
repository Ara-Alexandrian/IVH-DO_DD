#!/usr/bin/env python3
"""
Test script for CT Metal Artifact Characterization app
Tests core functionality without UI
"""

import sys
import os
sys.path.append('app')

from pathlib import Path
import numpy as np
from dicom_utils import load_dicom_series_to_hu, create_metal_mask_from_rtstruct
from metal_detection import detect_metal_volume, create_affine_from_dicom_meta
from contour_operations import create_bright_artifact_mask, create_dark_artifact_mask, create_bone_mask
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def test_patient(patient_name):
    """Test loading and processing a single patient"""
    print(f"\n{'='*60}")
    print(f"Testing {patient_name}")
    print('='*60)
    
    try:
        # Find patient directory
        data_dir = Path("data")
        patient_path = data_dir / patient_name
        
        if not patient_path.exists():
            print(f"❌ Patient directory not found: {patient_path}")
            return False
            
        # Find CT directory
        ct_dirs = [d for d in patient_path.iterdir() if d.is_dir() and "CT" in d.name]
        if not ct_dirs:
            print(f"❌ No CT directory found for {patient_name}")
            return False
            
        ct_dir = ct_dirs[0]
        print(f"✓ Found CT directory: {ct_dir.name}")
        
        # Load DICOM data
        print("Loading DICOM series...")
        ct_volume, ct_metadata = load_dicom_series_to_hu(str(ct_dir))
        
        if ct_volume is None:
            print("❌ Failed to load CT volume")
            return False
            
        print(f"✓ Loaded CT volume: shape={ct_volume.shape}, spacing={ct_metadata['spacing']}")
        print(f"  HU range: [{np.min(ct_volume):.0f}, {np.max(ct_volume):.0f}]")
        
        # Test metal detection
        print("\nTesting automatic metal detection...")
        result = detect_metal_volume(ct_volume, ct_metadata['spacing'], margin_cm=3.0)
        
        if result['mask'] is None or not np.any(result['mask']):
            print("❌ No metal detected")
            return False
            
        metal_voxels = np.sum(result['mask'])
        print(f"✓ Metal detected: {metal_voxels:,} voxels")
        print(f"  ROI bounds: Z[{result['roi_bounds']['z_min']}-{result['roi_bounds']['z_max']}]")
        print(f"  Center: {result['center_coords']}")
        
        # Test artifact segmentation
        print("\nTesting artifact segmentation...")
        bright_mask = create_bright_artifact_mask(
            ct_volume, result['mask'], result['roi_bounds'], 800, 3000
        )
        dark_mask = create_dark_artifact_mask(
            ct_volume, result['mask'], result['roi_bounds'], -200
        )
        bone_mask = create_bone_mask(
            ct_volume, result['mask'], bright_mask, dark_mask, 
            result['roi_bounds'], 150, 1500
        )
        
        print(f"✓ Bright artifacts: {np.sum(bright_mask):,} voxels")
        print(f"✓ Dark artifacts: {np.sum(dark_mask):,} voxels")
        print(f"✓ Bone tissue: {np.sum(bone_mask):,} voxels")
        
        # Test threshold evolution
        print("\nThreshold statistics:")
        thresholds = [t['thresholds'] for t in result['slice_thresholds'] if t['thresholds']]
        if thresholds:
            lower_vals = [t[0] for t in thresholds]
            upper_vals = [t[1] for t in thresholds]
            print(f"  Lower threshold range: [{min(lower_vals):.0f}, {max(lower_vals):.0f}]")
            print(f"  Upper threshold range: [{min(upper_vals):.0f}, {max(upper_vals):.0f}]")
        
        print(f"\n✅ {patient_name} - All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing {patient_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all patients"""
    print("CT Metal Artifact Characterization - Test Suite")
    print("=" * 60)
    
    # Test each patient
    patients = ["HIP1 Patient", "HIP2 Patient", "HIP3 Patient", "HIP4 Patient"]
    results = {}
    
    for patient in patients:
        results[patient] = test_patient(patient)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for patient, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{patient}: {status}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The app is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()