#!/usr/bin/env python3
"""Test the 3D adaptive metal detection"""

import sys
sys.path.append('app')

from pathlib import Path
import numpy as np
from dicom_utils import load_dicom_series_to_hu
from metal_detection_v3 import detect_metal_adaptive_3d

# Test with HIP4 (from the screenshot)
data_dir = Path("data")
patient_path = data_dir / "HIP4 Patient"
ct_dirs = [d for d in patient_path.iterdir() if d.is_dir() and "CT" in d.name]
ct_dir = ct_dirs[0]

print("Loading HIP4 data...")
ct_volume, ct_metadata = load_dicom_series_to_hu(str(ct_dir))

print("\nTesting 3D adaptive detection...")
result = detect_metal_adaptive_3d(
    ct_volume, 
    ct_metadata['spacing'],
    fw_percentage=75,
    margin_cm=2.0,
    intensity_percentile=99.5
)

if result['mask'] is not None:
    metal_voxels = np.sum(result['mask'])
    print(f"Total metal voxels: {metal_voxels:,}")
    
    if result['analysis']:
        thresh = result['analysis']['threshold_used']
        extent = result['analysis']['extent_voxels']
        bounds = result['analysis']['bounds_3d']
        print(f"Auto-detected threshold: {thresh:.0f} HU")
        print(f"3D extent: {extent['z']}×{extent['y']}×{extent['x']} voxels")
        print(f"3D bounds: Z[{bounds['z_min']}-{bounds['z_max']}], Y[{bounds['y_min']}-{bounds['y_max']}], X[{bounds['x_min']}-{bounds['x_max']}]")
    
    # Check individual regions
    if 'individual_regions' in result:
        total_regions = sum(len(regions) for regions in result['individual_regions'].values())
        print(f"Individual regions created: {total_regions} total across {len(result['individual_regions'])} slices")
        
        # Check slice 124 specifically
        if 124 in result['individual_regions']:
            regions_124 = result['individual_regions'][124]
            print(f"Slice 124 individual regions: {len(regions_124)}")
            for i, region in enumerate(regions_124):
                width = region['x_max'] - region['x_min']
                height = region['y_max'] - region['y_min']
                print(f"  Region {i+1}: {width}x{height} at ({region['x_min']}-{region['x_max']}, {region['y_min']}-{region['y_max']})")
    
    # Check slice 124 specifically (from the screenshot)
    if 124 < result['mask'].shape[0]:
        slice_metal = np.sum(result['mask'][124])
        print(f"Metal pixels in slice 124: {slice_metal}")
        
        # Show some threshold info for slice 124
        slice_results = result['slice_thresholds']
        slice_124_result = next((r for r in slice_results if r['slice'] == 124), None)
        if slice_124_result:
            if 'thresholds' in slice_124_result:
                thresholds = slice_124_result['thresholds']
                if isinstance(thresholds, tuple) and len(thresholds) == 2:
                    lower, upper = thresholds
                    print(f"Slice 124 thresholds: {lower:.0f} - {upper:.0f} HU")
                else:
                    print(f"Slice 124 thresholds: {thresholds}")
            method = slice_124_result.get('method', 'unknown')
            print(f"Slice 124 detection method: {method}")
else:
    print("No metal detected")