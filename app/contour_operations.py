import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes, generate_binary_structure, distance_transform_edt, gaussian_filter, median_filter
from core.discrimination import ArtifactDiscriminator, DiscriminationMethod
from body_mask import create_body_mask, constrain_to_body


def calculate_metal_distance_map(metal_mask, spacing):
    """
    Calculate distance map from metal implant.
    
    Args:
        metal_mask: Binary mask of metal regions
        spacing: Voxel spacing (z, y, x) in mm
        
    Returns:
        Distance map in mm from nearest metal voxel
    """
    # Invert metal mask for distance transform (distance to nearest True voxel)
    distance_map_voxels = distance_transform_edt(~metal_mask)
    
    # Convert voxel distances to physical distances (mm)
    # Use anisotropic spacing: z, y, x
    distance_map_mm = distance_map_voxels * np.mean(np.abs(spacing))
    
    return distance_map_mm


def analyze_local_neighborhood(ct_volume, center_coords, window_size=(7, 7, 3), 
                             bone_range=(500, 1500), tissue_range=(-100, 300)):
    """
    Enhanced neighborhood analysis including spatial and gradient features.
    
    Args:
        ct_volume: CT volume in HU
        center_coords: (z, y, x) coordinates to analyze
        window_size: Size of analysis window (z, y, x)
        bone_range: HU range for bone tissue
        tissue_range: HU range for soft tissue
        
    Returns:
        Dict with comprehensive neighborhood analysis results
    """
    z, y, x = center_coords
    wz, wy, wx = window_size
    
    # Define window bounds
    z_min = max(0, z - wz//2)
    z_max = min(ct_volume.shape[0], z + wz//2 + 1)
    y_min = max(0, y - wy//2)
    y_max = min(ct_volume.shape[1], y + wy//2 + 1)
    x_min = max(0, x - wx//2)
    x_max = min(ct_volume.shape[2], x + wx//2 + 1)
    
    # Extract neighborhood window
    neighborhood = ct_volume[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Count tissue types in neighborhood
    bone_mask = (neighborhood >= bone_range[0]) & (neighborhood <= bone_range[1])
    tissue_mask = (neighborhood >= tissue_range[0]) & (neighborhood <= tissue_range[1])
    
    total_voxels = neighborhood.size
    bone_count = np.sum(bone_mask)
    tissue_count = np.sum(tissue_mask)
    
    # Calculate ratios
    bone_ratio = bone_count / total_voxels if total_voxels > 0 else 0
    tissue_ratio = tissue_count / total_voxels if total_voxels > 0 else 0
    
    # Basic intensity statistics
    mean_hu = np.mean(neighborhood)
    std_hu = np.std(neighborhood)
    
    # Enhanced features for better discrimination
    # 1. Gradient analysis - artifacts have sharper boundaries
    if neighborhood.shape[1] > 2 and neighborhood.shape[2] > 2:
        # Calculate gradients in the central slice
        central_slice = neighborhood[wz//2] if neighborhood.shape[0] > 1 else neighborhood[0]
        grad_y = np.gradient(central_slice, axis=0)
        grad_x = np.gradient(central_slice, axis=1)
        gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        avg_gradient = np.mean(gradient_magnitude)
        max_gradient = np.max(gradient_magnitude)
    else:
        avg_gradient = 0
        max_gradient = 0
    
    # 2. Intensity homogeneity - bone tends to be more homogeneous
    intensity_range = np.max(neighborhood) - np.min(neighborhood)
    coefficient_variation = std_hu / (mean_hu + 1e-6)  # Avoid division by zero
    
    # 3. High HU concentration - artifacts tend to have higher HU values
    high_hu_threshold = 1200  # Above typical bone
    high_hu_ratio = np.sum(neighborhood > high_hu_threshold) / total_voxels
    
    return {
        'bone_ratio': bone_ratio,
        'tissue_ratio': tissue_ratio,
        'bone_count': bone_count,
        'tissue_count': tissue_count,
        'total_voxels': total_voxels,
        'mean_hu': mean_hu,
        'std_hu': std_hu,
        'avg_gradient': avg_gradient,
        'max_gradient': max_gradient,
        'intensity_range': intensity_range,
        'coefficient_variation': coefficient_variation,
        'high_hu_ratio': high_hu_ratio,
        'dominant_tissue': 'bone' if bone_ratio > tissue_ratio else 'tissue'
    }


def classify_bright_artifacts_contextually(ct_volume, bright_mask, metal_mask, spacing,
                                         bone_range=(500, 1500), tissue_range=(-100, 300),
                                         bone_distance_threshold_mm=20.0, bone_ratio_threshold=0.4,
                                         window_size=(7, 7, 3)):
    """
    Classify bright artifacts based on distance from metal and local tissue context.
    
    Args:
        ct_volume: CT volume in HU
        bright_mask: Binary mask of bright artifacts
        metal_mask: Binary mask of metal regions
        spacing: Voxel spacing (z, y, x) in mm
        bone_range: HU range for bone tissue
        tissue_range: HU range for soft tissue
        bone_distance_threshold_mm: Distance threshold for bone-dominant region (mm)
        bone_ratio_threshold: Minimum ratio of bone voxels to classify as bone context
        window_size: Size of neighborhood analysis window
        
    Returns:
        Dict with contextual bright artifact masks
    """
    print(f"Classifying {np.sum(bright_mask):,} bright artifact voxels contextually...")
    
    # Calculate distance map from metal
    distance_map = calculate_metal_distance_map(metal_mask, spacing)
    
    # Initialize contextual masks
    bright_artifact_bone = np.zeros_like(bright_mask, dtype=bool)
    bright_artifact_tissue = np.zeros_like(bright_mask, dtype=bool)
    bright_artifact_mixed = np.zeros_like(bright_mask, dtype=bool)
    
    # Get coordinates of bright artifact voxels
    bright_coords = np.where(bright_mask)
    n_voxels = len(bright_coords[0])
    
    if n_voxels == 0:
        return {
            'bright_artifact_bone': bright_artifact_bone,
            'bright_artifact_tissue': bright_artifact_tissue,
            'bright_artifact_mixed': bright_artifact_mixed
        }
    
    # Process in batches for performance
    batch_size = min(10000, n_voxels)
    n_bone = n_tissue = n_mixed = 0
    
    for i in range(0, n_voxels, batch_size):
        end_idx = min(i + batch_size, n_voxels)
        batch_coords = (bright_coords[0][i:end_idx], 
                       bright_coords[1][i:end_idx], 
                       bright_coords[2][i:end_idx])
        
        for j, (z, y, x) in enumerate(zip(*batch_coords)):
            # Get distance from metal
            distance_mm = distance_map[z, y, x]
            
            # Analyze local neighborhood
            neighborhood = analyze_local_neighborhood(
                ct_volume, (z, y, x), window_size, bone_range, tissue_range
            )
            
            # Enhanced contextual classification logic
            
            # Feature-based scoring for bone vs artifact
            bone_score = 0
            artifact_score = 0
            
            # 1. Distance factor - closer to metal slightly favors artifacts
            if distance_mm < 8:
                artifact_score += 1  # Very close = likely artifact
            elif distance_mm > 25:
                bone_score += 1      # Far = more likely bone
            
            # 2. Neighborhood composition - strong indicator
            if neighborhood['bone_ratio'] > 0.6:
                bone_score += 3      # Strongly surrounded by bone-HU voxels
            elif neighborhood['bone_ratio'] > 0.4:
                bone_score += 1      # Moderately surrounded by bone
            if neighborhood['tissue_ratio'] > 0.4:
                bone_score += 1      # Some tissue context = more bone-like
                
            # 3. Intensity characteristics - more conservative thresholds
            if neighborhood['mean_hu'] > 1600:
                artifact_score += 2  # Very high HU = likely artifact
            elif neighborhood['mean_hu'] > 1200:
                artifact_score += 1  # High HU = somewhat artifact
            elif neighborhood['mean_hu'] < 900:
                bone_score += 2      # Lower HU = more bone-like
                
            # 4. Gradient/sharpness - artifacts have sharper boundaries
            if neighborhood['avg_gradient'] > 150:
                artifact_score += 2  # High gradient = sharp boundaries = artifact
            elif neighborhood['avg_gradient'] > 80:
                artifact_score += 1  # Moderate gradient = somewhat artifact
            elif neighborhood['avg_gradient'] < 40:
                bone_score += 1      # Low gradient = smooth = bone
                
            # 5. Intensity homogeneity - bone is more homogeneous  
            if neighborhood['coefficient_variation'] > 0.4:
                artifact_score += 1  # High variation = artifact-like
            elif neighborhood['coefficient_variation'] < 0.2:
                bone_score += 2      # Low variation = bone-like
                
            # 6. High HU concentration - artifacts have more extreme values
            if neighborhood['high_hu_ratio'] > 0.4:
                artifact_score += 2  # Many high HU voxels = artifact
            elif neighborhood['high_hu_ratio'] > 0.2:
                artifact_score += 1  # Some high HU voxels = somewhat artifact
            elif neighborhood['high_hu_ratio'] < 0.05:
                bone_score += 1      # Few high HU voxels = bone
            
            # Make classification decision based on scores (FIXED: was backwards)
            if bone_score > artifact_score + 1:  # Bone tissue being corrupted
                bright_artifact_tissue[z, y, x] = True  # Bright artifact corrupting tissue (should restore to tissue HU)
                n_tissue += 1
            elif artifact_score > bone_score + 1:  # Soft tissue being corrupted
                bright_artifact_bone[z, y, x] = True  # Bright artifact corrupting bone (should restore to bone HU)
                n_bone += 1
            else:  # Uncertain/close scores
                bright_artifact_mixed[z, y, x] = True  # Uncertain what tissue is being corrupted
                n_mixed += 1
    
    print(f"Contextual classification results:")
    print(f"  Bright artifacts over bone: {n_bone:,} voxels ({100*n_bone/n_voxels:.1f}%)")
    print(f"  Bright artifacts over tissue: {n_tissue:,} voxels ({100*n_tissue/n_voxels:.1f}%)")
    print(f"  Mixed/uncertain artifacts: {n_mixed:,} voxels ({100*n_mixed/n_voxels:.1f}%)")
    
    return {
        'bright_artifact_bone': bright_artifact_bone,
        'bright_artifact_tissue': bright_artifact_tissue,
        'bright_artifact_mixed': bright_artifact_mixed
    }


def estimate_expected_tissue_values(ct_volume, metal_mask, spacing, window_size=(15, 15, 5)):
    """
    Estimate what tissue HU values should be in each region, accounting for metal artifacts.
    
    This creates a "baseline" tissue map by removing metal influence and estimating
    the underlying tissue structure that's being corrupted by artifacts.
    
    Args:
        ct_volume: CT volume in HU
        metal_mask: Binary mask of metal regions  
        spacing: Voxel spacing (z, y, x) in mm
        window_size: Size of smoothing window
        
    Returns:
        expected_tissue_map: 3D array of expected HU values for each voxel
    """
    print(f"Estimating expected tissue values across {ct_volume.shape} volume...")
    
    # 1. Create metal-excluded volume for analysis
    metal_excluded_volume = ct_volume.copy().astype(np.float32)
    metal_excluded_volume[metal_mask] = np.nan  # Exclude metal from analysis
    
    # 2. Create distance-based weighting from metal (artifacts stronger closer to metal)
    metal_distance_map = calculate_metal_distance_map(metal_mask, spacing)
    
    # 3. Apply median filter to reduce extreme values while preserving edges
    # Use nanmedian to handle metal exclusions
    from scipy.ndimage import generic_filter
    
    def nanmedian_filter(input_array, size):
        """Median filter that ignores NaN values"""
        def nanmedian(x):
            x_clean = x[~np.isnan(x)]
            return np.median(x_clean) if len(x_clean) > 0 else np.nan
        return generic_filter(input_array, nanmedian, size=size)
    
    # Apply median filtering to get cleaner tissue estimates
    wz, wy, wx = window_size
    filtered_volume = nanmedian_filter(metal_excluded_volume, size=(wz, wy, wx))
    
    # 4. For areas near metal, use more aggressive smoothing
    # Apply Gaussian smoothing with distance-weighted sigma
    expected_tissue_map = filtered_volume.copy()
    
    # 5. Fill in metal regions with interpolated values
    # Use distance-weighted interpolation from nearby non-metal tissue
    for z in range(ct_volume.shape[0]):
        slice_mask = metal_mask[z]
        if np.any(slice_mask):
            slice_data = expected_tissue_map[z]
            
            # For metal regions, interpolate from nearby tissue
            metal_coords = np.where(slice_mask)
            for i, (y, x) in enumerate(zip(metal_coords[0], metal_coords[1])):
                # Find nearby non-metal tissue values
                y_min = max(0, y - wy//2)
                y_max = min(ct_volume.shape[1], y + wy//2)
                x_min = max(0, x - wx//2)
                x_max = min(ct_volume.shape[2], x + wx//2)
                
                neighborhood = slice_data[y_min:y_max, x_min:x_max]
                valid_values = neighborhood[~np.isnan(neighborhood)]
                
                if len(valid_values) > 0:
                    expected_tissue_map[z, y, x] = np.median(valid_values)
                else:
                    # Fallback: use overall slice statistics
                    valid_slice = slice_data[~np.isnan(slice_data)]
                    if len(valid_slice) > 0:
                        expected_tissue_map[z, y, x] = np.median(valid_slice)
                    else:
                        expected_tissue_map[z, y, x] = 0  # Default fallback
    
    # 6. Apply gentle smoothing to final result
    expected_tissue_map = gaussian_filter(expected_tissue_map, sigma=1.0)
    
    print(f"Expected tissue range: {np.nanmin(expected_tissue_map):.1f} to {np.nanmax(expected_tissue_map):.1f} HU")
    
    return expected_tissue_map


def detect_context_aware_bright_artifacts(ct_volume, metal_mask, spacing, 
                                         elevation_threshold=1.5, min_elevation_hu=100,
                                         max_elevation_hu=2000):
    """
    Detect bright artifacts based on tissue context rather than absolute HU values.
    
    This identifies voxels that are significantly elevated above what the local
    tissue should naturally be, regardless of absolute HU value.
    
    Args:
        ct_volume: CT volume in HU
        metal_mask: Binary mask of metal regions
        spacing: Voxel spacing (z, y, x) in mm
        elevation_threshold: Multiplier for elevation (1.5 = 50% above expected)
        min_elevation_hu: Minimum HU elevation to consider artifact
        max_elevation_hu: Maximum HU elevation to consider artifact (above this = extreme artifacts)
        
    Returns:
        Dict with context-aware artifact masks
    """
    print(f"Running context-aware bright artifact detection...")
    
    # 1. Estimate expected tissue values
    expected_tissue_map = estimate_expected_tissue_values(ct_volume, metal_mask, spacing)
    
    # 2. Calculate elevation above expected
    elevation_map = ct_volume - expected_tissue_map
    elevation_ratio = ct_volume / (expected_tissue_map + 1e-6)  # Avoid division by zero
    
    # 3. Identify bright artifacts using multiple criteria
    # Criterion 1: Relative elevation (50% above expected)
    relative_artifacts = (elevation_ratio > elevation_threshold) & (elevation_map > min_elevation_hu)
    
    # Criterion 2: Absolute elevation (significantly above expected, regardless of ratio)
    absolute_artifacts = elevation_map > min_elevation_hu
    
    # Criterion 3: Extreme artifacts (way above expected)
    extreme_artifacts = elevation_map > max_elevation_hu
    
    # 4. Combine criteria with body constraint
    body_mask = create_body_mask(ct_volume, air_threshold=-300)
    
    # Create unified bright artifact mask
    bright_artifacts_unified = (relative_artifacts | absolute_artifacts) & body_mask & (~metal_mask)
    
    # 5. Classify by elevation level
    mild_artifacts = (elevation_map > min_elevation_hu) & (elevation_map <= min_elevation_hu * 2) & bright_artifacts_unified
    moderate_artifacts = (elevation_map > min_elevation_hu * 2) & (elevation_map <= max_elevation_hu) & bright_artifacts_unified  
    severe_artifacts = (elevation_map > max_elevation_hu) & bright_artifacts_unified
    
    # 6. Statistics
    total_bright = np.sum(bright_artifacts_unified)
    if total_bright > 0:
        mild_count = np.sum(mild_artifacts)
        moderate_count = np.sum(moderate_artifacts)
        severe_count = np.sum(severe_artifacts)
        
        print(f"Context-aware bright artifact detection results:")
        print(f"  Total bright artifacts: {total_bright:,} voxels")
        print(f"  Mild (+{min_elevation_hu}-{min_elevation_hu*2} HU): {mild_count:,} ({100*mild_count/total_bright:.1f}%)")
        print(f"  Moderate (+{min_elevation_hu*2}-{max_elevation_hu} HU): {moderate_count:,} ({100*moderate_count/total_bright:.1f}%)")
        print(f"  Severe (+{max_elevation_hu}+ HU): {severe_count:,} ({100*severe_count/total_bright:.1f}%)")
    
    return {
        'bright_artifacts_unified': bright_artifacts_unified,
        'bright_artifacts_mild': mild_artifacts,
        'bright_artifacts_moderate': moderate_artifacts, 
        'bright_artifacts_severe': severe_artifacts,
        'expected_tissue_map': expected_tissue_map,
        'elevation_map': elevation_map,
        'elevation_ratio': elevation_ratio
    }


def create_context_aware_masks(ct_volume, metal_mask, spacing, **kwargs):
    """
    Create tissue masks using context-aware bright artifact detection.
    
    This approach detects bright artifacts based on tissue context rather than 
    absolute HU values, catching artifacts across the full spectrum.
    
    Args:
        ct_volume: CT volume in HU
        metal_mask: Binary mask of metal regions
        spacing: Voxel spacing (z, y, x)
        **kwargs: Additional parameters
        
    Returns:
        Dict with all mask types including context-aware bright artifacts
    """
    roi_bounds = kwargs.get('roi_bounds', None)
    debug = kwargs.get('debug', False)
    
    # Create body mask to exclude air outside patient
    body_mask = create_body_mask(ct_volume, air_threshold=-300)
    if debug:
        print(f"Debug - Body mask voxels: {np.sum(body_mask)}")

    # Create ROI mask if bounds provided
    roi_mask = None
    if roi_bounds is not None:
        if isinstance(roi_bounds, dict):
            z_min = int(roi_bounds['z_min'])
            z_max = int(roi_bounds['z_max'])
            y_min = int(roi_bounds['y_min'])
            y_max = int(roi_bounds['y_max'])
            x_min = int(roi_bounds['x_min'])
            x_max = int(roi_bounds['x_max'])
        else:
            z_min, z_max, y_min, y_max, x_min, x_max = roi_bounds
            z_min, z_max, y_min, y_max, x_min, x_max = int(z_min), int(z_max), int(y_min), int(y_max), int(x_min), int(x_max)
        roi_mask = np.zeros_like(ct_volume, dtype=bool)
        roi_mask[z_min:z_max, y_min:y_max, x_min:x_max] = True

    # 1. Create dark artifacts mask (unchanged)
    dark_range = kwargs.get('dark_range', [-1024, -150])
    dark_mask = (ct_volume >= dark_range[0]) & (ct_volume <= dark_range[1]) & (~metal_mask) & body_mask
    
    # Apply ROI constraint to dark mask
    if roi_mask is not None:
        dark_mask = dark_mask & roi_mask
        
    # Further constrain dark artifacts to areas near metal to avoid bowel gas
    metal_vicinity = binary_dilation(metal_mask, iterations=20)
    dark_mask = dark_mask & metal_vicinity

    # 2. Run context-aware bright artifact detection
    context_results = detect_context_aware_bright_artifacts(
        ct_volume, metal_mask, spacing,
        elevation_threshold=kwargs.get('elevation_threshold', 1.5),
        min_elevation_hu=kwargs.get('min_elevation_hu', 100),
        max_elevation_hu=kwargs.get('max_elevation_hu', 2000)
    )
    
    # Apply ROI constraints to context-aware results
    if roi_mask is not None:
        for key in ['bright_artifacts_unified', 'bright_artifacts_mild', 'bright_artifacts_moderate', 'bright_artifacts_severe']:
            if key in context_results:
                context_results[key] = context_results[key] & roi_mask

    # 3. Create bone mask from remaining tissue
    # Use traditional bone range, excluding metal and artifacts
    bone_range = kwargs.get('bone_range', [500, 1500])
    bone_mask = (ct_volume >= bone_range[0]) & (ct_volume <= bone_range[1]) & (~metal_mask) & body_mask
    bone_mask = bone_mask & (~dark_mask) & (~context_results['bright_artifacts_unified'])
    
    # Apply ROI constraint to bone mask
    if roi_mask is not None:
        bone_mask = bone_mask & roi_mask

    # 4. Apply contextual classification to bright artifacts (similar to original)
    contextual_artifacts = classify_bright_artifacts_contextually(
        ct_volume, 
        context_results['bright_artifacts_unified'],
        metal_mask, 
        spacing,
        bone_range=bone_range,
        tissue_range=(-100, 300),
        bone_distance_threshold_mm=20.0,
        bone_ratio_threshold=0.4
    )

    if debug:
        print(f"Debug - Context-aware results:")
        print(f"  Dark artifacts: {np.sum(dark_mask)}")
        print(f"  Bone: {np.sum(bone_mask)}")
        print(f"  Bright artifacts (unified): {np.sum(context_results['bright_artifacts_unified'])}")
        print(f"  Bright artifacts (mild): {np.sum(context_results['bright_artifacts_mild'])}")
        print(f"  Bright artifacts (moderate): {np.sum(context_results['bright_artifacts_moderate'])}")
        print(f"  Bright artifacts (severe): {np.sum(context_results['bright_artifacts_severe'])}")

    return {
        'metal': metal_mask,
        'dark_artifacts': dark_mask,
        'bone': bone_mask,
        'bright_artifacts': context_results['bright_artifacts_unified'],  # Legacy compatibility
        'bright_artifacts_mild': context_results['bright_artifacts_mild'],
        'bright_artifacts_moderate': context_results['bright_artifacts_moderate'], 
        'bright_artifacts_severe': context_results['bright_artifacts_severe'],
        'bright_artifact_bone': contextual_artifacts['bright_artifact_bone'],
        'bright_artifact_tissue': contextual_artifacts['bright_artifact_tissue'],
        'bright_artifact_mixed': contextual_artifacts['bright_artifact_mixed'],
        # Additional diagnostic data
        'expected_tissue_map': context_results['expected_tissue_map'],
        'elevation_map': context_results['elevation_map']
    }


# Wrapper functions for backward compatibility
def create_sequential_masks(ct_volume, metal_mask, spacing, discrimination_method='distance', **kwargs):
    """Sequential mask creation with configurable discrimination method.
    
    Args:
        ct_volume: CT volume in Hounsfield Units
        metal_mask: Binary mask of metal regions
        spacing: Voxel spacing (z, y, x)
        discrimination_method: 'distance', 'edge', 'profile', or 'context_aware'
        **kwargs: Additional parameters including threshold ranges and ROI bounds
    """
    # Check if using context-aware bright artifact detection
    use_context_aware = kwargs.get('use_context_aware', False) or discrimination_method == 'context_aware'
    
    if use_context_aware:
        return create_context_aware_masks(ct_volume, metal_mask, spacing, **kwargs)
    
    # Map method names to discrimination methods
    method_map = {
        'distance': DiscriminationMethod.DISTANCE_BASED,
        'fast': DiscriminationMethod.DISTANCE_BASED,
        'edge': DiscriminationMethod.EDGE_BASED,
        'enhanced': DiscriminationMethod.EDGE_BASED,
        'profile': DiscriminationMethod.PROFILE_BASED
    }
    
    disc_method = method_map.get(discrimination_method, DiscriminationMethod.DISTANCE_BASED)
    discriminator = ArtifactDiscriminator(disc_method)
    
    # Get threshold ranges
    bright_range = kwargs.get('bright_range', [800, 3500])
    bone_range = kwargs.get('bone_range', [500, 1500])
    dark_range = kwargs.get('dark_range', [-1024, -150])
    roi_bounds = kwargs.get('roi_bounds', None)
    debug = kwargs.get('debug', False)
    
    # Create body mask to exclude air outside patient
    body_mask = create_body_mask(ct_volume, air_threshold=-300)
    if debug:
        print(f"Debug - Body mask voxels: {np.sum(body_mask)}")
    
    # Create ROI mask if bounds provided
    roi_mask = None
    if roi_bounds is not None:
        if isinstance(roi_bounds, dict):
            z_min = int(roi_bounds['z_min'])
            z_max = int(roi_bounds['z_max'])
            y_min = int(roi_bounds['y_min'])
            y_max = int(roi_bounds['y_max'])
            x_min = int(roi_bounds['x_min'])
            x_max = int(roi_bounds['x_max'])
        else:
            z_min, z_max, y_min, y_max, x_min, x_max = roi_bounds
            z_min, z_max, y_min, y_max, x_min, x_max = int(z_min), int(z_max), int(y_min), int(y_max), int(x_min), int(x_max)
        roi_mask = np.zeros_like(ct_volume, dtype=bool)
        roi_mask[z_min:z_max, y_min:y_max, x_min:x_max] = True
    
    # Create combined bright mask (includes both bone and bright artifact ranges for discrimination)
    bright_mask = ((ct_volume >= bright_range[0]) & (ct_volume <= bright_range[1])) | \
                  ((ct_volume >= bone_range[0]) & (ct_volume <= bone_range[1]))
    bright_mask = bright_mask & (~metal_mask) & body_mask  # Exclude metal and constrain to body
    
    # Apply ROI constraint if provided
    if roi_mask is not None:
        bright_mask = bright_mask & roi_mask
    
    if debug:
        print(f"Debug - Bright mask voxels before discrimination: {np.sum(bright_mask)}")
        print(f"Debug - Bright range: {bright_range}, Bone range: {bone_range}")
    
    # Discriminate bone from artifacts
    result = discriminator.discriminate(ct_volume, metal_mask, bright_mask, spacing)
    
    if debug:
        print(f"Debug - After discrimination - Bone: {np.sum(result['bone_mask'])}, Artifacts: {np.sum(result['artifact_mask'])}")
    
    # Perform contextual classification of bright artifacts
    contextual_artifacts = classify_bright_artifacts_contextually(
        ct_volume, 
        result['artifact_mask'],  # Only classify the artifact voxels (not bone)
        metal_mask, 
        spacing,
        bone_range=bone_range,
        tissue_range=(-100, 300),  # Soft tissue range
        bone_distance_threshold_mm=25.0,
        bone_ratio_threshold=0.6
    )
    
    # Create dark mask - constrained to body to avoid air outside patient
    dark_mask = (ct_volume >= dark_range[0]) & (ct_volume <= dark_range[1]) & (~metal_mask) & body_mask
    
    # Apply ROI constraint to dark mask as well
    if roi_mask is not None:
        dark_mask = dark_mask & roi_mask
    
    # Further constrain dark artifacts to areas near metal to avoid bowel gas
    from scipy.ndimage import binary_dilation
    metal_vicinity = binary_dilation(metal_mask, iterations=20)
    dark_mask = dark_mask & metal_vicinity
    
    if debug:
        print(f"Debug - Dark mask voxels: {np.sum(dark_mask)}, Dark range: {dark_range}")
    
    return {
        'metal': metal_mask,
        'dark_artifacts': dark_mask,
        'bone': result['bone_mask'],
        'bright_artifacts': result['artifact_mask'],  # Keep original for backward compatibility
        'bright_artifact_bone': contextual_artifacts['bright_artifact_bone'],
        'bright_artifact_tissue': contextual_artifacts['bright_artifact_tissue'],
        'bright_artifact_mixed': contextual_artifacts['bright_artifact_mixed']
    }


# Legacy wrapper functions for backward compatibility
def create_fast_russian_doll_segmentation(ct_volume, metal_mask, spacing, **kwargs):
    """Fast Russian doll segmentation using distance-based discrimination."""
    return create_sequential_masks(ct_volume, metal_mask, spacing, discrimination_method='fast', **kwargs)


def create_enhanced_russian_doll_segmentation(ct_volume, metal_mask, spacing, **kwargs):
    """Enhanced Russian doll segmentation using edge-based discrimination."""
    return create_sequential_masks(ct_volume, metal_mask, spacing, discrimination_method='enhanced', **kwargs)


def refine_bone_artifact_discrimination(bone_mask, artifact_mask, ct_volume, spacing, **kwargs):
    """Refine bone/artifact discrimination using morphological operations."""
    # Apply morphological operations
    struct = generate_binary_structure(3, 1)
    bone_refined = binary_fill_holes(bone_mask)
    bone_refined = binary_erosion(bone_refined, struct, iterations=1)
    bone_refined = binary_dilation(bone_refined, struct, iterations=1)
    
    artifact_refined = binary_fill_holes(artifact_mask)
    artifact_refined = binary_erosion(artifact_refined, struct, iterations=1)
    artifact_refined = binary_dilation(artifact_refined, struct, iterations=1)
    
    return {
        'bone_mask': bone_refined,
        'artifact_mask': artifact_refined,
        'success': True
    }


def boolean_subtract(mask1, mask2):
    """
    Subtract mask2 from mask1 (mask1 AND NOT mask2).
    
    Args:
        mask1: First binary mask (numpy array)
        mask2: Second binary mask (numpy array)
        
    Returns:
        Binary mask with mask2 subtracted from mask1
    """
    return mask1 & ~mask2


def boolean_union(mask1, mask2):
    """
    Union of two masks (mask1 OR mask2).
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        Binary mask representing the union
    """
    return mask1 | mask2


def boolean_intersection(mask1, mask2):
    """
    Intersection of two masks (mask1 AND mask2).
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        Binary mask representing the intersection
    """
    return mask1 & mask2


def create_bright_artifact_mask(ct_volume, metal_mask, roi_bounds, 
                               bright_threshold_low=800, bright_threshold_high=3000):
    """
    Create bright artifact mask by thresholding and subtracting metal.
    
    Args:
        ct_volume: 3D CT data in HU
        metal_mask: 3D binary mask of metal
        roi_bounds: Dictionary with ROI boundaries
        bright_threshold_low: Lower HU threshold for bright artifacts
        bright_threshold_high: Upper HU threshold for bright artifacts
        
    Returns:
        3D binary mask of bright artifacts (excluding metal)
    """
    # Create initial bright region mask
    bright_mask = (ct_volume >= bright_threshold_low) & (ct_volume <= bright_threshold_high)
    
    # Constrain to ROI
    roi_mask = np.zeros_like(ct_volume, dtype=bool)
    roi_mask[roi_bounds['z_min']:roi_bounds['z_max'],
             roi_bounds['y_min']:roi_bounds['y_max'],
             roi_bounds['x_min']:roi_bounds['x_max']] = True
    
    bright_mask = bright_mask & roi_mask
    
    # Subtract metal to get isolated bright artifacts
    bright_artifacts_mask = boolean_subtract(bright_mask, metal_mask)
    
    return bright_artifacts_mask


def create_dark_artifact_mask(ct_volume, metal_mask, roi_bounds,
                             dark_threshold_high=-200):
    """
    Create dark artifact mask for low HU streaking artifacts.
    
    Args:
        ct_volume: 3D CT data in HU
        metal_mask: 3D binary mask of metal
        roi_bounds: Dictionary with ROI boundaries
        dark_threshold_high: Upper HU threshold for dark artifacts
        
    Returns:
        3D binary mask of dark artifacts
    """
    # Create dark region mask
    dark_mask = ct_volume <= dark_threshold_high
    
    # Constrain to ROI
    roi_mask = np.zeros_like(ct_volume, dtype=bool)
    roi_mask[roi_bounds['z_min']:roi_bounds['z_max'],
             roi_bounds['y_min']:roi_bounds['y_max'],
             roi_bounds['x_min']:roi_bounds['x_max']] = True
    
    dark_mask = dark_mask & roi_mask
    
    # Dark artifacts shouldn't overlap with metal
    dark_artifacts_mask = boolean_subtract(dark_mask, metal_mask)
    
    return dark_artifacts_mask


def refine_mask(mask, min_size=10, fill_holes=True, smooth_iterations=1):
    """
    Refine a binary mask by removing small components and optionally filling holes.
    
    Args:
        mask: Binary mask to refine
        min_size: Minimum component size in voxels
        fill_holes: Whether to fill holes in the mask
        smooth_iterations: Number of morphological smoothing iterations
        
    Returns:
        Refined binary mask
    """
    from scipy.ndimage import label
    
    refined_mask = mask.copy()
    
    # Determine if this is a 3D mask
    is_3d = mask.ndim == 3
    
    # Fill holes if requested
    if fill_holes and np.any(refined_mask):
        try:
            refined_mask = binary_fill_holes(refined_mask)
        except (MemoryError, ValueError) as e:
            print(f"  Warning: hole filling failed for shape {mask.shape}: {str(e)}")
    
    # Use appropriate connectivity for labeling
    if is_3d:
        # 3D connectivity structure (26-connected)
        struct = generate_binary_structure(3, 3)
    else:
        # 2D connectivity structure (8-connected)
        struct = generate_binary_structure(2, 2)
    
    # Remove small components
    labeled_array, num_features = label(refined_mask, structure=struct)
    print(f"  Found {num_features} connected components (min_size={min_size})")
    
    if num_features > 0:
        # Count component sizes
        component_sizes = []
        for i in range(1, num_features + 1):
            component_mask = labeled_array == i
            size = np.sum(component_mask)
            component_sizes.append(size)
            if size < min_size:
                refined_mask[component_mask] = False
        
        if component_sizes:
            print(f"  Component sizes: min={min(component_sizes)}, max={max(component_sizes)}, mean={np.mean(component_sizes):.1f}")
    
    # Smooth with morphological operations
    if smooth_iterations > 0 and np.any(refined_mask):
        # Use 3D structuring element for 3D data
        if is_3d:
            struct_erode = generate_binary_structure(3, 1)  # 6-connected for erosion
            struct_dilate = generate_binary_structure(3, 2)  # 18-connected for dilation
        else:
            struct_erode = struct_dilate = None
            
        for _ in range(smooth_iterations):
            refined_mask = binary_dilation(refined_mask, iterations=1, structure=struct_dilate)
            refined_mask = binary_erosion(refined_mask, iterations=1, structure=struct_erode)
    
    return refined_mask


def create_bone_mask(ct_volume, metal_mask, bright_mask, dark_mask, roi_bounds,
                    bone_threshold_low=400, bone_threshold_high=1500):
    """
    Create bone mask by thresholding and subtracting other masks.
    
    Args:
        ct_volume: 3D CT data in HU
        metal_mask: 3D binary mask of metal
        bright_mask: 3D binary mask of bright artifacts
        dark_mask: 3D binary mask of dark artifacts
        roi_bounds: Dictionary with ROI boundaries
        bone_threshold_low: Lower HU threshold for bone
        bone_threshold_high: Upper HU threshold for bone
        
    Returns:
        3D binary mask of bone tissue
    """
    # Create initial bone region mask
    bone_mask = (ct_volume >= bone_threshold_low) & (ct_volume <= bone_threshold_high)
    
    # Constrain to ROI
    roi_mask = np.zeros_like(ct_volume, dtype=bool)
    roi_mask[roi_bounds['z_min']:roi_bounds['z_max'],
             roi_bounds['y_min']:roi_bounds['y_max'],
             roi_bounds['x_min']:roi_bounds['x_max']] = True
    
    bone_mask = bone_mask & roi_mask
    
    # Subtract all other tissue types
    bone_mask = boolean_subtract(bone_mask, metal_mask)
    bone_mask = boolean_subtract(bone_mask, bright_mask)
    bone_mask = boolean_subtract(bone_mask, dark_mask)
    
    return bone_mask


def create_russian_doll_segmentation(ct_volume, metal_mask, spacing, roi_bounds=None,
                                   dark_threshold_high=-150,
                                   dark_threshold_low=-1024,  # Add parameter for lower bound
                                   bone_threshold_low=500, bone_threshold_high=1500,
                                   bright_threshold_low=None, bright_threshold_high=None,
                                   bright_artifact_max_distance_cm=10.0,
                                   use_fast_mode=True,
                                   use_enhanced_mode=False,
                                   use_advanced_mode=False,
                                   use_refinement=True,
                                   progress_callback=None):
    """
    Create segmentation using Russian doll approach with smart bone/artifact discrimination.
    
    Args:
        ct_volume: 3D CT data in HU
        metal_mask: Already segmented metal mask
        spacing: Voxel spacing (z, y, x) in mm
        roi_bounds: Optional ROI bounds to constrain analysis
        dark_threshold_high: Upper threshold for dark artifacts
        bone_threshold_low: Lower threshold for bone tissue
        bone_threshold_high: Upper threshold for bone tissue
        bright_threshold_low: Lower threshold for bright artifacts (optional, defaults to bone_threshold_low)
        bright_threshold_high: Upper threshold for bright artifacts (optional, defaults to higher value)
        bright_artifact_max_distance_cm: Max distance from metal for artifacts
        use_fast_mode: Use fast discrimination (distance-based) instead of profile analysis
        use_enhanced_mode: Use enhanced edge-based discrimination
        use_advanced_mode: Use advanced texture/gradient-based discrimination
        use_refinement: Apply second-pass refinement to improve bone/artifact discrimination
        progress_callback: Optional callback function(progress, message) for progress updates
        
    Returns:
        dict: All segmentation masks including discrimination results
    """
    # Set bright thresholds if not provided (backward compatibility)
    if bright_threshold_low is None:
        bright_threshold_low = bone_threshold_low
    if bright_threshold_high is None:
        bright_threshold_high = max(bone_threshold_high, 3500)  # Bright artifacts can go much higher than bone
    # Choose discrimination method
    if use_advanced_mode:
        # Use new advanced texture/gradient-based discrimination
        # Use texture-based discrimination from consolidated module
        discriminator = ArtifactDiscriminator(DiscriminationMethod.TEXTURE_BASED)
        
        def classify_bone_vs_artifact(ct_volume, bright_mask, metal_mask=None, use_ml=False):
            result = discriminator.discriminate(ct_volume, metal_mask, bright_mask, spacing)
            confidence_map = result.get('confidence_map', np.ones_like(bright_mask, dtype=float))
            return result['bone_mask'], result['artifact_mask'], confidence_map
        
        # Create body mask to exclude air outside patient
        body_mask = create_body_mask(ct_volume, air_threshold=-400)
        
        # Create ROI mask if bounds provided
        if roi_bounds is not None:
            if isinstance(roi_bounds, dict):
                z_min = int(roi_bounds['z_min'])
                z_max = int(roi_bounds['z_max'])
                y_min = int(roi_bounds['y_min'])
                y_max = int(roi_bounds['y_max'])
                x_min = int(roi_bounds['x_min'])
                x_max = int(roi_bounds['x_max'])
            else:
                z_min, z_max, y_min, y_max, x_min, x_max = roi_bounds
                z_min, z_max, y_min, y_max, x_min, x_max = int(z_min), int(z_max), int(y_min), int(y_max), int(x_min), int(x_max)
            roi_mask = np.zeros_like(ct_volume, dtype=bool)
            roi_mask[z_min:z_max, y_min:y_max, x_min:x_max] = True
            constraint_mask = body_mask & roi_mask
        else:
            constraint_mask = body_mask
        
        # First segment dark artifacts (excluding metal and constrained to body and ROI)
        dark_mask = (ct_volume >= dark_threshold_low) & (ct_volume <= dark_threshold_high)
        dark_mask = boolean_subtract(dark_mask, metal_mask) & constraint_mask
        
        # Get combined bright regions that need discrimination (union of bright artifact and bone ranges)
        bright_artifact_mask = (ct_volume >= bright_threshold_low) & (ct_volume <= bright_threshold_high)
        bone_range_mask = (ct_volume >= bone_threshold_low) & (ct_volume <= bone_threshold_high)
        bright_mask = (bright_artifact_mask | bone_range_mask) & constraint_mask  # Union of both ranges, constrained
        bright_mask = boolean_subtract(bright_mask, metal_mask)
        bright_mask = boolean_subtract(bright_mask, dark_mask)
        
        # Apply advanced discrimination
        bone_mask, bright_artifacts, confidence_map = classify_bone_vs_artifact(
            ct_volume, bright_mask, metal_mask
        )
        
        segmentation_result = {
            'metal': metal_mask,
            'dark_artifacts': dark_mask,
            'bright_artifacts': bright_artifacts,
            'bone': bone_mask,
            'confidence_map': confidence_map,
            'method': 'advanced_texture_gradient'
        }
        
    elif use_enhanced_mode:
        segmentation_result = create_enhanced_russian_doll_segmentation(
            ct_volume,
            metal_mask,
            spacing,
            dark_range=(dark_threshold_low, dark_threshold_high),
            bone_range=(bone_threshold_low, bone_threshold_high),
            bright_range=(bright_threshold_low, bright_threshold_high),
            max_distance_cm=bright_artifact_max_distance_cm,
            roi_bounds=roi_bounds,
            progress_callback=progress_callback
        )
    elif use_fast_mode:
        segmentation_result = create_fast_russian_doll_segmentation(
            ct_volume,
            metal_mask,
            spacing,
            dark_range=(dark_threshold_low, dark_threshold_high),
            bone_range=(bone_threshold_low, bone_threshold_high),
            bright_range=(bright_threshold_low, bright_threshold_high),
            max_distance_cm=bright_artifact_max_distance_cm,
            roi_bounds=roi_bounds
        )
    else:
        segmentation_result = create_sequential_masks(
            ct_volume, 
            metal_mask, 
            spacing,
            dark_range=(dark_threshold_low, dark_threshold_high),
            bone_range=(bone_threshold_low, bone_threshold_high),
            bright_range=(bright_threshold_low, bright_threshold_high),
            bright_artifact_max_distance_cm=bright_artifact_max_distance_cm,
            roi_bounds=roi_bounds
        )
    
    # If ROI bounds provided, constrain results
    if roi_bounds is not None:
        roi_mask = np.zeros_like(ct_volume, dtype=bool)
        roi_mask[roi_bounds['z_min']:roi_bounds['z_max'],
                 roi_bounds['y_min']:roi_bounds['y_max'],
                 roi_bounds['x_min']:roi_bounds['x_max']] = True
        
        for mask_name in ['dark_artifacts', 'bone', 'bright_artifacts']:
            if mask_name in segmentation_result:
                segmentation_result[mask_name] = segmentation_result[mask_name] & roi_mask
    
    # Apply refinement to all masks
    print("\nApplying refinement to masks...")
    # Apply second-pass refinement if requested
    if use_refinement and 'bone' in segmentation_result and 'bright_artifacts' in segmentation_result:
        print("\nApplying second-pass refinement...")
        refinement_result = refine_bone_artifact_discrimination(
            segmentation_result['bone'],
            segmentation_result['bright_artifacts'],
            ct_volume,
            spacing
        )
        # Update the masks
        segmentation_result['bone'] = refinement_result['bone_mask']
        segmentation_result['bright_artifacts'] = refinement_result['artifact_mask']
    
    # Store originals in case refinement fails
    originals = {}
    for mask_name in ['dark_artifacts', 'bone', 'bright_artifacts']:
        if mask_name in segmentation_result:
            originals[mask_name] = segmentation_result[mask_name].copy()
            original_count = np.sum(segmentation_result[mask_name])
            
            # Use different refinement parameters based on the mode
            if use_enhanced_mode:
                # For enhanced mode, be much more conservative
                # The edge analysis already produces clean results
                min_component_size = 5 if mask_name == 'bone' else 3
                smooth_iters = 0  # No smoothing - preserve edge details
            else:
                # Original parameters for other modes
                min_component_size = 10
                smooth_iters = 1
            
            segmentation_result[mask_name] = refine_mask(
                segmentation_result[mask_name], 
                min_size=min_component_size,
                fill_holes=True,
                smooth_iterations=smooth_iters
            )
            refined_count = np.sum(segmentation_result[mask_name])
            print(f"  {mask_name}: {original_count:,} -> {refined_count:,} voxels after refinement")
            
            # Safety check - if refinement removed everything, restore original
            if refined_count == 0 and original_count > 0:
                print(f"  WARNING: Refinement removed all {mask_name} voxels! Restoring original.")
                segmentation_result[mask_name] = originals[mask_name]
    
    print(f"\nFinal segmentation results:")
    for mask_name, mask in segmentation_result.items():
        if isinstance(mask, np.ndarray):
            print(f"  {mask_name}: shape={mask.shape}, voxels={np.sum(mask):,}")
    
    return segmentation_result


def save_all_contours_as_nifti(masks_dict, affine, output_prefix):
    """
    Save all contour masks as separate NIFTI files.
    
    Args:
        masks_dict: Dictionary of mask_name: mask_array pairs
        affine: 4x4 affine transformation matrix
        output_prefix: Prefix for output filenames
    """
    for mask_name, mask in masks_dict.items():
        if isinstance(mask, np.ndarray):
            output_path = f"{output_prefix}_{mask_name}.nii.gz"
            mask_int = mask.astype(np.uint8)
            nifti_img = nib.Nifti1Image(mask_int, affine)
            nib.save(nifti_img, output_path)
            print(f"Saved {mask_name} to {output_path}")


def load_nifti_mask(filepath):
    """
    Load a binary mask from a NIFTI file.
    
    Args:
        filepath: Path to the NIFTI file
        
    Returns:
        Binary mask as numpy array
    """
    nifti_img = nib.load(filepath)
    mask = nifti_img.get_fdata().astype(bool)
    return mask, nifti_img.affine


def combine_masks_multilabel(masks_dict):
    """
    Combine multiple binary masks into a single multi-label mask.
    
    Args:
        masks_dict: Dictionary of {label_name: (mask, label_value)}
        
    Returns:
        Multi-label mask where each tissue type has a unique integer value
    """
    # Get shape from first mask
    first_mask = next(iter(masks_dict.values()))[0]
    multilabel = np.zeros_like(first_mask, dtype=np.uint8)
    
    # Assign labels in order of priority (metal should override others)
    priority_order = ['metal', 'bright_artifacts', 'dark_artifacts', 'bone']
    
    label_map = {}
    current_label = 1
    
    for tissue_type in priority_order:
        if tissue_type in masks_dict:
            mask, _ = masks_dict[tissue_type]
            multilabel[mask] = current_label
            label_map[tissue_type] = current_label
            current_label += 1
    
    # Handle any remaining masks
    for tissue_type, (mask, _) in masks_dict.items():
        if tissue_type not in priority_order:
            multilabel[mask] = current_label
            label_map[tissue_type] = current_label
            current_label += 1
    
    return multilabel, label_map