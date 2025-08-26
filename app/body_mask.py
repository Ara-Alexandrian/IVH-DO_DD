"""
Body mask extraction utilities to constrain artifacts to within the body.
"""
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation, label
from scipy.ndimage import generate_binary_structure


def create_body_mask(ct_volume, air_threshold=-300):
    """
    Create a mask of the body region, excluding air outside the patient.
    
    Args:
        ct_volume: 3D CT volume in HU
        air_threshold: HU threshold below which is considered air (default -300)
        
    Returns:
        3D binary mask where True indicates inside the body
    """
    # Create initial body mask by thresholding - use higher threshold to be more conservative
    body_mask = ct_volume > air_threshold
    
    # Process each slice to get the largest connected component (the body)
    processed_mask = np.zeros_like(body_mask)
    
    for z in range(body_mask.shape[0]):
        slice_mask = body_mask[z]
        
        if not np.any(slice_mask):
            continue
            
        # Fill holes (like lungs)
        filled = binary_fill_holes(slice_mask)
        
        # Find largest connected component (the body)
        labeled, num_features = label(filled)
        if num_features > 0:
            # Find the largest component
            sizes = np.bincount(labeled.ravel())[1:]  # Skip background
            if len(sizes) > 0:
                largest_label = np.argmax(sizes) + 1
                processed_mask[z] = labeled == largest_label
    
    # Apply more aggressive morphological operations to remove thin connections
    struct = generate_binary_structure(3, 1)
    # More erosion to remove thin exterior regions
    processed_mask = binary_erosion(processed_mask, struct, iterations=2)
    processed_mask = binary_dilation(processed_mask, struct, iterations=2)
    processed_mask = binary_erosion(processed_mask, struct, iterations=1)
    
    # Additional cleanup: remove small disconnected components in 3D
    labeled_3d, num_features_3d = label(processed_mask)
    if num_features_3d > 0:
        sizes_3d = np.bincount(labeled_3d.ravel())[1:]
        if len(sizes_3d) > 0:
            # Keep only the largest component (the main body)
            largest_3d = np.argmax(sizes_3d) + 1
            processed_mask = labeled_3d == largest_3d
    
    return processed_mask


def constrain_to_body(mask, body_mask):
    """
    Constrain a mask to only include voxels within the body.
    
    Args:
        mask: Binary mask to constrain
        body_mask: Body region mask
        
    Returns:
        Constrained mask
    """
    return mask & body_mask