"""
Unified Artifact Discrimination Module
Consolidates all bone/artifact discrimination algorithms into a single interface.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel
from scipy.ndimage import binary_opening, binary_closing, label


class DiscriminationMethod(Enum):
    """Available discrimination methods."""
    DISTANCE_BASED = "distance_based"      # Fast, distance from metal
    EDGE_BASED = "edge_based"              # Enhanced edge coherence
    TEXTURE_BASED = "texture_based"        # Advanced texture/gradient
    STAR_PROFILE = "star_profile"          # Original star profile
    PROFILE_BASED = "star_profile"         # Alias for backward compatibility


class ArtifactDiscriminator:
    """
    Unified interface for bone vs bright artifact discrimination.
    
    Methods:
        DISTANCE_BASED: Fast discrimination using distance from metal
        EDGE_BASED: Enhanced edge coherence analysis
        TEXTURE_BASED: Advanced texture and gradient features
        STAR_PROFILE: Original star profile analysis
    """
    
    def __init__(self, method: DiscriminationMethod = DiscriminationMethod.DISTANCE_BASED):
        """
        Initialize discriminator with specified method.
        
        Args:
            method: Discrimination method to use
        """
        self.method = method
        self.discriminators = {
            DiscriminationMethod.DISTANCE_BASED: self._discriminate_distance,
            DiscriminationMethod.EDGE_BASED: self._discriminate_edge,
            DiscriminationMethod.TEXTURE_BASED: self._discriminate_texture,
            DiscriminationMethod.STAR_PROFILE: self._discriminate_star
        }
    
    def discriminate(self, ct_volume: np.ndarray, metal_mask: np.ndarray,
                    bright_mask: np.ndarray, spacing: Tuple[float, float, float],
                    **kwargs) -> Dict:
        """
        Discriminate between bone and bright artifacts.
        
        Args:
            ct_volume: 3D CT volume in HU
            metal_mask: Binary mask of metal regions
            bright_mask: Binary mask of bright regions to discriminate
            spacing: Voxel spacing (z, y, x) in mm
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary containing:
                - bone_mask: Binary mask of bone tissue
                - artifact_mask: Binary mask of bright artifacts
                - confidence_map: Confidence scores for discrimination
                - metadata: Method-specific metadata
        """
        discriminator = self.discriminators.get(self.method)
        if not discriminator:
            raise ValueError(f"Unknown discrimination method: {self.method}")
        
        return discriminator(ct_volume, metal_mask, bright_mask, spacing, **kwargs)
    
    def _discriminate_distance(self, ct_volume: np.ndarray, metal_mask: np.ndarray,
                              bright_mask: np.ndarray, spacing: Tuple[float, float, float],
                              max_distance_cm: float = 10.0) -> Dict:
        """
        Fast distance-based discrimination.
        
        Principle: Bright artifacts are typically closer to metal than bone.
        
        Algorithm:
        1. Calculate distance from metal
        2. Apply distance-based classification
        3. Use local smoothness as secondary criterion
        """
        # Calculate distance from metal
        inverted_metal = np.logical_not(metal_mask)
        distances = distance_transform_edt(inverted_metal, sampling=spacing)
        distances_cm = distances / 10.0
        
        # Smooth CT for texture analysis
        smoothed = gaussian_filter(ct_volume.astype(float), sigma=2.0)
        
        # Calculate local variance
        local_variance = np.zeros_like(ct_volume, dtype=float)
        for z in range(1, ct_volume.shape[0]-1):
            slice_std = np.std([ct_volume[z-1], ct_volume[z], ct_volume[z+1]], axis=0)
            local_variance[z] = slice_std
        
        # Classification based on distance and smoothness
        bone_mask = np.zeros_like(bright_mask)
        artifact_mask = np.zeros_like(bright_mask)
        confidence_map = np.zeros_like(ct_volume, dtype=float)
        
        # More relaxed bone criteria - bone is typically:
        # - At moderate distance from metal (not too close, not too far)
        # - Has moderate HU values (400-1500)
        # - Has relatively low variance (smooth structure)
        bone_criteria = bright_mask & \
                       (distances_cm > 1.0) & \
                       (ct_volume >= 400) & (ct_volume <= 1500) & \
                       (local_variance < 300)  # Increased variance threshold
        
        # Artifact characteristics: very close to metal OR high variance OR very high HU
        artifact_criteria = bright_mask & \
                          ((distances_cm < 1.0) |  # Very close to metal
                           (local_variance > 400) |  # High variance
                           (ct_volume > 1500))  # Very high HU (likely artifact)
        
        bone_mask = bone_criteria
        artifact_mask = artifact_criteria & (~bone_mask)
        
        # Handle unclassified regions with better heuristics
        unclassified = bright_mask & (~bone_mask) & (~artifact_mask)
        if np.any(unclassified):
            # Use HU value as primary criterion for unclassified
            # Bone is typically 400-1000 HU, artifacts can be higher
            bone_hu_range = unclassified & (ct_volume >= 400) & (ct_volume <= 1000)
            artifact_hu_range = unclassified & (ct_volume > 1000)
            
            # Secondary criterion: distance
            # Very close = artifact, moderate distance = bone
            very_close = unclassified & (distances_cm < 0.5)
            moderate_dist = unclassified & (distances_cm >= 0.5) & (distances_cm < 3.0)
            
            # Combine criteria
            bone_mask |= (bone_hu_range & moderate_dist)
            artifact_mask |= (artifact_hu_range | very_close)
            
            # Remaining unclassified: use distance threshold
            still_unclassified = unclassified & (~bone_mask) & (~artifact_mask)
            near_metal = still_unclassified & (distances_cm < 2.0)
            far_from_metal = still_unclassified & (distances_cm >= 2.0)
            artifact_mask |= near_metal
            bone_mask |= far_from_metal
        
        # Calculate confidence based on distance and variance
        confidence_map[bright_mask] = np.clip(
            1.0 - (local_variance[bright_mask] / 500.0), 0, 1
        )
        
        return {
            'bone_mask': bone_mask,
            'artifact_mask': artifact_mask,
            'confidence_map': confidence_map,
            'distance_map': distances_cm,
            'method': 'distance_based',
            'metadata': {
                'max_distance_cm': max_distance_cm,
                'bone_voxels': np.sum(bone_mask),
                'artifact_voxels': np.sum(artifact_mask)
            }
        }
    
    def _discriminate_edge(self, ct_volume: np.ndarray, metal_mask: np.ndarray,
                          bright_mask: np.ndarray, spacing: Tuple[float, float, float],
                          **kwargs) -> Dict:
        """
        Enhanced edge-based discrimination.
        
        Principle: Bone has coherent, continuous edges while artifacts have chaotic edges.
        
        Algorithm:
        1. Compute edge coherence using structure tensor
        2. Analyze edge continuity across slices
        3. Measure radial vs tangential edge alignment
        """
        # Compute gradients
        grad_z = sobel(ct_volume, axis=0)
        grad_y = sobel(ct_volume, axis=1)
        grad_x = sobel(ct_volume, axis=2)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
        
        # Edge coherence (simplified structure tensor analysis)
        smoothed_grad_mag = gaussian_filter(grad_mag, sigma=1.0)
        
        # Initialize masks
        bone_mask = np.zeros_like(bright_mask)
        artifact_mask = np.zeros_like(bright_mask)
        confidence_map = np.zeros_like(ct_volume, dtype=float)
        
        # Bone: strong, coherent edges
        strong_edges = (smoothed_grad_mag > np.percentile(smoothed_grad_mag[bright_mask], 75))
        coherent_regions = bright_mask & strong_edges
        
        # Analyze edge continuity
        for z in range(ct_volume.shape[0]):
            if not np.any(bright_mask[z]):
                continue
            
            slice_bright = bright_mask[z]
            slice_edges = coherent_regions[z]
            
            # Bone has continuous edges
            labeled_edges, num_features = label(slice_edges)
            for i in range(1, num_features + 1):
                component = labeled_edges == i
                if np.sum(component) > 50:  # Significant edge component
                    bone_mask[z] |= component & slice_bright
        
        # Remaining bright regions are artifacts
        artifact_mask = bright_mask & (~bone_mask)
        
        # Confidence based on edge strength
        confidence_map[bright_mask] = np.clip(
            smoothed_grad_mag[bright_mask] / np.max(smoothed_grad_mag[bright_mask]), 0, 1
        )
        
        return {
            'bone_mask': bone_mask,
            'artifact_mask': artifact_mask,
            'confidence_map': confidence_map,
            'edge_magnitude': grad_mag,
            'method': 'edge_based',
            'metadata': {
                'bone_voxels': np.sum(bone_mask),
                'artifact_voxels': np.sum(artifact_mask)
            }
        }
    
    def _discriminate_texture(self, ct_volume: np.ndarray, metal_mask: np.ndarray,
                             bright_mask: np.ndarray, spacing: Tuple[float, float, float],
                             **kwargs) -> Dict:
        """
        Advanced texture-based discrimination using multiple features.
        
        Principle: Bone and artifacts have different textural characteristics.
        
        Features:
        - Local variance
        - Gradient direction variance
        - Smoothness measures
        - Distance from metal
        """
        # Calculate texture features
        local_variance = self._calculate_local_variance(ct_volume, window_size=5)
        gradient_variance = self._calculate_gradient_variance(ct_volume)
        smoothness = self._calculate_smoothness(ct_volume)
        
        # Distance from metal
        inverted_metal = np.logical_not(metal_mask)
        distances = distance_transform_edt(inverted_metal, sampling=spacing)
        distances_cm = distances / 10.0
        
        # Combine features for classification
        # Normalize features
        features = {
            'variance': local_variance / (np.max(local_variance) + 1e-10),
            'gradient': gradient_variance / (np.max(gradient_variance) + 1e-10),
            'smoothness': smoothness / (np.max(smoothness) + 1e-10),
            'distance': np.clip(distances_cm / 10.0, 0, 1)
        }
        
        # Artifact score (higher = more likely artifact)
        artifact_score = np.zeros_like(ct_volume, dtype=float)
        
        # Weights for features
        weights = {
            'variance': 0.3,      # High variance = artifact
            'gradient': 0.3,      # High gradient variance = artifact
            'smoothness': -0.2,   # Low smoothness = artifact
            'distance': -0.2      # Close to metal = artifact
        }
        
        for feature_name, weight in weights.items():
            artifact_score += weight * features[feature_name]
        
        # Normalize scores
        artifact_score = np.clip(artifact_score, 0, 1)
        
        # Threshold to create masks (using Otsu-like approach)
        bright_scores = artifact_score[bright_mask]
        if len(bright_scores) > 0:
            threshold = np.median(bright_scores)
        else:
            threshold = 0.5
        
        # Create masks
        artifact_mask = bright_mask & (artifact_score > threshold)
        bone_mask = bright_mask & (artifact_score <= threshold)
        
        # Post-processing
        artifact_mask = binary_opening(artifact_mask, iterations=1)
        artifact_mask = binary_closing(artifact_mask, iterations=1)
        bone_mask = binary_opening(bone_mask, iterations=1)
        bone_mask = binary_closing(bone_mask, iterations=1)
        
        # Confidence map
        confidence_map = np.abs(artifact_score - threshold) * 2
        confidence_map = np.clip(confidence_map, 0, 1)
        
        return {
            'bone_mask': bone_mask,
            'artifact_mask': artifact_mask,
            'confidence_map': confidence_map,
            'artifact_score': artifact_score,
            'method': 'texture_based',
            'metadata': {
                'threshold': threshold,
                'bone_voxels': np.sum(bone_mask),
                'artifact_voxels': np.sum(artifact_mask)
            }
        }
    
    def _discriminate_star(self, ct_volume: np.ndarray, metal_mask: np.ndarray,
                          bright_mask: np.ndarray, spacing: Tuple[float, float, float],
                          num_angles: int = 16, **kwargs) -> Dict:
        """
        Original star profile discrimination.
        
        Principle: Analyze radial profiles from metal centers to classify tissue.
        
        Algorithm:
        1. Shoot radial lines from metal centers
        2. Analyze HU profiles along lines
        3. Classify based on profile characteristics
        """
        # Find metal centers
        z_coords, y_coords, x_coords = np.where(metal_mask)
        if len(z_coords) == 0:
            return self._empty_result()
        
        center_z = int(np.mean(z_coords))
        center_y = int(np.mean(y_coords))
        center_x = int(np.mean(x_coords))
        
        # Initialize masks
        bone_mask = np.zeros_like(bright_mask)
        artifact_mask = np.zeros_like(bright_mask)
        confidence_map = np.zeros_like(ct_volume, dtype=float)
        
        # Analyze each slice with metal
        for z in range(ct_volume.shape[0]):
            if not np.any(metal_mask[z]) or not np.any(bright_mask[z]):
                continue
            
            slice_data = ct_volume[z]
            slice_bright = bright_mask[z]
            
            # Find metal center in this slice
            y_metal, x_metal = np.where(metal_mask[z])
            if len(y_metal) > 0:
                slice_center_y = int(np.mean(y_metal))
                slice_center_x = int(np.mean(x_metal))
                
                # Analyze profiles
                profiles = self._get_star_profiles(
                    slice_data, slice_center_y, slice_center_x, num_angles
                )
                
                # Classify based on profile characteristics
                for y, x in zip(*np.where(slice_bright)):
                    # Distance from center
                    dist = np.sqrt((y - slice_center_y)**2 + (x - slice_center_x)**2)
                    
                    # Angle from center
                    angle = np.arctan2(y - slice_center_y, x - slice_center_x)
                    
                    # Find closest profile
                    profile_idx = int((angle + np.pi) / (2 * np.pi) * num_angles) % num_angles
                    
                    if profile_idx < len(profiles):
                        profile = profiles[profile_idx]
                        
                        # Analyze profile at this distance
                        if dist < len(profile):
                            local_value = slice_data[y, x]
                            profile_value = profile[int(dist)]
                            
                            # Bone: consistent with smooth profile
                            # Artifact: deviates from smooth profile
                            deviation = abs(local_value - profile_value)
                            
                            if deviation < 200:  # Consistent with profile
                                bone_mask[z, y, x] = True
                            else:  # Deviates from profile
                                artifact_mask[z, y, x] = True
                            
                            confidence_map[z, y, x] = 1.0 - (deviation / 1000.0)
        
        # Ensure mutual exclusivity
        artifact_mask = artifact_mask & (~bone_mask)
        
        return {
            'bone_mask': bone_mask,
            'artifact_mask': artifact_mask,
            'confidence_map': np.clip(confidence_map, 0, 1),
            'method': 'star_profile',
            'metadata': {
                'num_angles': num_angles,
                'bone_voxels': np.sum(bone_mask),
                'artifact_voxels': np.sum(artifact_mask)
            }
        }
    
    def _calculate_local_variance(self, volume: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Calculate local variance in a sliding window."""
        from scipy.ndimage import uniform_filter
        
        mean = uniform_filter(volume.astype(float), size=window_size)
        sqr_mean = uniform_filter(volume.astype(float)**2, size=window_size)
        variance = sqr_mean - mean**2
        
        return np.maximum(variance, 0)
    
    def _calculate_gradient_variance(self, volume: np.ndarray) -> np.ndarray:
        """Calculate gradient direction variance."""
        grad_z = sobel(volume, axis=0)
        grad_y = sobel(volume, axis=1)
        grad_x = sobel(volume, axis=2)
        
        # Calculate angles
        with np.errstate(divide='ignore', invalid='ignore'):
            angles_xy = np.arctan2(grad_y, grad_x)
            angles_xz = np.arctan2(grad_z, grad_x)
        
        # Local variance of angles
        angle_variance = self._calculate_local_variance(angles_xy, window_size=3)
        angle_variance += self._calculate_local_variance(angles_xz, window_size=3)
        
        return angle_variance
    
    def _calculate_smoothness(self, volume: np.ndarray) -> np.ndarray:
        """Calculate local smoothness measure."""
        smoothed = gaussian_filter(volume.astype(float), sigma=2.0)
        difference = np.abs(volume - smoothed)
        smoothness = 1.0 / (1.0 + difference)
        
        return smoothness
    
    def _get_star_profiles(self, slice_data: np.ndarray, center_y: int, center_x: int,
                          num_angles: int) -> list:
        """Get radial profiles from center point."""
        profiles = []
        max_radius = max(slice_data.shape)
        
        for i in range(num_angles):
            angle = 2 * np.pi * i / num_angles
            profile = []
            
            for r in range(max_radius):
                y = int(center_y + r * np.sin(angle))
                x = int(center_x + r * np.cos(angle))
                
                if 0 <= y < slice_data.shape[0] and 0 <= x < slice_data.shape[1]:
                    profile.append(slice_data[y, x])
                else:
                    break
            
            if profile:
                # Smooth the profile
                from scipy.ndimage import gaussian_filter1d
                profile = gaussian_filter1d(profile, sigma=2.0)
                profiles.append(profile)
        
        return profiles
    
    def _empty_result(self) -> Dict:
        """Return empty result when discrimination fails."""
        return {
            'bone_mask': np.array([]),
            'artifact_mask': np.array([]),
            'confidence_map': np.array([]),
            'method': self.method.value,
            'metadata': {}
        }


# Convenience functions for backward compatibility
def discriminate_fast(ct_volume: np.ndarray, metal_mask: np.ndarray,
                      bright_mask: np.ndarray, spacing: Tuple[float, float, float], **kwargs) -> Dict:
    """Fast distance-based discrimination."""
    discriminator = ArtifactDiscriminator(DiscriminationMethod.DISTANCE_BASED)
    return discriminator.discriminate(ct_volume, metal_mask, bright_mask, spacing, **kwargs)


def discriminate_enhanced(ct_volume: np.ndarray, metal_mask: np.ndarray,
                         bright_mask: np.ndarray, spacing: Tuple[float, float, float], **kwargs) -> Dict:
    """Enhanced edge-based discrimination."""
    discriminator = ArtifactDiscriminator(DiscriminationMethod.EDGE_BASED)
    return discriminator.discriminate(ct_volume, metal_mask, bright_mask, spacing, **kwargs)


def discriminate_advanced(ct_volume: np.ndarray, metal_mask: np.ndarray,
                         bright_mask: np.ndarray, spacing: Tuple[float, float, float], **kwargs) -> Dict:
    """Advanced texture-based discrimination."""
    discriminator = ArtifactDiscriminator(DiscriminationMethod.TEXTURE_BASED)
    return discriminator.discriminate(ct_volume, metal_mask, bright_mask, spacing, **kwargs)