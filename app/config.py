"""
Centralized configuration module for HU thresholds and analysis parameters.
This module defines all threshold ranges, defaults, and validation logic.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import streamlit as st


@dataclass
class ThresholdRange:
    """Defines a threshold range with min/max bounds and defaults."""
    min_bound: float
    max_bound: float
    default_min: float
    default_max: float
    step: float = 1.0
    label: str = ""
    help_text: str = ""
    
    def validate(self, min_val: float, max_val: float) -> Tuple[bool, str]:
        """Validate threshold values are within bounds and properly ordered."""
        if min_val < self.min_bound or min_val > self.max_bound:
            return False, f"{self.label} minimum must be between {self.min_bound} and {self.max_bound}"
        if max_val < self.min_bound or max_val > self.max_bound:
            return False, f"{self.label} maximum must be between {self.min_bound} and {self.max_bound}"
        if min_val >= max_val:
            return False, f"{self.label} minimum must be less than maximum"
        return True, ""
    
    def get_defaults(self) -> Tuple[float, float]:
        """Return default min and max values."""
        return self.default_min, self.default_max


@dataclass
class SingleThreshold:
    """Defines a single threshold with bounds and default."""
    min_bound: float
    max_bound: float
    default: float
    step: float = 1.0
    label: str = ""
    help_text: str = ""
    
    def validate(self, value: float) -> Tuple[bool, str]:
        """Validate threshold value is within bounds."""
        if value < self.min_bound or value > self.max_bound:
            return False, f"{self.label} must be between {self.min_bound} and {self.max_bound}"
        return True, ""
    
    def get_default(self) -> float:
        """Return default value."""
        return self.default


class ThresholdConfig:
    """Central configuration for all HU thresholds."""
    
    # Dark Artifacts: Very low HU values (metal artifacts causing beam hardening)
    DARK_ARTIFACTS = ThresholdRange(
        min_bound=-1024,
        max_bound=-50,
        default_min=-1024,
        default_max=-150,
        step=10.0,
        label="Dark Artifacts",
        help_text="Dark streaking artifacts from beam hardening (below -150 HU, air is -1000 HU)"
    )
    
    # Bright Artifacts: High HU values (metal artifacts causing brightening)
    BRIGHT_ARTIFACTS = ThresholdRange(
        min_bound=50,
        max_bound=10000,
        default_min=200,
        default_max=3000,
        step=10.0,
        label="Bright Artifacts",
        help_text="Bright streaking artifacts from metal (50-3000+ HU, can occur in any tissue range)"
    )
    
    # Bone Tissue: Normal bone HU range  
    BONE_TISSUE = ThresholdRange(
        min_bound=300,
        max_bound=2500,
        default_min=400,
        default_max=1800,
        step=10.0,
        label="Bone Tissue",
        help_text="Normal bone tissue range (trabecular ~400-800 HU, cortical 800-2000 HU)"
    )
    
    # Metal Detection: Very high HU values for metal implants
    METAL_THRESHOLD = SingleThreshold(
        min_bound=2000,
        max_bound=15000,
        default=3000,
        step=50.0,
        label="Metal Detection",
        help_text="Minimum HU value to detect metal implants (>3000 HU, 16-bit CT allows >10000 HU)"
    )
    
    # Legacy method thresholds (for backward compatibility)
    LEGACY_BRIGHT_ARTIFACTS = ThresholdRange(
        min_bound=500,
        max_bound=4000,
        default_min=800,
        default_max=3000,
        step=50.0,
        label="Legacy Bright Artifacts",
        help_text="Legacy method bright artifact range"
    )
    
    LEGACY_DARK_THRESHOLD = SingleThreshold(
        min_bound=-1024,
        max_bound=0,
        default=-200,
        step=10.0,
        label="Legacy Dark Threshold",
        help_text="Legacy method dark artifact maximum threshold"
    )
    
    # Russian Doll thresholds - separate for bright artifacts and bone
    RUSSIAN_DOLL_BRIGHT_ARTIFACTS = ThresholdRange(
        min_bound=100,
        max_bound=8000,
        default_min=200,
        default_max=2500,
        step=50.0,
        label="Bright Artifacts HU Range",
        help_text="HU range for bright metal artifacts (can occur in soft tissue 50-200 HU to bone 2000+ HU)"
    )
    
    RUSSIAN_DOLL_BONE = ThresholdRange(
        min_bound=200,
        max_bound=2500,
        default_min=400,
        default_max=1800,
        step=50.0,
        label="Bone Tissue HU Range",
        help_text="HU range for bone tissue (trabecular ~200-800 HU, cortical 800-2000 HU)"
    )
    
    # Analysis parameters
    MAX_ARTIFACT_DISTANCE = SingleThreshold(
        min_bound=1.0,
        max_bound=20.0,
        default=10.0,
        step=0.5,
        label="Max Artifact Distance",
        help_text="Maximum distance from metal to consider artifacts (cm)"
    )
    
    SEARCH_MARGIN_3D = SingleThreshold(
        min_bound=0.5,
        max_bound=5.0,
        default=2.0,
        step=0.1,
        label="3D Search Margin",
        help_text="Margin around metal for 3D adaptive search (cm)"
    )
    
    @classmethod
    def get_russian_doll_defaults(cls) -> dict:
        """Get default threshold values for Russian Doll method."""
        return {
            'dark_min': cls.DARK_ARTIFACTS.default_min,
            'dark_max': cls.DARK_ARTIFACTS.default_max,
            'bright_min': cls.BRIGHT_ARTIFACTS.default_min,
            'bright_max': cls.BRIGHT_ARTIFACTS.default_max,
            'bone_min': cls.BONE_TISSUE.default_min,
            'bone_max': cls.BONE_TISSUE.default_max,
            'max_distance': cls.MAX_ARTIFACT_DISTANCE.default
        }
    
    @classmethod
    def get_legacy_defaults(cls) -> dict:
        """Get default threshold values for Legacy method."""
        return {
            'bright_min': cls.LEGACY_BRIGHT_ARTIFACTS.default_min,
            'bright_max': cls.LEGACY_BRIGHT_ARTIFACTS.default_max,
            'dark_max': cls.LEGACY_DARK_THRESHOLD.default,
            'bone_min': cls.BONE_TISSUE.default_min,
            'bone_max': cls.BONE_TISSUE.default_max
        }
    
    @classmethod
    def get_metal_detection_defaults(cls) -> dict:
        """Get default values for metal detection."""
        return {
            'metal_threshold': cls.METAL_THRESHOLD.default,
            'search_margin_3d': cls.SEARCH_MARGIN_3D.default
        }


def init_threshold_state():
    """Initialize Streamlit session state with threshold values."""
    if 'thresholds' not in st.session_state:
        st.session_state.thresholds = {}
    
    # Initialize Russian Doll thresholds
    if 'russian_doll' not in st.session_state.thresholds:
        st.session_state.thresholds['russian_doll'] = ThresholdConfig.get_russian_doll_defaults()
    
    # Initialize Legacy thresholds
    if 'legacy' not in st.session_state.thresholds:
        st.session_state.thresholds['legacy'] = ThresholdConfig.get_legacy_defaults()
    
    # Initialize Metal Detection thresholds
    if 'metal_detection' not in st.session_state.thresholds:
        st.session_state.thresholds['metal_detection'] = ThresholdConfig.get_metal_detection_defaults()


def reset_thresholds(method: str = 'all'):
    """Reset thresholds to default values.
    
    Args:
        method: Which thresholds to reset ('russian_doll', 'legacy', 'metal_detection', or 'all')
    """
    if method == 'all' or method == 'russian_doll':
        st.session_state.thresholds['russian_doll'] = ThresholdConfig.get_russian_doll_defaults()
    
    if method == 'all' or method == 'legacy':
        st.session_state.thresholds['legacy'] = ThresholdConfig.get_legacy_defaults()
    
    if method == 'all' or method == 'metal_detection':
        st.session_state.thresholds['metal_detection'] = ThresholdConfig.get_metal_detection_defaults()


def validate_all_thresholds() -> Tuple[bool, list]:
    """Validate all current threshold values.
    
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    
    # Validate Russian Doll thresholds
    if 'russian_doll' in st.session_state.thresholds:
        rd = st.session_state.thresholds['russian_doll']
        
        # Dark artifacts
        valid, msg = ThresholdConfig.DARK_ARTIFACTS.validate(rd['dark_min'], rd['dark_max'])
        if not valid:
            errors.append(msg)
        
        # Bright artifacts
        valid, msg = ThresholdConfig.BRIGHT_ARTIFACTS.validate(rd['bright_min'], rd['bright_max'])
        if not valid:
            errors.append(msg)
        
        # Bone tissue
        valid, msg = ThresholdConfig.BONE_TISSUE.validate(rd['bone_min'], rd['bone_max'])
        if not valid:
            errors.append(msg)
        
        # Check for overlaps
        if rd['dark_max'] >= rd['bright_min']:
            errors.append("Dark artifact maximum must be less than bright artifact minimum")
    
    # Validate Metal Detection thresholds
    if 'metal_detection' in st.session_state.thresholds:
        md = st.session_state.thresholds['metal_detection']
        
        valid, msg = ThresholdConfig.METAL_THRESHOLD.validate(md['metal_threshold'])
        if not valid:
            errors.append(msg)
    
    return len(errors) == 0, errors


# Color scheme for visualization (RGBA format)
COLORS = {
    'metal': (1.0, 0.0, 0.0, 0.7),      # Red
    'bright': (1.0, 1.0, 0.0, 0.6),     # Yellow
    'dark': (1.0, 0.0, 1.0, 0.6),       # Magenta
    'bone': (0.0, 0.2, 0.8, 0.5),       # Blue
    'roi': (0.0, 1.0, 0.0, 1.0),        # Lime (ROI boundary)
}

# Export formats
EXPORT_FORMATS = ['NIFTI', 'DICOM RT Structure']

# Performance settings
CACHE_ENABLED = True
PROGRESS_UPDATE_INTERVAL = 0.1  # seconds
MAX_PREVIEW_SLICES = 5  # for real-time preview