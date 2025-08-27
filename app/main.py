import streamlit as st
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import time

from dicom_utils import load_dicom_series_to_hu, create_metal_mask_from_rtstruct
# Legacy functions moved inline to fix import errors
from core.metal_detection import MetalDetector, MetalDetectionMethod

# Legacy wrapper functions
def detect_metal_volume(ct_volume, spacing, margin_cm=2.0, fw_percentage=75.0, 
                       min_metal_hu=2500, dilation_iterations=2):
    """Legacy wrapper for metal detection."""
    detector = MetalDetector(MetalDetectionMethod.LEGACY)
    return detector.detect(
        ct_volume, spacing,
        min_metal_hu=min_metal_hu,
        margin_cm=margin_cm,
        fw_percentage=fw_percentage,
        dilation_iterations=dilation_iterations
    )

def create_affine_from_dicom_meta(metadata):
    """Create affine transformation matrix from DICOM metadata."""
    origin = metadata.get('origin', [0, 0, 0])
    spacing = metadata.get('spacing', [1, 1, 1])
    
    affine = np.eye(4)
    affine[0, 0] = spacing[2]  # x spacing
    affine[1, 1] = spacing[1]  # y spacing
    affine[2, 2] = spacing[0]  # z spacing
    affine[0, 3] = origin[0]   # x origin
    affine[1, 3] = origin[1]   # y origin
    affine[2, 3] = origin[2]   # z origin
    return affine

def save_mask_as_nifti(mask, affine, filepath):
    """Save binary mask as NIFTI file."""
    img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nib.save(img, filepath)
    return filepath
from core.discrimination import ArtifactDiscriminator, DiscriminationMethod
from contour_operations import (create_bright_artifact_mask, create_dark_artifact_mask,
                               create_bone_mask, save_all_contours_as_nifti, refine_mask,
                               create_russian_doll_segmentation)
from visualization import (create_overlay_image, create_histogram, fig_to_base64, 
                          create_multi_slice_view, visualize_star_profiles,
                          plot_threshold_evolution, visualize_discrimination_slice,
                          create_histogram_with_thresholds, create_threshold_preview)
from config import ThresholdConfig, init_threshold_state, reset_thresholds, validate_all_thresholds

# Page configuration
st.set_page_config(
    page_title="CT Metal Artifact Characterization",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ct_volume' not in st.session_state:
    st.session_state.ct_volume = None
    st.session_state.ct_metadata = None
    st.session_state.current_slice = 0
    st.session_state.masks = {}
    st.session_state.selected_patient = None
    st.session_state.metal_detection_result = None
    st.session_state.affine = None

# Initialize threshold configuration state
init_threshold_state()

# Header
st.title("🏥 CT Metal Artifact Characterization")
st.markdown("Advanced segmentation with automatic metal detection and boolean operations")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Patient selection
    data_dir = Path("../data")
    if not data_dir.exists():
        data_dir = Path("data")  # Fallback for different run contexts
    
    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir() and "HIP" in d.name] if data_dir.exists() else []
    patient_names = sorted([d.name for d in patient_dirs])
    
    selected_patient = st.selectbox(
        "Select Patient",
        patient_names,
        help="Choose a patient dataset to analyze"
    )
    
    if selected_patient != st.session_state.selected_patient:
        # Reset state when patient changes
        st.session_state.ct_volume = None
        st.session_state.ct_metadata = None
        st.session_state.masks = {}
        st.session_state.metal_detection_result = None
        st.session_state.selected_patient = selected_patient
    
    # Load data button
    if st.button("Load Patient Data", type="primary"):
        with st.spinner("Loading DICOM data..."):
            patient_path = data_dir / selected_patient
            
            # Find CT directory
            ct_dirs = [d for d in patient_path.iterdir() if d.is_dir() and "CT" in d.name]
            if ct_dirs:
                ct_dir = ct_dirs[0]
                ct_volume, ct_metadata = load_dicom_series_to_hu(str(ct_dir))
                
                if ct_volume is not None:
                    st.session_state.ct_volume = ct_volume
                    st.session_state.ct_metadata = ct_metadata
                    st.session_state.current_slice = ct_volume.shape[0] // 2
                    st.session_state.affine = create_affine_from_dicom_meta(ct_metadata)
                    st.success(f"Loaded {ct_volume.shape[0]} slices successfully!")
                else:
                    st.error("Failed to load CT data")
            else:
                st.error("No CT directory found for this patient")
    
    st.markdown("---")
    
    # Analysis parameters
    st.subheader("Analysis Parameters")
    
    analysis_tab1, analysis_tab2 = st.tabs(["Metal Detection", "Artifact Thresholds"])
    
    with analysis_tab1:
        st.markdown("**Advanced Metal Detection**")
        st.info("💡 3D adaptive method combines coronal/sagittal analysis with star profile algorithm")
        
        detection_method = st.radio(
            "Detection Method",
            ["3D Adaptive + Star Algorithm (Recommended)", "Legacy with Initial Threshold"],
            help="3D Adaptive: Combines 3D coronal/sagittal analysis with star profile algorithm. Legacy: Uses initial HU threshold."
        )
        
        if detection_method == "3D Adaptive + Star Algorithm (Recommended)":
            margin_cm = st.slider(
                "3D Search Margin (cm)",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="Margin around detected metal extent in all planes"
            )
            
            fw_percentage = st.slider(
                "Full Width Percentage",
                min_value=50,
                max_value=90,
                value=75,
                step=5,
                help="Percentage of peak for threshold detection (lower = more inclusive)"
            )
            
            intensity_percentile = st.slider(
                "Initial Detection Percentile",
                min_value=99.0,
                max_value=99.9,
                value=99.5,
                step=0.1,
                help="Top percentile of voxels to use for initial detection"
            )
            
        else:
            # Legacy parameters
            roi_margin_cm = st.slider(
                "ROI Margin (cm)",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                help="Margin around detected metal in centimeters"
            )
            
            # Metal Detection Threshold Slider
            min_metal_hu = st.slider(
                "Initial Metal Threshold (HU)",
                min_value=int(ThresholdConfig.METAL_THRESHOLD.min_bound),
                max_value=int(ThresholdConfig.METAL_THRESHOLD.max_bound),
                value=int(st.session_state.thresholds['metal_detection']['metal_threshold']),
                step=int(ThresholdConfig.METAL_THRESHOLD.step),
                help=ThresholdConfig.METAL_THRESHOLD.help_text,
                key="metal_threshold_slider"
            )
            st.session_state.thresholds['metal_detection']['metal_threshold'] = min_metal_hu
            
            fw_percentage = st.slider(
                "Full Width Percentage",
                min_value=50,
                max_value=90,
                value=60,
                step=5,
                help="Percentage of peak for threshold detection"
            )
            
            dilation_iterations = st.slider(
                "Metal Region Connection",
                min_value=1,
                max_value=10,
                value=5,
                help="Dilation iterations to connect nearby metal regions"
            )
    
    with analysis_tab2:
        st.markdown("**Artifact Segmentation Method**")
        
        segmentation_method = st.radio(
            "Segmentation Approach",
            ["Russian Doll with Smart Discrimination (Recommended)", 
             "Context-Aware Artifact Detection (Best for Low HU Artifacts)",
             "Legacy Threshold-Based"],
            help="Smart: Fast distance-based discrimination. Context-Aware: Detects artifacts based on tissue context (catches 100-800 HU artifacts missed by other methods). Legacy: Simple threshold-based for comparison."
        )
        
        # Add reset button
        col_reset1, col_reset2 = st.columns([1, 2])
        with col_reset1:
            if st.button("🔄 Reset to Defaults", help="Reset all thresholds to default values"):
                if segmentation_method == "Legacy Threshold-Based":
                    reset_thresholds('legacy')
                else:
                    reset_thresholds('russian_doll')
                st.rerun()
        
        if segmentation_method == "Russian Doll with Smart Discrimination (Recommended)":
            st.info("🧠 Uses distance-based analysis for fast bone/artifact discrimination")
            
            # Dark Artifacts Range Slider
            st.markdown("**Dark Artifacts (Beam Hardening)**")
            dark_range = st.slider(
                "Dark Artifact HU Range",
                min_value=int(ThresholdConfig.DARK_ARTIFACTS.min_bound),
                max_value=int(ThresholdConfig.DARK_ARTIFACTS.max_bound),
                value=(int(st.session_state.thresholds['russian_doll']['dark_min']),
                       int(st.session_state.thresholds['russian_doll']['dark_max'])),
                step=int(ThresholdConfig.DARK_ARTIFACTS.step),
                help=ThresholdConfig.DARK_ARTIFACTS.help_text,
                key="dark_range_slider"
            )
            st.session_state.thresholds['russian_doll']['dark_min'] = dark_range[0]
            st.session_state.thresholds['russian_doll']['dark_max'] = dark_high = dark_range[1]
            
            # Bright Artifacts Range Slider (Dynamic based on metal detection)
            st.markdown("**Bright Artifacts (Metal-induced)**")
            
            # Use star profile calculated thresholds if available
            if 'metal_detection_result' in st.session_state and st.session_state.metal_detection_result:
                if 'threshold_evolution' in st.session_state.metal_detection_result:
                    # Get the final threshold from star profile
                    final_threshold = st.session_state.metal_detection_result['threshold_evolution'][-1]
                    # Use bright artifact range that extends up to metal threshold
                    # Everything below metal threshold could be bright artifacts
                    bright_default_min = 800  # Fixed lower bound for bright artifacts
                    bright_default_max = max(3000, int(final_threshold * 0.75))  # Up to 75% of metal threshold
                    st.caption(f"📊 Auto-adjusted to 75% of metal threshold: {final_threshold:.0f} HU (bright max: {bright_default_max} HU)")
                else:
                    bright_default_min = int(st.session_state.thresholds['russian_doll']['bright_min'])
                    bright_default_max = int(st.session_state.thresholds['russian_doll']['bright_max'])
            else:
                bright_default_min = int(st.session_state.thresholds['russian_doll']['bright_min'])
                bright_default_max = int(st.session_state.thresholds['russian_doll']['bright_max'])
            
            bright_range = st.slider(
                "Bright Artifacts HU Range",
                min_value=int(ThresholdConfig.RUSSIAN_DOLL_BRIGHT_ARTIFACTS.min_bound),
                max_value=int(ThresholdConfig.RUSSIAN_DOLL_BRIGHT_ARTIFACTS.max_bound),
                value=(bright_default_min, bright_default_max),
                step=int(ThresholdConfig.RUSSIAN_DOLL_BRIGHT_ARTIFACTS.step),
                help=ThresholdConfig.RUSSIAN_DOLL_BRIGHT_ARTIFACTS.help_text,
                key="bright_range_slider"
            )
            st.session_state.thresholds['russian_doll']['bright_min'] = bright_low = bright_range[0]
            st.session_state.thresholds['russian_doll']['bright_max'] = bright_high = bright_range[1]
            
            # Bone Tissue Range Slider (Independent)
            st.markdown("**Bone Tissue**")
            bone_range = st.slider(
                "Bone HU Range",
                min_value=int(ThresholdConfig.RUSSIAN_DOLL_BONE.min_bound),
                max_value=int(ThresholdConfig.RUSSIAN_DOLL_BONE.max_bound),
                value=(int(st.session_state.thresholds['russian_doll']['bone_min']),
                       int(st.session_state.thresholds['russian_doll']['bone_max'])),
                step=int(ThresholdConfig.RUSSIAN_DOLL_BONE.step),
                help=ThresholdConfig.RUSSIAN_DOLL_BONE.help_text,
                key="bone_range_slider"
            )
            st.session_state.thresholds['russian_doll']['bone_min'] = bone_low = bone_range[0]
            st.session_state.thresholds['russian_doll']['bone_max'] = bone_high = bone_range[1]
            
            # Distance from Metal
            artifact_distance_cm = st.slider(
                "Max Artifact Distance from Metal (cm)",
                min_value=ThresholdConfig.MAX_ARTIFACT_DISTANCE.min_bound,
                max_value=ThresholdConfig.MAX_ARTIFACT_DISTANCE.max_bound,
                value=st.session_state.thresholds['russian_doll']['max_distance'],
                step=ThresholdConfig.MAX_ARTIFACT_DISTANCE.step,
                help=ThresholdConfig.MAX_ARTIFACT_DISTANCE.help_text
            )
            st.session_state.thresholds['russian_doll']['max_distance'] = artifact_distance_cm
            
            # Validation feedback
            is_valid, errors = validate_all_thresholds()
            if not is_valid:
                for error in errors:
                    st.error(error)
            
        elif segmentation_method == "Russian Doll with Enhanced Edge Analysis":
            st.info("🔬 Advanced edge coherence analysis for superior bone/artifact discrimination")
            st.warning("⚠️ This method is slower but provides better accuracy for challenging cases")
            
            # Dark Artifacts Range Slider
            st.markdown("**Dark Artifacts (Beam Hardening)**")
            dark_range_enh = st.slider(
                "Dark Artifact HU Range",
                min_value=int(ThresholdConfig.DARK_ARTIFACTS.min_bound),
                max_value=int(ThresholdConfig.DARK_ARTIFACTS.max_bound),
                value=(int(st.session_state.thresholds['russian_doll']['dark_min']),
                       int(st.session_state.thresholds['russian_doll']['dark_max'])),
                step=int(ThresholdConfig.DARK_ARTIFACTS.step),
                help=ThresholdConfig.DARK_ARTIFACTS.help_text,
                key="dark_range_slider_enh"
            )
            st.session_state.thresholds['russian_doll']['dark_min'] = dark_range_enh[0]
            st.session_state.thresholds['russian_doll']['dark_max'] = dark_high = dark_range_enh[1]
            
            # Bright Artifacts Range Slider (Dynamic based on metal detection)
            st.markdown("**Bright Artifacts (Metal-induced)**")
            
            # Use star profile calculated thresholds if available
            if 'metal_detection_result' in st.session_state and st.session_state.metal_detection_result:
                if 'threshold_evolution' in st.session_state.metal_detection_result:
                    # Get the final threshold from star profile
                    final_threshold = st.session_state.metal_detection_result['threshold_evolution'][-1]
                    # Use bright artifact range that extends up to metal threshold
                    # Everything below metal threshold could be bright artifacts
                    bright_default_min = 800  # Fixed lower bound for bright artifacts
                    bright_default_max = max(3000, int(final_threshold * 0.75))  # Up to 75% of metal threshold
                    st.caption(f"📊 Auto-adjusted to 75% of metal threshold: {final_threshold:.0f} HU (bright max: {bright_default_max} HU)")
                else:
                    bright_default_min = int(st.session_state.thresholds['russian_doll']['bright_min'])
                    bright_default_max = int(st.session_state.thresholds['russian_doll']['bright_max'])
            else:
                bright_default_min = int(st.session_state.thresholds['russian_doll']['bright_min'])
                bright_default_max = int(st.session_state.thresholds['russian_doll']['bright_max'])
            
            bright_range_enh = st.slider(
                "Bright Artifacts HU Range",
                min_value=int(ThresholdConfig.RUSSIAN_DOLL_BRIGHT_ARTIFACTS.min_bound),
                max_value=int(ThresholdConfig.RUSSIAN_DOLL_BRIGHT_ARTIFACTS.max_bound),
                value=(bright_default_min, bright_default_max),
                step=int(ThresholdConfig.RUSSIAN_DOLL_BRIGHT_ARTIFACTS.step),
                help=ThresholdConfig.RUSSIAN_DOLL_BRIGHT_ARTIFACTS.help_text,
                key="bright_range_slider_enh"
            )
            st.session_state.thresholds['russian_doll']['bright_min'] = bright_low = bright_range_enh[0]
            st.session_state.thresholds['russian_doll']['bright_max'] = bright_high = bright_range_enh[1]
            
            # Bone Tissue Range Slider (Independent)
            st.markdown("**Bone Tissue**")
            bone_range_enh = st.slider(
                "Bone HU Range",
                min_value=int(ThresholdConfig.RUSSIAN_DOLL_BONE.min_bound),
                max_value=int(ThresholdConfig.RUSSIAN_DOLL_BONE.max_bound),
                value=(int(st.session_state.thresholds['russian_doll']['bone_min']),
                       int(st.session_state.thresholds['russian_doll']['bone_max'])),
                step=int(ThresholdConfig.RUSSIAN_DOLL_BONE.step),
                help=ThresholdConfig.RUSSIAN_DOLL_BONE.help_text,
                key="bone_range_slider_enh"
            )
            st.session_state.thresholds['russian_doll']['bone_min'] = bone_low = bone_range_enh[0]
            st.session_state.thresholds['russian_doll']['bone_max'] = bone_high = bone_range_enh[1]
            
            # Distance from Metal
            artifact_distance_cm = st.slider(
                "Max Analysis Distance (cm)",
                min_value=ThresholdConfig.MAX_ARTIFACT_DISTANCE.min_bound,
                max_value=ThresholdConfig.MAX_ARTIFACT_DISTANCE.max_bound,
                value=st.session_state.thresholds['russian_doll']['max_distance'],
                step=ThresholdConfig.MAX_ARTIFACT_DISTANCE.step,
                help=ThresholdConfig.MAX_ARTIFACT_DISTANCE.help_text,
                key="enh_dist"
            )
            st.session_state.thresholds['russian_doll']['max_distance'] = artifact_distance_cm
            
            st.markdown("**Enhanced Features:**")
            st.markdown("- Edge coherence analysis (bone has continuous edges)")
            st.markdown("- Gradient jump detection (bone has sharp transitions)")
            st.markdown("- Radial vs tangential feature analysis")
            st.markdown("- 3D structural continuity tracking")
            st.markdown("- Multi-scale edge persistence")
            
        elif segmentation_method == "Russian Doll with Advanced Texture/Gradient Analysis (Best Accuracy)":
            st.info("🔬 Advanced ML-based analysis using texture features (LBP, GLCM) and gradient analysis (LoG)")
            st.success("✨ Most accurate discrimination between bone and bright artifacts")
            
            # Use star profile calculated thresholds if available
            if 'metal_detection_result' in st.session_state and st.session_state.metal_detection_result:
                if 'threshold_evolution' in st.session_state.metal_detection_result:
                    # Get the final threshold from star profile
                    final_threshold = st.session_state.metal_detection_result['threshold_evolution'][-1]
                    st.info(f"📊 Using star profile calculated metal threshold: {final_threshold:.0f} HU")
                    
                    # Calculate bright artifact range based on 75% of metal threshold
                    default_bright_min = int(final_threshold * 0.75)  # 75% of metal threshold
                    default_bright_max = int(final_threshold - 500)   # Just below metal
                else:
                    default_bright_min = 800
                    default_bright_max = 3500
            else:
                default_bright_min = 800
                default_bright_max = 3500
            
            # Dark Artifacts Range Slider
            st.markdown("**Dark Artifacts (Beam Hardening)**")
            dark_range_adv = st.slider(
                "Dark Artifact HU Range",
                min_value=int(ThresholdConfig.DARK_ARTIFACTS.min_bound),
                max_value=int(ThresholdConfig.DARK_ARTIFACTS.max_bound),
                value=(int(st.session_state.thresholds.get('advanced', {}).get('dark_min', -1024)),
                       int(st.session_state.thresholds.get('advanced', {}).get('dark_max', -150))),
                step=int(ThresholdConfig.DARK_ARTIFACTS.step),
                help=ThresholdConfig.DARK_ARTIFACTS.help_text,
                key="dark_range_slider_adv"
            )
            
            # Initialize advanced thresholds if not exists
            if 'advanced' not in st.session_state.thresholds:
                st.session_state.thresholds['advanced'] = {}
            
            st.session_state.thresholds['advanced']['dark_min'] = dark_low = dark_range_adv[0]
            st.session_state.thresholds['advanced']['dark_max'] = dark_high = dark_range_adv[1]
            
            # Bright Artifacts Range with calculated defaults
            st.markdown("**Bright Artifacts (Metal-induced)**")
            bright_range_adv = st.slider(
                "Bright Artifacts HU Range (Auto-adjusted from star profile)",
                min_value=int(ThresholdConfig.RUSSIAN_DOLL_BRIGHT_ARTIFACTS.min_bound),
                max_value=int(ThresholdConfig.RUSSIAN_DOLL_BRIGHT_ARTIFACTS.max_bound),
                value=(int(st.session_state.thresholds.get('advanced', {}).get('bright_min', default_bright_min)),
                       int(st.session_state.thresholds.get('advanced', {}).get('bright_max', default_bright_max))),
                step=int(ThresholdConfig.RUSSIAN_DOLL_BRIGHT_ARTIFACTS.step),
                help="Range auto-adjusted based on detected metal threshold using 75% rule",
                key="bright_range_slider_adv"
            )
            st.session_state.thresholds['advanced']['bright_min'] = bright_low = bright_range_adv[0]
            st.session_state.thresholds['advanced']['bright_max'] = bright_high = bright_range_adv[1]
            
            # Bone Tissue Range (Independent)
            st.markdown("**Bone Tissue**")
            bone_range_adv = st.slider(
                "Bone HU Range",
                min_value=int(ThresholdConfig.RUSSIAN_DOLL_BONE.min_bound),
                max_value=int(ThresholdConfig.RUSSIAN_DOLL_BONE.max_bound),
                value=(int(st.session_state.thresholds.get('advanced', {}).get('bone_min', 300)),
                       int(st.session_state.thresholds.get('advanced', {}).get('bone_max', 1200))),
                step=int(ThresholdConfig.RUSSIAN_DOLL_BONE.step),
                help=ThresholdConfig.RUSSIAN_DOLL_BONE.help_text,
                key="bone_range_slider_adv"
            )
            st.session_state.thresholds['advanced']['bone_min'] = bone_low = bone_range_adv[0]
            st.session_state.thresholds['advanced']['bone_max'] = bone_high = bone_range_adv[1]
            
            # Distance from Metal
            artifact_distance_cm = st.slider(
                "Max Artifact Distance from Metal (cm)",
                min_value=ThresholdConfig.MAX_ARTIFACT_DISTANCE.min_bound,
                max_value=ThresholdConfig.MAX_ARTIFACT_DISTANCE.max_bound,
                value=st.session_state.thresholds.get('advanced', {}).get('max_distance', 10.0),
                step=ThresholdConfig.MAX_ARTIFACT_DISTANCE.step,
                help="Distance weighting for artifact probability",
                key="adv_dist"
            )
            st.session_state.thresholds['advanced']['max_distance'] = artifact_distance_cm
            
            st.markdown("**Advanced Features:**")
            st.markdown("- 🎨 **Texture Analysis**: LBP patterns, GLCM features, local variance")
            st.markdown("- 📈 **Gradient Analysis**: Laplacian of Gaussian, gradient direction variance")
            st.markdown("- 🧠 **Structure Tensor**: Coherence and anisotropy measures")
            st.markdown("- 🎯 **Confidence Scoring**: Per-voxel classification confidence")
            st.markdown("- 🔬 **Post-processing**: Morphological refinement and connected components")
            
        else:
            # Legacy parameters with sliders
            st.markdown("**Legacy Threshold-Based Method**")
            
            # Bright Artifacts Range
            st.markdown("**Bright Artifacts**")
            legacy_bright_range = st.slider(
                "Bright Artifact HU Range",
                min_value=int(ThresholdConfig.LEGACY_BRIGHT_ARTIFACTS.min_bound),
                max_value=int(ThresholdConfig.LEGACY_BRIGHT_ARTIFACTS.max_bound),
                value=(int(st.session_state.thresholds['legacy']['bright_min']),
                       int(st.session_state.thresholds['legacy']['bright_max'])),
                step=int(ThresholdConfig.LEGACY_BRIGHT_ARTIFACTS.step),
                help=ThresholdConfig.LEGACY_BRIGHT_ARTIFACTS.help_text,
                key="legacy_bright_slider"
            )
            bright_low = legacy_bright_range[0]
            bright_high = legacy_bright_range[1]
            st.session_state.thresholds['legacy']['bright_min'] = bright_low
            st.session_state.thresholds['legacy']['bright_max'] = bright_high
            
            # Dark Artifacts Threshold
            st.markdown("**Dark Artifacts**")
            dark_high = st.slider(
                "Dark Artifact Maximum HU",
                min_value=int(ThresholdConfig.LEGACY_DARK_THRESHOLD.min_bound),
                max_value=int(ThresholdConfig.LEGACY_DARK_THRESHOLD.max_bound),
                value=int(st.session_state.thresholds['legacy']['dark_max']),
                step=int(ThresholdConfig.LEGACY_DARK_THRESHOLD.step),
                help=ThresholdConfig.LEGACY_DARK_THRESHOLD.help_text,
                key="legacy_dark_slider"
            )
            st.session_state.thresholds['legacy']['dark_max'] = dark_high
            # For legacy mode, dark artifacts go from -1024 to dark_high
            dark_low = -1024
            
            # Bone Tissue Range
            st.markdown("**Bone Tissue**")
            bone_range = st.slider(
                "Bone HU Range",
                min_value=int(ThresholdConfig.BONE_TISSUE.min_bound),
                max_value=int(ThresholdConfig.BONE_TISSUE.max_bound),
                value=(int(st.session_state.thresholds['legacy']['bone_min']),
                       int(st.session_state.thresholds['legacy']['bone_max'])),
                step=int(ThresholdConfig.BONE_TISSUE.step),
                help=ThresholdConfig.BONE_TISSUE.help_text,
                key="legacy_bone_slider"
            )
            bone_low = bone_range[0]
            bone_high = bone_range[1]
            st.session_state.thresholds['legacy']['bone_min'] = bone_low
            st.session_state.thresholds['legacy']['bone_max'] = bone_high
            
            artifact_distance_cm = 10.0  # Default for compatibility
    
    st.markdown("---")
    
    # Contour Display Settings
    st.subheader("Contour Display")
    
    # Initialize contour visibility state
    if 'contour_visibility' not in st.session_state:
        st.session_state.contour_visibility = {
            'metal': True,
            'bright_artifacts': True,
            'dark_artifacts': True,
            'bone': True
        }
    
    # Initialize custom names
    if 'contour_names' not in st.session_state:
        st.session_state.contour_names = {
            'metal': 'Metal Implant',
            'bright_artifacts': 'Bright Artifacts',
            'dark_artifacts': 'Dark Artifacts',
            'bone': 'Bone'
        }
    
    # Contour visibility toggles
    st.markdown("**Visibility**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.contour_visibility['metal'] = st.checkbox(
            "Metal", value=st.session_state.contour_visibility['metal'], key="vis_metal")
        st.session_state.contour_visibility['bright_artifacts'] = st.checkbox(
            "Bright Artifacts (Legacy)", value=st.session_state.contour_visibility['bright_artifacts'], key="vis_bright")
        
        # Contextual bright artifact controls
        if 'bright_artifact_bone' in st.session_state.masks:
            st.session_state.contour_visibility['bright_artifact_bone'] = st.checkbox(
                "Bright Artifacts → Bone", value=st.session_state.contour_visibility.get('bright_artifact_bone', True), key="vis_bright_bone")
        if 'bright_artifact_tissue' in st.session_state.masks:
            st.session_state.contour_visibility['bright_artifact_tissue'] = st.checkbox(
                "Bright Artifacts → Tissue", value=st.session_state.contour_visibility.get('bright_artifact_tissue', True), key="vis_bright_tissue")
    
    with col2:
        st.session_state.contour_visibility['dark_artifacts'] = st.checkbox(
            "Dark Artifacts", value=st.session_state.contour_visibility['dark_artifacts'], key="vis_dark")
        st.session_state.contour_visibility['bone'] = st.checkbox(
            "Bone", value=st.session_state.contour_visibility['bone'], key="vis_bone")
        
        # Mixed contextual artifacts
        if 'bright_artifact_mixed' in st.session_state.masks:
            st.session_state.contour_visibility['bright_artifact_mixed'] = st.checkbox(
                "Bright Artifacts → Mixed", value=st.session_state.contour_visibility.get('bright_artifact_mixed', True), key="vis_bright_mixed")
    
    # Contour name editing
    if st.checkbox("Edit Contour Names"):
        st.markdown("**Custom Names**")
        for key in ['metal', 'bright_artifacts', 'dark_artifacts', 'bone']:
            st.session_state.contour_names[key] = st.text_input(
                f"{key.replace('_', ' ').title()} Name:", 
                value=st.session_state.contour_names[key],
                key=f"name_{key}"
            )
    
    st.markdown("---")
    
    # Export options
    st.subheader("Export Options")
    output_format = st.selectbox(
        "Output Format",
        ["NIFTI (.nii.gz)", "Multi-label NIFTI", "Separate Binary Masks", "DICOM RT Structure"],
        help="Choose export format for contours"
    )
    
    if st.button("Export All Contours", disabled=not st.session_state.masks):
        if st.session_state.masks:
            with st.spinner("Exporting contours..."):
                patient_name = selected_patient.replace(" ", "_")
                output_dir = Path("../output") / patient_name
                if not output_dir.parent.exists():
                    output_dir = Path("output") / patient_name
                output_dir.mkdir(exist_ok=True, parents=True)
                
                if output_format == "DICOM RT Structure":
                    # DICOM export
                    from dicom_export import create_rtstruct_from_masks, save_rtstruct
                    
                    # Prepare patient info
                    patient_info = {
                        'PatientName': patient_name,
                        'PatientID': patient_name.split('_')[0] if '_' in patient_name else patient_name
                    }
                    
                    # Create RT Structure Set
                    rtstruct = create_rtstruct_from_masks(
                        st.session_state.masks,
                        st.session_state.ct_metadata,
                        patient_info,
                        st.session_state.contour_names
                    )
                    
                    # Save RT Structure
                    output_path = output_dir / f"{patient_name}_RTSTRUCT.dcm"
                    save_rtstruct(rtstruct, str(output_path))
                    st.success(f"Exported DICOM RT Structure to {output_path}")
                
                else:
                    # NIFTI export
                    if st.session_state.affine is not None:
                        output_prefix = str(output_dir / patient_name)
                        save_all_contours_as_nifti(
                            st.session_state.masks, 
                            st.session_state.affine,
                            output_prefix
                        )
                        st.success(f"Exported NIFTI contours to {output_dir}")
                    else:
                        st.error("NIFTI export requires affine transformation matrix")

# Main content area
if st.session_state.ct_volume is not None:
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Single Slice Analysis", "Multi-Slice View", 
                                       "Metal Detection Details", "Statistics", "Threshold Preview"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("CT Slice Viewer")
            
            # Slice selector
            max_slice = st.session_state.ct_volume.shape[0] - 1
            current_slice = st.slider(
                "Select Slice",
                min_value=0,
                max_value=max_slice,
                value=st.session_state.current_slice,
                format="Slice %d"
            )
            st.session_state.current_slice = current_slice
            
            # Get current slice data
            ct_slice = st.session_state.ct_volume[current_slice]
            
            # Display slice info
            z_pos = st.session_state.ct_metadata['slice_z_positions'][current_slice]
            st.info(f"Slice {current_slice + 1} of {max_slice + 1} | Z position: {z_pos:.2f} mm")
            
            # Analysis buttons
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🎯 Detect Metal Automatically", type="primary"):
                    with st.spinner("Detecting metal implant..."):
                        start_time = time.time()
                        if detection_method == "3D Adaptive + Star Algorithm (Recommended)":
                            # 3D adaptive method with coronal/sagittal analysis + star profiles
                            detector = MetalDetector(MetalDetectionMethod.ADAPTIVE_3D)
                            # Fix spacing to be positive (some DICOM files have negative z-spacing)
                            spacing = np.abs(st.session_state.ct_metadata['spacing'])
                            result = detector.detect(
                                st.session_state.ct_volume,
                                spacing,
                                fw_percentage=fw_percentage,
                                margin_cm=margin_cm,
                                intensity_percentile=intensity_percentile
                            )
                            
                            # roi_bounds is already in the result as roi_bounds
                        else:
                            # Legacy method with initial threshold
                            # Fix spacing to be positive
                            spacing = np.abs(st.session_state.ct_metadata['spacing'])
                            result = detect_metal_volume(
                                st.session_state.ct_volume,
                                spacing,
                                margin_cm=roi_margin_cm,
                                fw_percentage=fw_percentage,
                                min_metal_hu=min_metal_hu,
                                dilation_iterations=dilation_iterations
                            )
                        
                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        if result['mask'] is not None and np.any(result['mask']):
                            st.session_state.metal_detection_result = result
                            st.session_state.masks['metal'] = result['mask']
                            
                            metal_count = np.sum(result['mask'])
                            if detection_method == "3D Adaptive + Star Algorithm (Recommended)":
                                st.success(f"3D adaptive + star algorithm detection complete! Found {metal_count:,} metal voxels")
                                if 'analysis' in result and result['analysis']:
                                    thresh = result['analysis']['threshold_used']
                                    extent = result['analysis']['extent_voxels']
                                    st.info(f"Auto-detected threshold: {thresh:.0f} HU")
                                    st.info(f"3D extent: {extent['z']}×{extent['y']}×{extent['x']} voxels")
                                if 'individual_regions' in result:
                                    total_regions = sum(len(regions) for regions in result['individual_regions'].values())
                                    st.info(f"Created {total_regions} focused ROI regions across {len(result['individual_regions'])} slices")
                            else:
                                st.success(f"Legacy detection complete! Found {metal_count:,} metal voxels")
                            st.info(f"⏱️ Metal detection took {elapsed_time:.2f} seconds.")
                        else:
                            st.error("No metal implant detected")
            
            with col_btn2:
                if st.button("🔍 Segment All Artifacts", 
                           disabled='metal' not in st.session_state.masks):
                    if 'metal' in st.session_state.masks:
                        with st.spinner("Segmenting artifacts..."):
                            start_time = time.time()
                            metal_mask = st.session_state.masks['metal']
                            roi_bounds = st.session_state.metal_detection_result['roi_bounds']
                            
                            # Get threshold values from session state based on segmentation method
                            if segmentation_method.startswith("Russian Doll"):
                                # Advanced/Russian Doll methods
                                dark_low = st.session_state.thresholds.get('advanced', {}).get('dark_min', -1024)
                                dark_high = st.session_state.thresholds.get('advanced', {}).get('dark_max', -150)
                                bright_low = st.session_state.thresholds.get('advanced', {}).get('bright_min', 800)
                                bright_high = st.session_state.thresholds.get('advanced', {}).get('bright_max', 3500)
                                bone_low = st.session_state.thresholds.get('advanced', {}).get('bone_min', 150)
                                bone_high = st.session_state.thresholds.get('advanced', {}).get('bone_max', 1500)
                                artifact_distance_cm = st.session_state.thresholds.get('advanced', {}).get('max_distance', 10.0)
                                
                                print(f"Using thresholds - Dark: [{dark_low}, {dark_high}], Bright: [{bright_low}, {bright_high}], Bone: [{bone_low}, {bone_high}]")
                            else:
                                # Legacy method
                                dark_low = -1024  # Legacy uses fixed lower bound
                                dark_high = st.session_state.thresholds.get('legacy', {}).get('dark_max', -150)
                                bright_low = st.session_state.thresholds.get('legacy', {}).get('bright_min', 500)
                                bright_high = st.session_state.thresholds.get('legacy', {}).get('bright_max', 3000)
                                bone_low = st.session_state.thresholds.get('legacy', {}).get('bone_min', 150)
                                bone_high = st.session_state.thresholds.get('legacy', {}).get('bone_max', 1500)
                                artifact_distance_cm = 10.0
                            
                            if segmentation_method == "Russian Doll with Smart Discrimination (Recommended)":
                                # Use the smart Russian doll segmentation
                                with st.spinner("Running smart bone/artifact discrimination..."):
                                    # Fix spacing to be positive
                                    spacing = np.abs(st.session_state.ct_metadata['spacing'])
                                    segmentation_result = create_russian_doll_segmentation(
                                        st.session_state.ct_volume,
                                        metal_mask,
                                        spacing,
                                        roi_bounds,
                                        dark_threshold_low=dark_low,
                                        dark_threshold_high=dark_high,
                                        bone_threshold_low=bone_low,
                                        bone_threshold_high=bone_high,
                                        bright_threshold_low=bright_low,
                                        bright_threshold_high=bright_high,
                                        bright_artifact_max_distance_cm=artifact_distance_cm,
                                        use_fast_mode=True,
                                        use_enhanced_mode=False
                                    )
                                
                                # Update masks - including contextual bright artifact masks
                                if segmentation_result:
                                    # Store all available masks from segmentation result
                                    mask_names = ['dark_artifacts', 'bone', 'bright_artifacts', 
                                                 'bright_artifact_bone', 'bright_artifact_tissue', 'bright_artifact_mixed',
                                                 'bright_artifacts_mild', 'bright_artifacts_moderate', 'bright_artifacts_severe']
                                    for mask_name in mask_names:
                                        mask = segmentation_result.get(mask_name)
                                        if mask is not None:
                                            st.session_state.masks[mask_name] = mask.astype(bool) if hasattr(mask, 'astype') else mask
                                    
                            elif segmentation_method == "Russian Doll with Advanced Texture/Gradient Analysis (Best Accuracy)":
                                # Use advanced texture/gradient-based discrimination
                                with st.spinner("Running advanced texture/gradient analysis... This may take a moment."):
                                    segmentation_result = create_russian_doll_segmentation(
                                        st.session_state.ct_volume,
                                        metal_mask,
                                        st.session_state.ct_metadata['spacing'],
                                        roi_bounds,
                                        dark_threshold_low=dark_low,
                                        dark_threshold_high=dark_high,
                                        bone_threshold_low=bone_low,
                                        bone_threshold_high=bone_high,
                                        bright_threshold_low=bright_low,
                                        bright_threshold_high=bright_high,
                                        bright_artifact_max_distance_cm=artifact_distance_cm,
                                        use_fast_mode=False,
                                        use_enhanced_mode=False,
                                        use_advanced_mode=True
                                    )
                                
                                # Update masks with advanced results - including contextual bright artifact masks
                                if segmentation_result:
                                    mask_names = ['dark_artifacts', 'bone', 'bright_artifacts', 
                                                 'bright_artifact_bone', 'bright_artifact_tissue', 'bright_artifact_mixed',
                                                 'bright_artifacts_mild', 'bright_artifacts_moderate', 'bright_artifacts_severe']
                                    for mask_name in mask_names:
                                        mask = segmentation_result.get(mask_name)
                                        if mask is not None:
                                            st.session_state.masks[mask_name] = mask.astype(bool) if hasattr(mask, 'astype') else mask
                                    
                                    # Store confidence map for visualization
                                    if 'confidence_map' in segmentation_result:
                                        st.session_state.segmentation_info = {
                                            'confidence_map': segmentation_result['confidence_map'],
                                            'method': 'advanced_texture_gradient'
                                        }
                                        
                                        # Show confidence statistics
                                        conf_map = segmentation_result['confidence_map']
                                        if np.any(conf_map > 0):
                                            avg_conf = np.mean(conf_map[conf_map > 0])
                                            high_conf = np.sum(conf_map > 0.7) / np.sum(conf_map > 0)
                                            st.success(f"✅ Advanced discrimination complete! Avg confidence: {avg_conf:.1%}, High confidence: {high_conf:.1%}")
                                
                            elif segmentation_method == "Russian Doll with Enhanced Edge Analysis":
                                # Use enhanced edge-based discrimination
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                def update_progress(progress, message):
                                    progress_bar.progress(progress)
                                    status_text.text(message)
                                
                                try:
                                    segmentation_result = create_russian_doll_segmentation(
                                        st.session_state.ct_volume,
                                        metal_mask,
                                        st.session_state.ct_metadata['spacing'],
                                        roi_bounds,
                                        dark_threshold_low=dark_low,
                                        dark_threshold_high=dark_high,
                                        bone_threshold_low=bone_low,
                                        bone_threshold_high=bone_high,
                                        bright_threshold_low=bright_low,
                                        bright_threshold_high=bright_high,
                                        bright_artifact_max_distance_cm=artifact_distance_cm,
                                        use_fast_mode=False,
                                        use_enhanced_mode=True,
                                        progress_callback=update_progress
                                    )
                                except Exception as e:
                                    progress_bar.empty()
                                    status_text.empty()
                                    st.error(f"Enhanced segmentation failed: {str(e)}")
                                    st.exception(e)
                                    segmentation_result = None
                                
                                # Clear progress indicators
                                progress_bar.empty()
                                status_text.empty()
                                
                                if segmentation_result:
                                    # Update masks for both Russian doll methods - including contextual bright artifact masks
                                    mask_names = ['dark_artifacts', 'bone', 'bright_artifacts', 
                                                 'bright_artifact_bone', 'bright_artifact_tissue', 'bright_artifact_mixed',
                                                 'bright_artifacts_mild', 'bright_artifacts_moderate', 'bright_artifacts_severe']
                                    for mask_name in mask_names:
                                        mask = segmentation_result.get(mask_name)
                                        if mask is not None:
                                            # Ensure mask is boolean
                                            st.session_state.masks[mask_name] = mask.astype(bool) if hasattr(mask, 'astype') else mask
                                    
                                    
                            elif segmentation_method == "Context-Aware Artifact Detection (Experimental)":
                                # Use context-aware bright artifact detection
                                with st.spinner("Running context-aware artifact detection... Analyzing tissue context..."):
                                    segmentation_result = create_sequential_masks(
                                        st.session_state.ct_volume,
                                        metal_mask,
                                        st.session_state.ct_metadata['spacing'],
                                        discrimination_method='context_aware',
                                        roi_bounds=roi_bounds,
                                        dark_range=(dark_low, dark_high),
                                        bone_range=(bone_low, bone_high),
                                        elevation_threshold=1.5,  # 50% above expected
                                        min_elevation_hu=100,     # Minimum elevation to consider artifact
                                        max_elevation_hu=2000,    # Severe artifact threshold
                                        debug=False
                                    )
                                
                                # Update masks with context-aware results
                                if segmentation_result:
                                    mask_names = ['dark_artifacts', 'bone', 'bright_artifacts', 
                                                 'bright_artifact_bone', 'bright_artifact_tissue', 'bright_artifact_mixed',
                                                 'bright_artifacts_mild', 'bright_artifacts_moderate', 'bright_artifacts_severe']
                                    for mask_name in mask_names:
                                        mask = segmentation_result.get(mask_name)
                                        if mask is not None:
                                            st.session_state.masks[mask_name] = mask.astype(bool) if hasattr(mask, 'astype') else mask
                                    
                                    st.success("Context-aware artifact detection complete!")
                                    
                                    # Show context-aware statistics
                                    if 'bright_artifacts_mild' in st.session_state.masks:
                                        mild_voxels = np.sum(st.session_state.masks['bright_artifacts_mild'])
                                        moderate_voxels = np.sum(st.session_state.masks['bright_artifacts_moderate']) 
                                        severe_voxels = np.sum(st.session_state.masks['bright_artifacts_severe'])
                                        total_context = mild_voxels + moderate_voxels + severe_voxels
                                        
                                        if total_context > 0:
                                            st.info(f"Context-aware detection found {total_context:,} bright artifacts:")
                                            st.info(f"• Mild: {mild_voxels:,} ({100*mild_voxels/total_context:.1f}%)")
                                            st.info(f"• Moderate: {moderate_voxels:,} ({100*moderate_voxels/total_context:.1f}%)")  
                                            st.info(f"• Severe: {severe_voxels:,} ({100*severe_voxels/total_context:.1f}%)")
                                else:
                                    st.warning("Context-aware segmentation returned no results")
                            
                            # Common code for both Russian doll methods (moved outside enhanced-only block)
                            if segmentation_result and segmentation_method.startswith("Russian Doll"):
                                # Store additional results for analysis
                                if 'segmentation_info' not in st.session_state:
                                    st.session_state.segmentation_info = {}
                                st.session_state.segmentation_info['confidence_map'] = segmentation_result.get('confidence_map')
                                st.session_state.segmentation_info['distance_map'] = segmentation_result.get('distance_map')
                                
                                if segmentation_method == "Russian Doll with Enhanced Edge Analysis":
                                    st.success("Enhanced edge-based segmentation complete!")
                                else:
                                    st.success("Smart artifact segmentation complete!")
                                
                                # Show statistics
                                bone_voxels = np.sum(st.session_state.masks['bone']) if 'bone' in st.session_state.masks else 0
                                bright_voxels = np.sum(st.session_state.masks['bright_artifacts']) if 'bright_artifacts' in st.session_state.masks else 0
                                st.info(f"Discriminated {bone_voxels:,} bone voxels from {bright_voxels:,} bright artifact voxels")
                                
                            else:
                                # Legacy method
                                bright_mask = create_bright_artifact_mask(
                                    st.session_state.ct_volume,
                                    metal_mask,
                                    roi_bounds,
                                    bright_low,
                                    bright_high
                                )
                                
                                dark_mask = create_dark_artifact_mask(
                                    st.session_state.ct_volume,
                                    metal_mask,
                                    roi_bounds,
                                    dark_high
                                )
                                
                                bone_mask = create_bone_mask(
                                    st.session_state.ct_volume,
                                    metal_mask,
                                    bright_mask,
                                    dark_mask,
                                    roi_bounds,
                                    bone_low,
                                    bone_high
                                )
                                
                                # Refine masks
                                st.session_state.masks['bright_artifacts'] = refine_mask(bright_mask)
                                st.session_state.masks['dark_artifacts'] = refine_mask(dark_mask)
                                st.session_state.masks['bone'] = refine_mask(bone_mask)
                                
                                st.success("Legacy artifact segmentation complete!")

                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            st.info(f"⏱️ Segmentation took {elapsed_time:.2f} seconds.")
            
            # Display visualization
            if st.session_state.masks:
                roi_bounds = None
                if st.session_state.metal_detection_result:
                    roi_bounds = st.session_state.metal_detection_result['roi_bounds']
                    # Convert 3D bounds to 2D for current slice
                    roi_bounds_2d = {
                        'y_min': roi_bounds['y_min'],
                        'y_max': roi_bounds['y_max'], 
                        'x_min': roi_bounds['x_min'],
                        'x_max': roi_bounds['x_max']
                    }
                
                # Create masks dict for current slice only - respect visibility settings
                slice_masks = {}
                for name, mask in st.session_state.masks.items():
                    if isinstance(mask, np.ndarray) and mask.ndim == 3:
                        # Only include if visibility is enabled
                        if st.session_state.contour_visibility.get(name, True):
                            slice_masks[name] = mask[current_slice]
                
                # Convert roi_bounds_2d dict to tuple format expected by create_overlay_image
                # Only show ROI if current slice is in valid_roi_slices
                roi_boundaries_tuple = None
                if roi_bounds_2d:
                    valid_roi_slices = st.session_state.metal_detection_result.get('valid_roi_slices', None)
                    if valid_roi_slices is None or current_slice in valid_roi_slices:
                        roi_boundaries_tuple = (
                            roi_bounds_2d['y_min'],
                            roi_bounds_2d['y_max'],
                            roi_bounds_2d['x_min'],
                            roi_bounds_2d['x_max']
                        )
                
                # Get individual regions for this slice if using 3D adaptive detection
                current_slice_regions = None
                if (st.session_state.metal_detection_result and 
                    'individual_regions' in st.session_state.metal_detection_result and
                    current_slice in st.session_state.metal_detection_result['individual_regions']):
                    current_slice_regions = st.session_state.metal_detection_result['individual_regions'][current_slice]
                
                fig = create_overlay_image(
                    ct_slice,
                    slice_masks,
                    roi_boundaries_tuple,
                    current_slice,
                    individual_regions=current_slice_regions,
                    custom_names=st.session_state.contour_names,
                    spacing=st.session_state.ct_metadata['spacing']
                )
                st.pyplot(fig)
                plt.close()
            else:
                # Show simple preview
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(ct_slice, cmap='gray', vmin=-150, vmax=250)
                ax.set_title(f"CT Slice {current_slice}")
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
        
        with col2:
            st.subheader("Analysis Results")
            
            if st.session_state.metal_detection_result:
                # Show detected thresholds
                st.markdown("**Adaptive Thresholds**")
                threshold = st.session_state.metal_detection_result.get('threshold', None)
                
                if threshold:
                    st.text(f"Metal: >{threshold:.0f} HU")
                else:
                    st.text("Metal: Default thresholds")
                
                st.text(f"Bright: {bright_low} - {bright_high} HU")
                st.text(f"Dark: < {dark_high} HU")
                st.text(f"Bone: {bone_low} - {bone_high} HU")
            
            # Display pixel counts
            if st.session_state.masks:
                st.markdown("**Segmentation Statistics**")
                for mask_name, mask in st.session_state.masks.items():
                    if isinstance(mask, np.ndarray):
                        if mask.ndim == 3:
                            count = np.sum(mask[current_slice])
                            total = np.sum(mask)
                            st.text(f"{mask_name}: {count:,} pixels (slice) / {total:,} voxels (total)")
                        else:
                            count = np.sum(mask)
                            st.text(f"{mask_name}: {count:,} pixels")
            
            # Show histograms
            if st.session_state.masks and st.checkbox("Show Intensity Histograms"):
                st.markdown("**Intensity Distributions**")
                
                colors = {
                    'metal': 'red',
                    'bright_artifacts': 'yellow',
                    'dark_artifacts': 'magenta',
                    'bone': 'blue'
                }
                
                for mask_name, mask in st.session_state.masks.items():
                    if mask_name in colors and isinstance(mask, np.ndarray):
                        if mask.ndim == 3:
                            mask_slice = mask[current_slice]
                        else:
                            mask_slice = mask
                        
                        hu_values = ct_slice[mask_slice]
                        if hu_values.size > 0:
                            fig = create_histogram(
                                hu_values,
                                mask_name.replace('_', ' ').title(),
                                colors[mask_name]
                            )
                            if fig:
                                st.pyplot(fig)
                                plt.close()
    
    with tab2:
        st.subheader("Multi-Slice Overview")
        
        if st.session_state.masks:
            # Select slices to display
            n_slices = st.slider("Number of slices to display", 4, 16, 8)
            
            # Get slice indices
            if st.session_state.metal_detection_result:
                roi_bounds = st.session_state.metal_detection_result['roi_bounds']
                z_min, z_max = roi_bounds['z_min'], roi_bounds['z_max']
                slice_indices = np.linspace(z_min, z_max-1, n_slices, dtype=int)
            else:
                slice_indices = np.linspace(0, st.session_state.ct_volume.shape[0]-1, 
                                          n_slices, dtype=int)
            
            # Create multi-slice view
            individual_regions = None
            if (st.session_state.metal_detection_result and 
                'individual_regions' in st.session_state.metal_detection_result):
                individual_regions = st.session_state.metal_detection_result['individual_regions']
            
            # Filter masks based on visibility settings
            visible_masks = {}
            for name, mask in st.session_state.masks.items():
                if st.session_state.contour_visibility.get(name, True):
                    visible_masks[name] = mask
            
            # Get valid ROI slices if available
            valid_roi_slices = None
            if st.session_state.metal_detection_result:
                valid_roi_slices = st.session_state.metal_detection_result.get('valid_roi_slices', None)
            
            fig = create_multi_slice_view(
                st.session_state.ct_volume,
                visible_masks,
                slice_indices,
                roi_bounds if st.session_state.metal_detection_result else None,
                individual_regions=individual_regions,
                valid_roi_slices=valid_roi_slices
            )
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Run metal detection and segmentation first to view multiple slices")
    
    with tab3:
        st.subheader("Metal Detection Analysis")
        
        if st.session_state.metal_detection_result:
            result = st.session_state.metal_detection_result
            
            # Show star profile visualization
            if st.checkbox("Show Star Profile Analysis"):
                current_slice = st.session_state.current_slice
                
                # Check if we have individual regions for this slice
                if ('individual_regions' in result and 
                    current_slice in result['individual_regions'] and
                    result['individual_regions'][current_slice]):
                    
                    # Use the first component's bounds for this slice
                    component = result['individual_regions'][current_slice][0]
                    roi_bounds_2d = {
                        'y_min': component['y_min'],
                        'y_max': component['y_max'],
                        'x_min': component['x_min'],
                        'x_max': component['x_max']
                    }
                    center_y = component['center_y']
                    center_x = component['center_x']
                    
                    # Generate star profiles for visualization
                    from core.metal_detection import get_star_profile_lines
                    
                    profiles = get_star_profile_lines(
                        st.session_state.ct_volume[current_slice],
                        center_y,
                        center_x,
                        roi_bounds_2d
                    )
                    
                    # Get threshold for visualization
                    threshold = result.get('threshold', 2500)
                    thresholds = (threshold * 0.75, threshold)  # Use 75% and 100% for visualization
                    
                    fig = visualize_star_profiles(
                        st.session_state.ct_volume[current_slice],
                        profiles,
                        (center_y, center_x),
                        roi_bounds_2d,
                        thresholds
                    )
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Star profile visualization requires individual region detection")
            
            # Show threshold evolution
            if st.checkbox("Show Threshold Evolution Across Slices"):
                threshold_data = result.get('threshold_evolution', result.get('slice_thresholds', []))
                if threshold_data:
                    # Create simple threshold display
                    st.metric("Detection Threshold", f"{result.get('threshold', 'N/A'):.0f} HU")
                else:
                    st.info("No threshold evolution data available")
            
            # Detection summary
            st.markdown("**Detection Summary**")
            center = result['center_coords']
            st.text(f"Metal center: ({center[0]}, {center[1]}, {center[2]})")
            st.text(f"ROI: Z [{roi_bounds['z_min']}-{roi_bounds['z_max']}], "
                   f"Y [{roi_bounds['y_min']}-{roi_bounds['y_max']}], "
                   f"X [{roi_bounds['x_min']}-{roi_bounds['x_max']}]")
        else:
            st.info("Run automatic metal detection to see detailed analysis")
    
    with tab4:
        st.subheader("Volume Statistics")
        
        if st.session_state.masks:
            # Calculate volumes
            spacing = st.session_state.ct_metadata['spacing']
            voxel_volume = np.prod(spacing) / 1000  # Convert to cm³
            
            st.markdown("**Tissue Volumes**")
            
            data = []
            for mask_name, mask in st.session_state.masks.items():
                if isinstance(mask, np.ndarray):
                    voxel_count = np.sum(mask)
                    volume_cm3 = voxel_count * voxel_volume
                    
                    data.append({
                        'Tissue Type': mask_name.replace('_', ' ').title(),
                        'Voxel Count': f"{voxel_count:,}",
                        'Volume (cm³)': f"{volume_cm3:.2f}",
                        'Percentage': f"{100 * voxel_count / mask.size:.2f}%"
                    })
            
            if data:
                import pandas as pd
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
            
            # Show HU statistics
            if st.checkbox("Show HU Statistics by Region"):
                st.markdown("**Hounsfield Unit Statistics**")
                
                for mask_name, mask in st.session_state.masks.items():
                    if isinstance(mask, np.ndarray):
                        hu_values = st.session_state.ct_volume[mask]
                        if hu_values.size > 0:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(f"{mask_name} Mean", f"{np.mean(hu_values):.0f} HU")
                            with col2:
                                st.metric("Std Dev", f"{np.std(hu_values):.0f}")
                            with col3:
                                st.metric("Min", f"{np.min(hu_values):.0f} HU")
                            with col4:
                                st.metric("Max", f"{np.max(hu_values):.0f} HU")
            
            # Show discrimination confidence if available
            if ('segmentation_info' in st.session_state and 
                'confidence_map' in st.session_state.segmentation_info and
                st.checkbox("Show Discrimination Confidence")):
                st.markdown("**Bone vs Artifact Discrimination Confidence**")
                
                confidence_map = st.session_state.segmentation_info['confidence_map']
                confident_voxels = confidence_map > 0
                
                if np.any(confident_voxels):
                    avg_confidence = np.mean(confidence_map[confident_voxels])
                    high_confidence = np.sum(confidence_map > 0.8)
                    medium_confidence = np.sum((confidence_map > 0.5) & (confidence_map <= 0.8))
                    low_confidence = np.sum((confidence_map > 0) & (confidence_map <= 0.5))
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                    with col2:
                        st.metric("High Confidence", f"{high_confidence:,} voxels")
                    with col3:
                        st.metric("Low Confidence", f"{low_confidence:,} voxels")
                    
                    # Show confidence distribution  
                    fig, ax = plt.subplots(figsize=(8, 4))
                    confidence_values = confidence_map[confident_voxels]
                    ax.hist(confidence_values, bins=50, color='purple', alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Confidence Score')
                    ax.set_ylabel('Number of Voxels')
                    ax.set_title('Discrimination Confidence Distribution')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                    
                # Show discrimination visualization for current slice
                if st.checkbox("Show Discrimination Visualization for Current Slice"):
                    current_slice = st.session_state.current_slice
                    
                    # Get the masks for the current slice
                    bone_slice = st.session_state.masks.get('bone', np.zeros_like(ct_volume[current_slice], dtype=bool))
                    artifact_slice = st.session_state.masks.get('bright_artifacts', np.zeros_like(ct_volume[current_slice], dtype=bool))
                    
                    if bone_slice.ndim == 3:
                        bone_slice = bone_slice[current_slice]
                    if artifact_slice.ndim == 3:
                        artifact_slice = artifact_slice[current_slice]
                    
                    # Get confidence for this slice
                    conf_slice = confidence_map[current_slice] if confidence_map.ndim == 3 else confidence_map
                    
                    # Create visualization
                    fig_disc = visualize_discrimination_slice(
                        st.session_state.ct_volume[current_slice],
                        bone_slice,
                        artifact_slice,
                        conf_slice,
                        current_slice
                    )
                    st.pyplot(fig_disc)
                    plt.close()
        else:
            st.info("Perform segmentation to see volume statistics")
    
    with tab5:
        st.subheader("Real-time Threshold Preview")
        st.info("🎯 Adjust thresholds in the sidebar to see real-time preview of segmentation")
        
        # Determine current segmentation method
        segmentation_method = "russian_doll"  # Default
        if 'segmentation_method' in locals():
            if "Legacy" in segmentation_method:
                method = 'legacy'
            else:
                method = 'russian_doll'
        else:
            method = 'russian_doll'
        
        # Show histogram with thresholds
        st.markdown("### HU Distribution with Threshold Overlays")
        
        # Add option to show full volume or current slice
        histogram_mode = st.radio(
            "Histogram Data Source",
            ["Current Slice", "Full Volume (Sampled)"],
            horizontal=True
        )
        
        if histogram_mode == "Current Slice":
            hist_fig = create_histogram_with_thresholds(
                st.session_state.ct_volume,
                st.session_state.thresholds,
                method=method,
                slice_index=st.session_state.current_slice
            )
        else:
            hist_fig = create_histogram_with_thresholds(
                st.session_state.ct_volume,
                st.session_state.thresholds,
                method=method
            )
        
        st.pyplot(hist_fig)
        plt.close()
        
        # Show threshold preview on current slice
        st.markdown("### Threshold Segmentation Preview")
        st.caption("Preview shows how current threshold settings would segment the displayed slice")
        
        current_slice_data = st.session_state.ct_volume[st.session_state.current_slice]
        preview_fig = create_threshold_preview(
            current_slice_data,
            st.session_state.thresholds,
            method=method
        )
        
        st.pyplot(preview_fig)
        plt.close()
        
        # Show threshold values summary
        st.markdown("### Current Threshold Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Dark Artifacts**")
            if method == 'russian_doll':
                st.write(f"Range: {st.session_state.thresholds['russian_doll']['dark_min']:.0f} to {st.session_state.thresholds['russian_doll']['dark_max']:.0f} HU")
            else:
                st.write(f"Max: {st.session_state.thresholds['legacy']['dark_max']:.0f} HU")
        
        with col2:
            st.markdown("**Bright Artifacts**")
            if method == 'russian_doll':
                st.write(f"Range: {st.session_state.thresholds['russian_doll']['bright_min']:.0f} to {st.session_state.thresholds['russian_doll']['bright_max']:.0f} HU")
            else:
                st.write(f"Range: {st.session_state.thresholds['legacy']['bright_min']:.0f} to {st.session_state.thresholds['legacy']['bright_max']:.0f} HU")
        
        with col3:
            st.markdown("**Metal Detection**")
            st.write(f"Threshold: {st.session_state.thresholds['metal_detection']['metal_threshold']:.0f} HU")
        
        # Info about real-time updates
        st.markdown("---")
        st.markdown("💡 **Tip**: Adjust threshold sliders in the sidebar to see immediate changes in the preview above")

else:
    # No data loaded
    st.info("👈 Please select a patient and load data from the sidebar to begin analysis")
    
    # Show instructions
    with st.expander("Quick Start Guide", expanded=True):
        st.markdown("""
        ### How to use this application:
        
        1. **Load Data**: Select a patient from the sidebar and click "Load Patient Data"
        2. **Detect Metal**: Click "Detect Metal Automatically" to find the implant using FW75% thresholding
        3. **Segment Artifacts**: Click "Segment All Artifacts" to identify bright/dark artifacts and bone
        4. **Explore Results**: 
           - Use the tabs to view different analysis modes
           - Adjust thresholds in the sidebar as needed
           - Export results as NIFTI files
        
        ### Key Features:
        - **Automatic Metal Detection**: Uses 16-point star profiles and FW75% maximum thresholding
        - **Boolean Operations**: Subtracts metal from artifact regions for accurate segmentation
        - **Adaptive Thresholding**: Each slice uses its own optimized thresholds
        - **NIFTI Export**: Save contours for machine learning pipelines
        """)
    
    # Display available patients
    st.subheader("Available Patients")
    data_dir = Path("../data")
    if not data_dir.exists():
        data_dir = Path("data")
    
    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir() and "HIP" in d.name] if data_dir.exists() else []
    
    if patient_dirs:
        cols = st.columns(3)
        for i, patient_dir in enumerate(sorted(patient_dirs)):
            with cols[i % 3]:
                dcm_files = list(patient_dir.rglob('*.dcm'))
                st.metric(patient_dir.name, f"{len(dcm_files)} DICOM files")
    else:
        st.warning("No patient data found in the data directory")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p style='color: #888'>CT Metal Artifact Characterization Tool | Advanced Medical Imaging Analysis</p>
    </div>
    """,
    unsafe_allow_html=True
)