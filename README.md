# CT Metal Artifact Characterization

An advanced Streamlit web application for automatic detection and characterization of metal artifacts in CT scans of patients with hip implants, designed for building machine learning datasets.

## Key Features

### 🎯 Automatic Metal Detection
- **3D Adaptive Detection**: Multi-planar analysis across axial, coronal, and sagittal views
- **16-point star profile analysis** for precise boundary detection
- **FW75% Maximum thresholding** - adaptive thresholds without initial HU values
- **Individual ROI regions** per metal component (avoids bilateral capture)
- **Legacy mode** available with initial HU thresholds

### 🔬 Advanced Segmentation
- **Russian Doll Segmentation**: Smart discrimination between bone and bright artifacts
- **Star profile-based discrimination** using peak characteristics:
  - Peak width analysis (bone: broad, artifacts: narrow)
  - Smoothness metrics (bone: smooth, artifacts: sharp)
  - Directional consistency (bone: uniform, artifacts: variable)
- **Sequential exclusion** ensuring no tissue overlap
- **Legacy threshold-based** option with boolean operations
- **GPU acceleration** support (optional with CuPy)

### 📊 Visualization & Analysis
- **Multi-slice grid view** with individual ROI indicators
- **Star profile visualization** showing all 16 analysis lines
- **Discrimination confidence maps** for bone/artifact separation
- **Threshold evolution plots** across slices
- **Volume statistics** with confidence metrics
- **Interactive contour visibility** toggles
- **Custom contour naming** for exports

### 💾 Export Capabilities
- **NIFTI format** (.nii.gz) for ML pipelines
- **DICOM RT Structure Sets** for clinical compatibility
- **Separate binary masks** for each tissue type
- **Multi-label NIFTI** option
- **Confidence maps** for discrimination results
- **Automatic affine matrix generation** from DICOM metadata

## Installation

### Using Conda (Recommended)
```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate IVH
```

### Using pip
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
```bash
./run.sh
```
The application will be available at `http://192.168.1.11:4224`

### Manual Start
```bash
cd app
streamlit run main.py --server.address 192.168.1.11 --server.port 4224
```

### Local Development
```bash
cd app
streamlit run main.py
```

## Project Structure

```
├── app/                      # Application source code
│   ├── main.py              # Streamlit web application
│   ├── dicom_utils.py       # DICOM loading and metadata handling
│   ├── dicom_export.py      # DICOM RT Structure Set creation
│   ├── metal_detection.py   # Legacy metal detection algorithms
│   ├── metal_detection_v3.py # 3D adaptive metal detection
│   ├── contour_operations.py # Boolean operations and Russian doll segmentation
│   ├── artifact_discrimination.py # Star profile-based bone/artifact discrimination
│   └── visualization.py     # Advanced plotting and visualization
├── data/                    # Patient DICOM data
│   ├── HIP* Patient/       # Patient datasets
├── output/                  # Exported masks and RT structures
├── misc/                    # Test outputs and screenshots
├── archive/                 # Original Jupyter notebooks
├── requirements.txt         # Python dependencies
├── run.sh                  # Launch script
├── CLAUDE.md               # AI assistant guidance
└── README.md               # This file
```

## Workflow

### 1. Load Patient Data
- Select patient from sidebar
- Click "Load Patient Data"
- System loads all DICOM slices and metadata

### 2. Automatic Metal Detection
- Select detection method:
  - **3D Adaptive + Star Algorithm** (Recommended)
  - Legacy with Initial Threshold
- Click "🎯 Detect Metal Automatically"
- 3D Adaptive Algorithm:
  1. Analyzes all three anatomical planes
  2. Finds metal components above 2500 HU
  3. Creates individual ROIs per component
  4. Performs 16-point star profile analysis
  5. Applies FW75% thresholding (no initial HU needed)
  6. Generates per-slice adaptive thresholds

### 3. Artifact Segmentation
- Select segmentation method:
  - **Russian Doll with Smart Discrimination** (Recommended)
  - Legacy Threshold-Based
- Click "🔍 Segment All Artifacts"
- Russian Doll Process:
  1. Segments dark artifacts (< -150 HU)
  2. Analyzes bone/bright artifact candidates (300-1500 HU)
  3. Uses star profiles to discriminate bone from artifacts:
     - Measures peak width, smoothness, directional variance
     - Assigns confidence scores to each voxel
  4. Ensures mutual exclusion between all tissue types
  5. Applies morphological refinement

### 4. Analysis & Export
- View results in multiple tabs:
  - Single Slice Analysis
  - Multi-Slice View
  - Metal Detection Details
  - Volume Statistics
- Export as NIFTI files for ML pipelines

## Algorithm Details

### FW75% Maximum Thresholding
The Full Width at 75% Maximum method:
1. Generates HU vs distance profiles along 16 radial lines
2. Finds peak HU values (metal centers)
3. Calculates 75% of peak intensity as threshold
4. Averages thresholds across all profiles for robustness
5. No initial HU threshold required

### Russian Doll Segmentation
Sequential tissue segmentation with mutual exclusion:
1. **Dark artifacts**: Simple thresholding (< -150 HU) excluding metal
2. **Bone/Bright discrimination**: 
   - Analyzes star profiles for each candidate voxel
   - Bone characteristics: broad peaks (~3-5mm), smooth transitions, consistent across directions
   - Artifact characteristics: narrow peaks (<2mm), sharp edges, variable across directions
3. **Mutual exclusion**: Each voxel belongs to exactly one tissue type

### Profile-Based Discrimination Metrics
- **Peak Width**: FWHM (Full Width Half Maximum) in mm
- **Smoothness**: 1/(1 + variance of first derivative)
- **Directional Variance**: Consistency of characteristics across 16 directions
- **Edge Sharpness**: Maximum gradient magnitude
- **Confidence Score**: Weighted combination of all metrics

### 3D Adaptive Analysis
- **Multi-planar projections**: Analyzes axial, coronal, and sagittal views
- **Individual component tracking**: Separate ROIs for each metal region
- **Extent calculation**: Full 3D bounding box from all projections
- **Focused processing**: Only analyzes slices with confirmed metal

## Configuration

### Metal Detection Parameters
- **3D Search Margin**: 1.0-5.0 cm (default: 2.0 cm)
- **FW Percentage**: 50-90% (default: 75%)
- **Intensity Percentile**: 99.0-99.9% (default: 99.5%)

### Segmentation Parameters (Russian Doll)
- **Dark Artifacts**: < -150 HU (adjustable)
- **Bone/Bright Range**: 300-1500 HU (discriminated by profiles)
- **Max Artifact Distance**: 2.0-15.0 cm from metal (default: 10.0 cm)

### Legacy Parameters
- **Initial Metal Threshold**: 1000-4000 HU (default: 2000)
- **Bright Artifacts**: 800-3000 HU
- **Bone**: 150-1500 HU

## Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- Modern web browser
- DICOM CT series

### Recommended
- Python 3.12+
- 16GB RAM
- NVIDIA GPU with CUDA (for acceleration)
- CuPy for GPU processing (optional)

## Recent Enhancements

- ✅ Bilateral implant support (via individual ROIs)
- ✅ DICOM RT Structure export
- ✅ Smart bone/artifact discrimination
- ✅ GPU acceleration support
- ✅ Confidence mapping
- ✅ Custom contour naming

## Future Enhancements

- Batch processing for multiple patients
- Machine learning model integration
- 3D visualization
- Automated quality metrics
- Cloud deployment
- API endpoints for integration