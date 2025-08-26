import numpy as np
import os
import pydicom
from skimage.draw import polygon2mask


def load_dicom_series_to_hu(dicom_dir):
    """
    Load a DICOM series from a directory and convert to Hounsfield Units.
    
    Args:
        dicom_dir: Path to directory containing DICOM files
        
    Returns:
        tuple: (3D numpy array of HU values, dict with spatial metadata)
    """
    slices = []
    for s in os.listdir(dicom_dir):
        try:
            filepath = os.path.join(dicom_dir, s)
            ds = pydicom.dcmread(filepath)
            if ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.2' and 'PixelData' in ds:
                slices.append(ds)
        except (pydicom.errors.InvalidDicomError, AttributeError, KeyError):
            # Skip non-DICOM files or files with missing required attributes
            continue
    
    if not slices:
        print(f"No valid CT DICOM images found in {dicom_dir}")
        return None, None
    
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    
    pixel_spacing = slices[0].PixelSpacing
    slice_thickness = float(slices[0].SliceThickness)
    spacing_between_slices = getattr(slices[0], 'SpacingBetweenSlices', slice_thickness)
    image_position_patient = np.array(slices[0].ImagePositionPatient)
    origin_3d = image_position_patient
    spacing_3d = np.array([pixel_spacing[1], pixel_spacing[0], spacing_between_slices])
    
    image_3d_shape = (len(slices), slices[0].Rows, slices[0].Columns)
    image_3d_hu = np.zeros(image_3d_shape, dtype=np.int16)
    
    for i, s in enumerate(slices):
        raw_pixel_array = s.pixel_array
        rescale_slope = getattr(s, 'RescaleSlope', 1)
        rescale_intercept = getattr(s, 'RescaleIntercept', 0)
        hu_array = raw_pixel_array * rescale_slope + rescale_intercept
        image_3d_hu[i, :, :] = hu_array
    
    slice_z_positions = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    
    return image_3d_hu, {
        'origin': origin_3d,
        'spacing': spacing_3d,
        'shape': image_3d_hu.shape,
        'slice_z_positions': slice_z_positions,
        'slice_thickness': slice_thickness
    }


def create_metal_mask_from_rtstruct(rtstruct_path, ct_spatial_meta, slice_index, implant_roi_names):
    """
    Creates a boolean mask for a specific slice from an RTSTRUCT file.

    Args:
        rtstruct_path: Path to the DICOM RTSTRUCT file
        ct_spatial_meta: The spatial metadata from loading function
        slice_index: The index of the slice to generate the mask for
        implant_roi_names: List of possible names for the implant contours

    Returns:
        2D boolean numpy array representing the combined mask, or None
    """
    rtstruct = pydicom.dcmread(rtstruct_path)
    slice_z_pos = ct_spatial_meta['slice_z_positions'][slice_index]
    slice_shape = (ct_spatial_meta['shape'][1], ct_spatial_meta['shape'][2])
    origin = ct_spatial_meta['origin']
    spacing = ct_spatial_meta['spacing']

    combined_mask = np.zeros(slice_shape, dtype=bool)

    # Find the ROI Number(s) corresponding to implant names
    roi_numbers = []
    for roi in rtstruct.StructureSetROISequence:
        if any(name.lower() in roi.ROIName.lower() for name in implant_roi_names):
            roi_numbers.append(roi.ROINumber)

    if not roi_numbers:
        print(f"Warning: No contours found with names matching {implant_roi_names}")
        return None

    # Loop through all contours and find the ones matching ROI number(s)
    for roi_contour in rtstruct.ROIContourSequence:
        if roi_contour.ReferencedROINumber in roi_numbers:
            if not hasattr(roi_contour, 'ContourSequence'):
                continue
            
            # Loop through each contour slice
            for contour_slice in roi_contour.ContourSequence:
                contour_data = np.array(contour_slice.ContourData).reshape(-1, 3)
                contour_z = contour_data[0, 2]

                # Check if the contour is on the CT slice we are analyzing
                if np.isclose(contour_z, slice_z_pos):
                    # Convert world coordinates (mm) to pixel coordinates
                    pixel_coords = contour_data[:, :2]
                    pixel_coords[:, 0] = (pixel_coords[:, 0] - origin[0]) / spacing[1]
                    pixel_coords[:, 1] = (pixel_coords[:, 1] - origin[1]) / spacing[0]
                    pixel_coords = pixel_coords[:, [1, 0]]  # Swap to (row, col)

                    # Create a mask from the polygon and merge it
                    poly_mask = polygon2mask(slice_shape, pixel_coords)
                    combined_mask |= poly_mask

    if not np.any(combined_mask):
        print(f"Warning: Matching contours were found, but none were on slice {slice_index} (Z={slice_z_pos:.2f}mm)")
        return None

    return combined_mask