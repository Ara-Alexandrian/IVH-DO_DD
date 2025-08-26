import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
from datetime import datetime
from skimage import measure


def create_rtstruct_from_masks(masks_dict, ct_metadata, patient_info, contour_names=None):
    """
    Create a DICOM RT Structure Set from binary masks.
    
    Args:
        masks_dict: Dictionary of binary masks {name: 3D numpy array}
        ct_metadata: Metadata from the original CT scan
        patient_info: Patient information dictionary
        contour_names: Custom names for contours
        
    Returns:
        pydicom Dataset containing the RT Structure Set
    """
    # Create file meta information
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'  # RT Structure Set
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    
    # Create the main dataset
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Patient Module
    ds.PatientName = patient_info.get('PatientName', 'Anonymous')
    ds.PatientID = patient_info.get('PatientID', '000000')
    ds.PatientBirthDate = patient_info.get('PatientBirthDate', '')
    ds.PatientSex = patient_info.get('PatientSex', '')
    
    # Study Module
    ds.StudyInstanceUID = ct_metadata.get('StudyInstanceUID', generate_uid())
    ds.StudyDate = ct_metadata.get('StudyDate', datetime.now().strftime('%Y%m%d'))
    ds.StudyTime = ct_metadata.get('StudyTime', datetime.now().strftime('%H%M%S'))
    ds.StudyID = ct_metadata.get('StudyID', '1')
    
    # Series Module
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesDate = datetime.now().strftime('%Y%m%d')
    ds.SeriesTime = datetime.now().strftime('%H%M%S')
    ds.Modality = 'RTSTRUCT'
    ds.SeriesNumber = 1
    
    # RT Structure Set Module
    ds.StructureSetLabel = 'Metal Artifact Characterization'
    ds.StructureSetName = 'MAC_Contours'
    ds.StructureSetDate = datetime.now().strftime('%Y%m%d')
    ds.StructureSetTime = datetime.now().strftime('%H%M%S')
    
    # Frame of Reference Module
    ds.FrameOfReferenceUID = ct_metadata.get('FrameOfReferenceUID', generate_uid())
    
    # Referenced Frame of Reference Sequence
    refd_frame_of_ref = Dataset()
    refd_frame_of_ref.FrameOfReferenceUID = ds.FrameOfReferenceUID
    
    # RT Referenced Study Sequence
    rt_refd_study = Dataset()
    rt_refd_study.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.2'  # Study Component Management
    rt_refd_study.ReferencedSOPInstanceUID = ds.StudyInstanceUID
    
    # RT Referenced Series Sequence
    rt_refd_series = Dataset()
    rt_refd_series.SeriesInstanceUID = ct_metadata.get('SeriesInstanceUID', generate_uid())
    
    # Contour Image Sequence - reference all CT slices
    contour_image_seq = []
    for uid in ct_metadata.get('sop_instance_uids', []):
        contour_image = Dataset()
        contour_image.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        contour_image.ReferencedSOPInstanceUID = uid
        contour_image_seq.append(contour_image)
    
    rt_refd_series.ContourImageSequence = contour_image_seq
    rt_refd_study.RTReferencedSeriesSequence = [rt_refd_series]
    refd_frame_of_ref.RTReferencedStudySequence = [rt_refd_study]
    ds.ReferencedFrameOfReferenceSequence = [refd_frame_of_ref]
    
    # Structure Set ROI Sequence
    roi_seq = []
    roi_contour_seq = []
    roi_observations_seq = []
    
    # Define colors for each structure type
    colors = {
        'metal': [255, 0, 0],          # Red
        'bright_artifacts': [255, 255, 0],  # Yellow
        'dark_artifacts': [255, 0, 255],    # Magenta
        'bone': [0, 51, 204]           # Blue
    }
    
    # Process each mask
    roi_number = 1
    for mask_name, mask in masks_dict.items():
        if not isinstance(mask, np.ndarray) or mask.ndim != 3:
            continue
            
        # Structure Set ROI
        roi = Dataset()
        roi.ROINumber = roi_number
        roi.ReferencedFrameOfReferenceUID = ds.FrameOfReferenceUID
        roi.ROIName = contour_names.get(mask_name, mask_name.replace('_', ' ').title()) if contour_names else mask_name
        roi.ROIGenerationAlgorithm = 'AUTOMATIC'
        roi_seq.append(roi)
        
        # ROI Contour
        roi_contour = Dataset()
        roi_contour.ROIDisplayColor = colors.get(mask_name, [128, 128, 128])
        roi_contour.ReferencedROINumber = roi_number
        
        # Create contours for each slice
        contour_seq = []
        spacing = ct_metadata['spacing']
        slice_positions = ct_metadata['slice_z_positions']
        
        for z_idx in range(mask.shape[0]):
            if not np.any(mask[z_idx]):
                continue
                
            # Find contours in this slice
            contours = measure.find_contours(mask[z_idx].astype(float), 0.5)
            
            for contour in contours:
                if len(contour) < 3:  # Skip very small contours
                    continue
                    
                contour_item = Dataset()
                contour_item.ContourImageSequence = [contour_image_seq[z_idx]] if z_idx < len(contour_image_seq) else []
                
                # Convert contour points to patient coordinates
                contour_data = []
                for point in contour:
                    # point is in (row, col) format
                    y_idx, x_idx = point
                    
                    # Convert to patient coordinates
                    x_mm = x_idx * spacing[2]
                    y_mm = y_idx * spacing[1]
                    z_mm = slice_positions[z_idx] if z_idx < len(slice_positions) else z_idx * spacing[0]
                    
                    contour_data.extend([x_mm, y_mm, z_mm])
                
                contour_item.ContourGeometricType = 'CLOSED_PLANAR'
                contour_item.NumberOfContourPoints = len(contour)
                contour_item.ContourData = contour_data
                contour_seq.append(contour_item)
        
        roi_contour.ContourSequence = contour_seq
        roi_contour_seq.append(roi_contour)
        
        # RT ROI Observations
        roi_obs = Dataset()
        roi_obs.ObservationNumber = roi_number
        roi_obs.ReferencedROINumber = roi_number
        roi_obs.ROIObservationLabel = roi.ROIName
        roi_obs.RTROIInterpretedType = 'ORGAN'
        roi_observations_seq.append(roi_obs)
        
        roi_number += 1
    
    # Add sequences to dataset
    ds.StructureSetROISequence = roi_seq
    ds.ROIContourSequence = roi_contour_seq
    ds.RTROIObservationsSequence = roi_observations_seq
    
    # SOP Common Module
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    
    return ds


def save_rtstruct(ds, output_path):
    """
    Save RT Structure Set to file.
    
    Args:
        ds: pydicom Dataset
        output_path: Path to save the file
    """
    ds.save_as(output_path, write_like_original=False)