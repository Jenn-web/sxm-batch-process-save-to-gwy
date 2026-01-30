"""
SXM Batch Processor - Process and convert .sxm files to .gwy format

This module provides functionality to:
- Find all .sxm files in a directory tree
- Apply various processing operations (plane leveling, row alignment, etc.)
- Save processed data to .gwy format preserving metadata
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
import nanonispy as nap
import gwyfile


def find_sxm_files(directory: Union[str, Path]) -> List[Path]:
    """
    Find all .sxm files in directory and subdirectories.
    
    Args:
        directory: Path to directory to search
        
    Returns:
        List of Path objects for all .sxm files found
    """
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory {directory} does not exist")
    
    sxm_files = list(directory.rglob("*.sxm"))
    return sxm_files


def level_by_mean_plane(data: np.ndarray) -> np.ndarray:
    """
    Level data by mean plane subtraction.
    
    Fits a plane z = ax + by + c to the data and subtracts it.
    
    Args:
        data: 2D numpy array of data
        
    Returns:
        Leveled data
    """
    rows, cols = data.shape
    
    # Create coordinate arrays
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = data.flatten()
    
    # Remove NaN values if present
    mask = ~np.isnan(z_flat)
    x_flat = x_flat[mask]
    y_flat = y_flat[mask]
    z_flat = z_flat[mask]
    
    # Build design matrix for plane fitting: z = ax + by + c
    A = np.column_stack([x_flat, y_flat, np.ones_like(x_flat)])
    
    # Solve least squares problem
    coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
    
    # Calculate plane
    plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]
    
    # Subtract plane
    leveled = data - plane
    
    return leveled


def align_rows_mean(data: np.ndarray) -> np.ndarray:
    """
    Align rows by subtracting the mean of each row.
    
    Args:
        data: 2D numpy array of data
        
    Returns:
        Row-aligned data
    """
    aligned = data.astype(np.float64, copy=True)
    for i in range(data.shape[0]):
        row_mean = np.nanmean(data[i, :])
        aligned[i, :] -= row_mean
    return aligned


def align_rows_median(data: np.ndarray) -> np.ndarray:
    """
    Align rows by subtracting the median of each row.
    
    Args:
        data: 2D numpy array of data
        
    Returns:
        Row-aligned data
    """
    aligned = data.astype(np.float64, copy=True)
    for i in range(data.shape[0]):
        row_median = np.nanmedian(data[i, :])
        aligned[i, :] -= row_median
    return aligned


def align_rows_match_height(data: np.ndarray, reference: str = 'first') -> np.ndarray:
    """
    Align rows by matching heights at edges.
    
    Args:
        data: 2D numpy array of data
        reference: 'first' to align to first row, 'previous' to align to previous row
        
    Returns:
        Row-aligned data
    """
    aligned = data.astype(np.float64, copy=True)
    
    if reference == 'first':
        # Align all rows to the first row's average value
        first_mean = np.nanmean(data[0, :])
        for i in range(1, data.shape[0]):
            row_mean = np.nanmean(data[i, :])
            aligned[i, :] -= (row_mean - first_mean)
    elif reference == 'previous':
        # Align each row to the previous row
        for i in range(1, data.shape[0]):
            prev_mean = np.nanmean(aligned[i-1, :])
            curr_mean = np.nanmean(data[i, :])
            aligned[i, :] -= (curr_mean - prev_mean)
    
    return aligned


def parabolic_subtraction(data: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Subtract parabolic background along specified axis.
    
    Args:
        data: 2D numpy array of data
        axis: 0 for column-wise, 1 for row-wise
        
    Returns:
        Data with parabolic background subtracted
    """
    corrected = data.copy()
    
    if axis == 1:  # Row-wise
        for i in range(data.shape[0]):
            x = np.arange(data.shape[1])
            y = data[i, :]
            
            # Fit parabola: y = ax^2 + bx + c
            valid = ~np.isnan(y)
            if np.sum(valid) > 3:
                coeffs = np.polyfit(x[valid], y[valid], 2)
                parabola = np.polyval(coeffs, x)
                corrected[i, :] -= parabola
    else:  # Column-wise
        for j in range(data.shape[1]):
            x = np.arange(data.shape[0])
            y = data[:, j]
            
            # Fit parabola
            valid = ~np.isnan(y)
            if np.sum(valid) > 3:
                coeffs = np.polyfit(x[valid], y[valid], 2)
                parabola = np.polyval(coeffs, x)
                corrected[:, j] -= parabola
    
    return corrected


def remove_scars(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Remove scars from data by detecting and interpolating anomalous lines.
    
    Uses statistical methods to detect rows/columns that deviate significantly
    from neighbors and replaces them with interpolated values.
    
    Args:
        data: 2D numpy array of data
        threshold: Number of standard deviations for scar detection
        
    Returns:
        Data with scars removed
    """
    corrected = data.copy()
    
    # Detect horizontal scars (anomalous rows)
    row_diffs = np.abs(np.diff(data, axis=0))
    row_diff_means = np.nanmean(row_diffs, axis=1)
    row_threshold = np.nanmean(row_diff_means) + threshold * np.nanstd(row_diff_means)
    
    for i in range(len(row_diff_means)):
        if row_diff_means[i] > row_threshold:
            # Interpolate this row
            if i > 0 and i < data.shape[0] - 2:
                corrected[i+1, :] = (data[i, :] + data[i+2, :]) / 2
    
    # Detect vertical scars (anomalous columns)
    col_diffs = np.abs(np.diff(data, axis=1))
    col_diff_means = np.nanmean(col_diffs, axis=0)
    col_threshold = np.nanmean(col_diff_means) + threshold * np.nanstd(col_diff_means)
    
    for j in range(len(col_diff_means)):
        if col_diff_means[j] > col_threshold:
            # Interpolate this column
            if j > 0 and j < data.shape[1] - 2:
                corrected[:, j+1] = (corrected[:, j] + corrected[:, j+2]) / 2
    
    return corrected


def process_data(data: np.ndarray, 
                 level_plane: bool = True,
                 align_rows: Optional[str] = 'mean',
                 parabolic_sub: bool = True,
                 remove_scars_flag: bool = True,
                 scar_threshold: float = 3.0) -> np.ndarray:
    """
    Apply processing pipeline to data.
    
    Args:
        data: 2D numpy array of data
        level_plane: Whether to apply mean plane subtraction
        align_rows: Row alignment method ('mean', 'median', 'match_height', or None)
        parabolic_sub: Whether to apply parabolic subtraction
        remove_scars_flag: Whether to remove scars
        scar_threshold: Threshold for scar detection
        
    Returns:
        Processed data
    """
    # Ensure data is float type for processing
    processed = data.astype(np.float64, copy=True)
    
    if level_plane:
        processed = level_by_mean_plane(processed)
    
    if align_rows:
        if align_rows == 'mean':
            processed = align_rows_mean(processed)
        elif align_rows == 'median':
            processed = align_rows_median(processed)
        elif align_rows == 'match_height':
            processed = align_rows_match_height(processed)
    
    if parabolic_sub:
        processed = parabolic_subtraction(processed, axis=1)
    
    if remove_scars_flag:
        processed = remove_scars(processed, threshold=scar_threshold)
    
    return processed


def sxm_to_gwy(sxm_file: Union[str, Path], 
               output_file: Optional[Union[str, Path]] = None,
               level_plane: bool = True,
               align_rows: Optional[str] = 'mean',
               parabolic_sub: bool = True,
               remove_scars_flag: bool = True,
               scar_threshold: float = 3.0) -> Path:
    """
    Convert .sxm file to .gwy format with processing.
    
    Args:
        sxm_file: Path to input .sxm file
        output_file: Path to output .gwy file (default: same name as input with .gwy extension)
        level_plane: Whether to apply mean plane subtraction
        align_rows: Row alignment method ('mean', 'median', 'match_height', or None)
        parabolic_sub: Whether to apply parabolic subtraction
        remove_scars_flag: Whether to remove scars
        scar_threshold: Threshold for scar detection
        
    Returns:
        Path to created .gwy file
    """
    sxm_file = Path(sxm_file)
    
    if output_file is None:
        output_file = sxm_file.with_suffix('.gwy')
    else:
        output_file = Path(output_file)
    
    # Read SXM file
    scan = nap.read.Scan(str(sxm_file))
    
    # Create GWY file object
    gwy_obj = gwyfile.GwyObject()
    
    # Process each channel in the SXM file
    for channel_name, channel_data in scan.signals.items():
        # Get data and metadata
        data = channel_data['forward']  # Use forward scan
        
        # Process the data
        processed_data = process_data(
            data,
            level_plane=level_plane,
            align_rows=align_rows,
            parabolic_sub=parabolic_sub,
            remove_scars_flag=remove_scars_flag,
            scar_threshold=scar_threshold
        )
        
        # Get physical dimensions
        height, width = processed_data.shape
        
        # Get physical size from scan parameters
        # SXM files store size in meters
        x_size = scan.header['scan_range'][0]  # in meters
        y_size = scan.header['scan_range'][1]  # in meters
        
        # Create GWY DataField
        datafield = gwyfile.GwyDataField(
            processed_data,
            xreal=x_size,
            yreal=y_size,
            si_unit_xy=gwyfile.GwySIUnit('m'),
            si_unit_z=gwyfile.GwySIUnit(channel_data.get('unit', 'm'))
        )
        
        # Add to GWY object
        channel_key = f"/{len(gwy_obj)}/data"
        gwy_obj[channel_key] = datafield
        
        # Add metadata
        title_key = f"/{len(gwy_obj)-1}/data/title"
        gwy_obj[title_key] = channel_name
        
        # Add original metadata as meta
        meta_key = f"/{len(gwy_obj)-1}/meta"
        meta = gwyfile.GwyObject()
        
        # Store important scan parameters
        if hasattr(scan, 'header') and scan.header:
            for key, value in scan.header.items():
                try:
                    # Convert value to string for storage in metadata
                    meta[key] = str(value)
                except:
                    pass
        
        gwy_obj[meta_key] = meta
    
    # Save GWY file
    gwyfile.dump(gwy_obj, str(output_file))
    
    return output_file


def batch_process_directory(directory: Union[str, Path],
                            level_plane: bool = True,
                            align_rows: Optional[str] = 'mean',
                            parabolic_sub: bool = True,
                            remove_scars_flag: bool = True,
                            scar_threshold: float = 3.0,
                            verbose: bool = True) -> List[Path]:
    """
    Process all .sxm files in directory tree and save as .gwy files.
    
    Args:
        directory: Root directory to search for .sxm files
        level_plane: Whether to apply mean plane subtraction
        align_rows: Row alignment method ('mean', 'median', 'match_height', or None)
        parabolic_sub: Whether to apply parabolic subtraction
        remove_scars_flag: Whether to remove scars
        scar_threshold: Threshold for scar detection
        verbose: Whether to print progress messages
        
    Returns:
        List of paths to created .gwy files
    """
    # Find all SXM files
    sxm_files = find_sxm_files(directory)
    
    if verbose:
        print(f"Found {len(sxm_files)} .sxm files")
    
    created_files = []
    
    for i, sxm_file in enumerate(sxm_files, 1):
        if verbose:
            print(f"Processing {i}/{len(sxm_files)}: {sxm_file.name}")
        
        try:
            gwy_file = sxm_to_gwy(
                sxm_file,
                level_plane=level_plane,
                align_rows=align_rows,
                parabolic_sub=parabolic_sub,
                remove_scars_flag=remove_scars_flag,
                scar_threshold=scar_threshold
            )
            created_files.append(gwy_file)
            
            if verbose:
                print(f"  -> Saved to {gwy_file.name}")
        except Exception as e:
            if verbose:
                print(f"  -> Error: {e}")
    
    if verbose:
        print(f"\nCompleted: {len(created_files)}/{len(sxm_files)} files processed successfully")
    
    return created_files
