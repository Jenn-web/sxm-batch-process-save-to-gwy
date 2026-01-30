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
import nanonispy2 as nap
import gwyfile
import gwyfile.objects as gwyobj


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
        
    Raises:
        ValueError: If reference is not 'first' or 'previous'
    """
    if reference not in ('first', 'previous'):
        raise ValueError(f"reference must be 'first' or 'previous', got '{reference}'")
    
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
        
    Raises:
        ValueError: If axis is not 0 or 1
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    
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


def mask_unscanned_areas(data: np.ndarray, 
                         zero_threshold: float = 1e-10,
                         constant_threshold: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect and mask unscanned areas in the data.
    
    Unscanned areas are typically filled with zeros or constant values,
    which can affect the color bar and make useful information invisible.
    
    Args:
        data: 2D numpy array of data
        zero_threshold: Threshold for detecting zero values (absolute value)
        constant_threshold: Threshold for detecting constant regions (std dev)
        
    Returns:
        Tuple of (mask, masked_data) where mask is True for valid data points
        and masked_data has NaN in unscanned areas
    """
    mask = np.ones(data.shape, dtype=bool)
    
    # Detect areas that are exactly or very close to zero
    zero_mask = np.abs(data) > zero_threshold
    
    # Detect constant regions using local standard deviation
    # Simple approach: check if local variation is very small
    window_size = min(5, max(3, data.shape[0] // 10), max(3, data.shape[1] // 10))
    
    # Use a simple manual convolution approach for local std
    # Calculate local standard deviation without scipy
    constant_mask = np.ones(data.shape, dtype=bool)
    
    if window_size >= 3:
        rows, cols = data.shape
        half_w = window_size // 2
        
        for i in range(half_w, rows - half_w):
            for j in range(half_w, cols - half_w):
                # Get local window
                window = data[i-half_w:i+half_w+1, j-half_w:j+half_w+1]
                local_std = np.std(window)
                
                # Mark as constant if std is very small
                if local_std <= constant_threshold:
                    constant_mask[i, j] = False
    
    # Combine masks
    mask = zero_mask & constant_mask
    
    # Create masked data with NaN in unscanned areas
    masked_data = data.copy()
    masked_data[~mask] = np.nan
    
    return mask, masked_data


def handle_outliers(data: np.ndarray, 
                    method: str = 'clip',
                    sigma: float = 3.0,
                    percentile_range: Tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """
    Handle outliers and sharp changes in scan data.
    
    Sharp changes and outliers can affect the image quality and color scaling.
    This function provides several methods to deal with them.
    
    Args:
        data: 2D numpy array of data
        method: Method to handle outliers - 'clip', 'percentile', or 'median_filter'
        sigma: Number of standard deviations for clipping (for 'clip' method)
        percentile_range: Tuple of (low, high) percentiles for clipping (for 'percentile' method)
        
    Returns:
        Data with outliers handled
        
    Raises:
        ValueError: If method is not 'clip', 'percentile', or 'median_filter'
    """
    if method not in ('clip', 'percentile', 'median_filter'):
        raise ValueError(f"method must be 'clip', 'percentile', or 'median_filter', got '{method}'")
    
    processed = data.copy()
    
    if method == 'clip':
        # Clip based on mean and standard deviation
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            mean = np.mean(valid_data)
            std = np.std(valid_data)
            lower_bound = mean - sigma * std
            upper_bound = mean + sigma * std
            processed = np.clip(processed, lower_bound, upper_bound)
    
    elif method == 'percentile':
        # Clip based on percentiles
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            lower_bound = np.percentile(valid_data, percentile_range[0])
            upper_bound = np.percentile(valid_data, percentile_range[1])
            processed = np.clip(processed, lower_bound, upper_bound)
    
    elif method == 'median_filter':
        # Use simple median filter to smooth sharp changes
        # Manual implementation without scipy
        kernel_size = 3
        rows, cols = data.shape
        filtered = data.copy()
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # Get 3x3 window around pixel
                window = data[i-1:i+2, j-1:j+2].flatten()
                # Remove NaN values
                window = window[~np.isnan(window)]
                if len(window) > 0:
                    filtered[i, j] = np.median(window)
        
        # Only replace extreme values
        finite_mask = np.isfinite(data)
        if np.any(finite_mask):
            valid_data = data[finite_mask]
            if len(valid_data) > 0:
                mean = np.mean(valid_data)
                std = np.std(valid_data)
                extreme_mask = np.abs(data - mean) > sigma * std
                processed[extreme_mask & finite_mask] = filtered[extreme_mask & finite_mask]
    
    return processed


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
                 scar_threshold: float = 3.0,
                 mask_unscanned: bool = True,
                 handle_outliers_flag: bool = True,
                 outlier_method: str = 'percentile',
                 outlier_sigma: float = 3.0) -> np.ndarray:
    """
    Apply processing pipeline to data.
    
    Args:
        data: 2D numpy array of data
        level_plane: Whether to apply mean plane subtraction
        align_rows: Row alignment method ('mean', 'median', 'match_height', or None)
        parabolic_sub: Whether to apply parabolic subtraction
        remove_scars_flag: Whether to remove scars
        scar_threshold: Threshold for scar detection
        mask_unscanned: Whether to mask unscanned areas (zeros or constant values)
        handle_outliers_flag: Whether to handle outliers and sharp changes
        outlier_method: Method for outlier handling ('clip', 'percentile', or 'median_filter')
        outlier_sigma: Sigma threshold for outlier detection
        
    Returns:
        Processed data
        
    Raises:
        ValueError: If align_rows is not None, 'mean', 'median', or 'match_height'
    """
    # Validate align_rows parameter
    if align_rows is not None and align_rows not in ('mean', 'median', 'match_height'):
        raise ValueError(f"align_rows must be None, 'mean', 'median', or 'match_height', got '{align_rows}'")
    
    # Ensure data is float type for processing
    processed = data.astype(np.float64, copy=True)
    
    # Step 1: Mask unscanned areas if requested (before other processing)
    if mask_unscanned:
        _, processed = mask_unscanned_areas(processed)
    
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
    
    # Step 2: Handle outliers and sharp changes (after other processing)
    if handle_outliers_flag:
        processed = handle_outliers(processed, method=outlier_method, sigma=outlier_sigma)
    
    return processed


def sxm_to_gwy(sxm_file: Union[str, Path], 
               output_file: Optional[Union[str, Path]] = None,
               level_plane: bool = True,
               align_rows: Optional[str] = 'mean',
               parabolic_sub: bool = True,
               remove_scars_flag: bool = True,
               scar_threshold: float = 3.0,
               mask_unscanned: bool = True,
               handle_outliers_flag: bool = True,
               outlier_method: str = 'percentile',
               outlier_sigma: float = 3.0) -> Path:
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
        mask_unscanned: Whether to mask unscanned areas
        handle_outliers_flag: Whether to handle outliers and sharp changes
        outlier_method: Method for outlier handling ('clip', 'percentile', or 'median_filter')
        outlier_sigma: Sigma threshold for outlier detection
        
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
    container = gwyobj.GwyContainer()
    
    # Process each channel in the SXM file
    channel_idx = 0
    for channel_name, channel_data in scan.signals.items():
        # Get data - try forward scan first, fallback to backward
        if 'forward' in channel_data:
            data = channel_data['forward']
        elif 'backward' in channel_data:
            data = channel_data['backward']
        else:
            # Skip this channel if no data available
            continue
        
        # Process the data
        processed_data = process_data(
            data,
            level_plane=level_plane,
            align_rows=align_rows,
            parabolic_sub=parabolic_sub,
            remove_scars_flag=remove_scars_flag,
            scar_threshold=scar_threshold,
            mask_unscanned=mask_unscanned,
            handle_outliers_flag=handle_outliers_flag,
            outlier_method=outlier_method,
            outlier_sigma=outlier_sigma
        )
        
        # Get physical size from scan parameters
        # SXM files store size in meters
        x_size = scan.header['scan_range'][0]  # in meters
        y_size = scan.header['scan_range'][1]  # in meters
        
        # Create GWY DataField
        datafield = gwyobj.GwyDataField(
            processed_data,
            xreal=x_size,
            yreal=y_size,
            si_unit_xy=gwyobj.GwySIUnit('m'),
            si_unit_z=gwyobj.GwySIUnit(channel_data.get('unit', 'm'))
        )
        
        # Add to GWY object with correct channel indexing
        channel_key = f"/{channel_idx}/data"
        container[channel_key] = datafield
        
        # Add metadata
        title_key = f"/{channel_idx}/data/title"
        container[title_key] = 'Z (Forward)'
        
        # Add original metadata as meta
        meta_key = f"/{channel_idx}/meta"
        meta = gwyobj.GwyContainer()
        
        # Store important scan parameters
        if hasattr(scan, 'header') and scan.header:
            for key, value in scan.header.items():
                try:
                    # Convert value to string for storage in metadata
                    meta[key] = str(value)
                except (TypeError, ValueError):
                    # Skip values that can't be converted to string
                    pass
        
        container[meta_key] = meta
        channel_idx += 1
    
    # Save GWY file
    container.tofile(str(output_file))
    
    return output_file


def batch_process_directory(directory: Union[str, Path],
                            level_plane: bool = True,
                            align_rows: Optional[str] = 'mean',
                            parabolic_sub: bool = True,
                            remove_scars_flag: bool = True,
                            scar_threshold: float = 3.0,
                            mask_unscanned: bool = True,
                            handle_outliers_flag: bool = True,
                            outlier_method: str = 'percentile',
                            outlier_sigma: float = 3.0,
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
        mask_unscanned: Whether to mask unscanned areas
        handle_outliers_flag: Whether to handle outliers and sharp changes
        outlier_method: Method for outlier handling ('clip', 'percentile', or 'median_filter')
        outlier_sigma: Sigma threshold for outlier detection
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
                scar_threshold=scar_threshold,
                mask_unscanned=mask_unscanned,
                handle_outliers_flag=handle_outliers_flag,
                outlier_method=outlier_method,
                outlier_sigma=outlier_sigma
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
