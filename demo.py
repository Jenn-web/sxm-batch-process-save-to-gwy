#!/usr/bin/env python3
"""
Simple demo script for sxm_batch_processor

This script demonstrates basic usage of the module.
"""

import sxm_batch_processor as sbp
from pathlib import Path


def main():
    """Main demonstration function"""
    
    print("=" * 70)
    print("SXM Batch Processor - Demo Script")
    print("=" * 70)
    print()
    
    print("This module provides easy-to-use functions for batch processing")
    print(".sxm files and converting them to .gwy format.")
    print()
    
    # Example 1: Process a single file
    print("Example 1: Process a single file")
    print("-" * 70)
    print("Code:")
    print("  output_file = sbp.sxm_to_gwy('data/sample.sxm')")
    print()
    
    # Example 2: Batch process a directory
    print("Example 2: Batch process entire directory")
    print("-" * 70)
    print("Code:")
    print("  created_files = sbp.batch_process_directory(")
    print("      'data/',")
    print("      level_plane=True,")
    print("      align_rows='mean',")
    print("      parabolic_sub=True,")
    print("      remove_scars_flag=True,")
    print("      mask_unscanned=True,        # NEW: Mask unscanned areas")
    print("      handle_outliers_flag=True,  # NEW: Handle outliers")
    print("      verbose=True")
    print("  )")
    print()
    
    # Example 3: Custom processing settings
    print("Example 3: Custom processing settings")
    print("-" * 70)
    print("Code:")
    print("  # Use median row alignment and custom outlier handling")
    print("  created_files = sbp.batch_process_directory(")
    print("      'data/',")
    print("      align_rows='median',         # More robust to outliers")
    print("      scar_threshold=2.5,          # More sensitive scar detection")
    print("      outlier_method='percentile', # Use percentile clipping")
    print("      percentile_range=(1, 99),    # Custom percentile range")
    print("      parabolic_sub=False,         # Skip parabolic subtraction")
    print("  )")
    print()
    
    # Example 4: New features demonstration
    print("Example 4: NEW Features - Unscanned area masking & outlier handling")
    print("-" * 70)
    print("Code:")
    print("  # Enable new features for better color scaling")
    print("  created_files = sbp.batch_process_directory(")
    print("      'data/',")
    print("      mask_unscanned=True,         # Mask zeros/constant areas")
    print("      handle_outliers_flag=True,   # Handle sharp changes")
    print("      outlier_method='percentile', # 'clip', 'percentile', or 'median_filter'")
    print("      percentile_range=(0.5, 99.5) # Percentile bounds")
    print("  )")
    print()
    print("Benefits:")
    print("  • Unscanned areas (zeros/constants) don't affect color bar")
    print("  • Sharp changes and outliers are clipped for better visualization")
    print("  • Percentile method is robust and works well for most data")
    print()
    
    # Show available functions
    print("Available Functions:")
    print("-" * 70)
    functions = [
        ('find_sxm_files(directory)', 'Find all .sxm files recursively'),
        ('sxm_to_gwy(sxm_file)', 'Convert single file to .gwy'),
        ('batch_process_directory(dir)', 'Process all files in directory'),
        ('level_by_mean_plane(data)', 'Apply plane leveling'),
        ('align_rows_mean(data)', 'Align rows by mean'),
        ('align_rows_median(data)', 'Align rows by median'),
        ('parabolic_subtraction(data)', 'Remove parabolic background'),
        ('remove_scars(data)', 'Remove anomalous lines'),
        ('mask_unscanned_areas(data)', 'NEW: Mask zeros/constant regions'),
        ('handle_outliers(data)', 'NEW: Handle sharp changes/outliers'),
        ('process_data(data)', 'Apply full processing pipeline'),
    ]
    
    for func, desc in functions:
        print(f"  • {func:35} - {desc}")
    
    print()
    print("=" * 70)
    print("For more examples, see batch_process_example.ipynb")
    print("=" * 70)


if __name__ == "__main__":
    main()
