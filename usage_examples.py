#!/usr/bin/env python3
"""
Example usage script for sxm_batch_processor

This script shows real-world usage patterns and demonstrates
how to verify that .gwy files can be read properly.
"""

import sxm_batch_processor as sbp
import gwyfile
from pathlib import Path


def example_basic_usage():
    """Example: Basic batch processing"""
    print("=" * 70)
    print("Example 1: Basic Batch Processing")
    print("=" * 70)
    
    # Define your data directory
    data_dir = "path/to/your/sxm/files"
    
    # Process all .sxm files with default settings
    created_files = sbp.batch_process_directory(
        data_dir,
        level_plane=True,
        align_rows='mean',
        parabolic_sub=True,
        remove_scars_flag=True,
        verbose=True
    )
    
    print(f"\nSuccessfully created {len(created_files)} .gwy files")
    print()


def example_custom_processing():
    """Example: Custom processing settings"""
    print("=" * 70)
    print("Example 2: Custom Processing Settings")
    print("=" * 70)
    
    data_dir = "path/to/your/sxm/files"
    
    # Use median row alignment (more robust to outliers)
    # Increase scar detection sensitivity
    # Skip parabolic subtraction
    created_files = sbp.batch_process_directory(
        data_dir,
        level_plane=True,
        align_rows='median',      # Use median instead of mean
        parabolic_sub=False,      # Skip parabolic subtraction
        remove_scars_flag=True,
        scar_threshold=2.5,       # More sensitive (default: 3.0)
        verbose=True
    )
    
    print(f"\nCreated {len(created_files)} .gwy files with custom settings")
    print()


def example_single_file():
    """Example: Process a single file"""
    print("=" * 70)
    print("Example 3: Process a Single File")
    print("=" * 70)
    
    input_file = "path/to/your/file.sxm"
    
    # Process single file
    output_file = sbp.sxm_to_gwy(
        input_file,
        level_plane=True,
        align_rows='mean',
        parabolic_sub=True,
        remove_scars_flag=True
    )
    
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print()


def example_verify_gwy_files():
    """Example: Verify .gwy files can be read and processed"""
    print("=" * 70)
    print("Example 4: Verify GWY Files Can Be Read")
    print("=" * 70)
    
    gwy_file = "path/to/your/file.gwy"
    
    # Load the .gwy file
    gwy_obj = gwyfile.load(gwy_file)
    
    print(f"Successfully loaded: {gwy_file}")
    print()
    
    # List all channels/datafields
    print("Channels in file:")
    for key in gwy_obj.keys():
        if key.endswith('/data'):
            datafield = gwy_obj[key]
            
            # Get title if available
            title_key = key.replace('/data', '/data/title')
            title = gwy_obj.get(title_key, 'Unknown')
            
            print(f"  Channel: {title}")
            print(f"    Shape: {datafield.data.shape}")
            print(f"    Physical size: {datafield.xreal:.2e} x {datafield.yreal:.2e} m")
            print(f"    Unit (XY): {datafield.si_unit_xy}")
            print(f"    Unit (Z): {datafield.si_unit_z}")
            print()
    
    # Show metadata
    print("Metadata:")
    for key in gwy_obj.keys():
        if key.endswith('/meta'):
            meta = gwy_obj[key]
            for meta_key in list(meta.keys())[:5]:  # Show first 5
                print(f"  {meta_key}: {meta[meta_key]}")
    print()
    
    # Demonstrate further processing
    print("The data can be further processed:")
    for key in gwy_obj.keys():
        if key.endswith('/data'):
            datafield = gwy_obj[key]
            data = datafield.data
            
            print(f"  Data statistics:")
            print(f"    Mean: {data.mean():.6e}")
            print(f"    Std:  {data.std():.6e}")
            print(f"    Min:  {data.min():.6e}")
            print(f"    Max:  {data.max():.6e}")
            break
    print()


def example_find_files():
    """Example: Find all .sxm files"""
    print("=" * 70)
    print("Example 5: Find All SXM Files")
    print("=" * 70)
    
    data_dir = "path/to/your/sxm/files"
    
    # Find all .sxm files recursively
    sxm_files = sbp.find_sxm_files(data_dir)
    
    print(f"Found {len(sxm_files)} .sxm files in {data_dir}")
    
    # Show first few files
    if len(sxm_files) > 0:
        print("\nFirst few files:")
        for f in sxm_files[:5]:
            print(f"  {f}")
        
        if len(sxm_files) > 5:
            print(f"  ... and {len(sxm_files) - 5} more")
    print()


def example_different_row_alignment_methods():
    """Example: Different row alignment methods"""
    print("=" * 70)
    print("Example 6: Different Row Alignment Methods")
    print("=" * 70)
    
    input_file = Path("path/to/your/file.sxm")
    
    # Method 1: Mean alignment (default, removes horizontal stripes)
    print("Processing with mean row alignment...")
    output1 = sbp.sxm_to_gwy(
        input_file,
        output_file=input_file.with_stem(f'{input_file.stem}_mean'),
        align_rows='mean'
    )
    
    # Method 2: Median alignment (more robust to outliers)
    print("Processing with median row alignment...")
    output2 = sbp.sxm_to_gwy(
        input_file,
        output_file=input_file.with_stem(f'{input_file.stem}_median'),
        align_rows='median'
    )
    
    # Method 3: Match height alignment (good for thermal drift)
    print("Processing with match height alignment...")
    output3 = sbp.sxm_to_gwy(
        input_file,
        output_file=input_file.with_stem(f'{input_file.stem}_match'),
        align_rows='match_height'
    )
    
    # Method 4: No alignment
    print("Processing without row alignment...")
    output4 = sbp.sxm_to_gwy(
        input_file,
        output_file=input_file.with_stem(f'{input_file.stem}_noalign'),
        align_rows=None
    )
    
    print(f"\nCreated 4 versions with different row alignment methods")
    print()


def main():
    """Show all examples"""
    print("\n")
    print("*" * 70)
    print("SXM Batch Processor - Usage Examples")
    print("*" * 70)
    print()
    print("This script contains several usage examples.")
    print("Uncomment the examples you want to run in the main() function.")
    print()
    
    # Uncomment the examples you want to run:
    # example_basic_usage()
    # example_custom_processing()
    # example_single_file()
    # example_verify_gwy_files()
    # example_find_files()
    # example_different_row_alignment_methods()
    
    print("To use these examples:")
    print("1. Edit this file and uncomment the example you want to run")
    print("2. Replace 'path/to/your/...' with your actual data paths")
    print("3. Run: python3 usage_examples.py")
    print()


if __name__ == "__main__":
    main()
