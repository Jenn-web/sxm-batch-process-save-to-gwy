"""
Test script for sxm_batch_processor module

This script demonstrates the basic functionality without requiring actual .sxm files.
It tests individual processing functions with synthetic data.
"""

import numpy as np
import sxm_batch_processor as sbp

def test_level_by_mean_plane():
    """Test mean plane subtraction"""
    print("Testing level_by_mean_plane...")
    
    # Create synthetic data with a plane
    x, y = np.meshgrid(np.arange(100), np.arange(100))
    plane = 2 * x + 3 * y + 10
    noise = np.random.randn(100, 100) * 0.1
    data = plane + noise
    
    # Level the data
    leveled = sbp.level_by_mean_plane(data)
    
    # Check that the mean is close to zero
    assert abs(np.mean(leveled)) < 0.5, "Mean should be close to zero after leveling"
    print("  ✓ Mean plane subtraction works correctly")


def test_align_rows_mean():
    """Test row alignment by mean"""
    print("Testing align_rows_mean...")
    
    # Create data with different row offsets
    data = np.random.randn(100, 100)
    for i in range(100):
        data[i, :] += i * 0.1  # Add offset to each row
    
    # Align rows
    aligned = sbp.align_rows_mean(data)
    
    # Check that row means are more uniform
    original_std = np.std([np.mean(data[i, :]) for i in range(100)])
    aligned_std = np.std([np.mean(aligned[i, :]) for i in range(100)])
    assert aligned_std < original_std, "Row alignment should reduce variation in row means"
    print("  ✓ Row alignment by mean works correctly")


def test_align_rows_median():
    """Test row alignment by median"""
    print("Testing align_rows_median...")
    
    # Create data with different row offsets
    data = np.random.randn(100, 100)
    for i in range(100):
        data[i, :] += i * 0.1
    
    # Align rows
    aligned = sbp.align_rows_median(data)
    
    # Check that row medians are more uniform
    original_std = np.std([np.median(data[i, :]) for i in range(100)])
    aligned_std = np.std([np.median(aligned[i, :]) for i in range(100)])
    assert aligned_std < original_std, "Row alignment should reduce variation in row medians"
    print("  ✓ Row alignment by median works correctly")


def test_parabolic_subtraction():
    """Test parabolic background subtraction"""
    print("Testing parabolic_subtraction...")
    
    # Create data with parabolic background
    x = np.arange(100)
    parabola = 0.01 * x**2 + 2 * x + 10
    data = np.tile(parabola, (100, 1)) + np.random.randn(100, 100) * 0.1
    
    # Remove parabolic background
    corrected = sbp.parabolic_subtraction(data, axis=1)
    
    # Check that the parabolic trend is reduced
    original_range = np.ptp(np.mean(data, axis=0))
    corrected_range = np.ptp(np.mean(corrected, axis=0))
    assert corrected_range < original_range, "Parabolic subtraction should reduce trend"
    print("  ✓ Parabolic subtraction works correctly")


def test_remove_scars():
    """Test scar removal"""
    print("Testing remove_scars...")
    
    # Create data with a scar (anomalous row)
    data = np.random.randn(100, 100)
    data[50, :] += 10  # Add a strong horizontal scar
    
    # Remove scars
    corrected = sbp.remove_scars(data, threshold=2.0)
    
    # Check that the scar is reduced
    original_scar_intensity = np.mean(data[50, :])
    corrected_scar_intensity = np.mean(corrected[50, :])
    assert abs(corrected_scar_intensity) < abs(original_scar_intensity), "Scar should be reduced"
    print("  ✓ Scar removal works correctly")


def test_process_data():
    """Test complete processing pipeline"""
    print("Testing process_data...")
    
    # Create synthetic data with various issues
    x, y = np.meshgrid(np.arange(100), np.arange(100))
    data = (2 * x + 3 * y + 10).astype(np.float64)  # Plane (ensure float)
    data += 0.01 * x**2  # Parabolic background
    for i in range(100):
        data[i, :] += i * 0.1  # Row offsets
    data[50, :] += 5  # Scar
    data += np.random.randn(100, 100) * 0.1  # Noise
    
    # Process with all options
    processed = sbp.process_data(
        data,
        level_plane=True,
        align_rows='mean',
        parabolic_sub=True,
        remove_scars_flag=True
    )
    
    # Check that data is processed (shape unchanged)
    assert processed.shape == data.shape, "Shape should be preserved"
    print("  ✓ Complete processing pipeline works correctly")


def test_find_sxm_files():
    """Test file finding functionality"""
    print("Testing find_sxm_files...")
    
    # Test with current directory (should find no .sxm files)
    import os
    files = sbp.find_sxm_files(os.getcwd())
    assert isinstance(files, list), "Should return a list"
    print(f"  ✓ File finding works correctly (found {len(files)} .sxm files)")


def main():
    print("=" * 60)
    print("SXM Batch Processor - Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_level_by_mean_plane,
        test_align_rows_mean,
        test_align_rows_median,
        test_parabolic_subtraction,
        test_remove_scars,
        test_process_data,
        test_find_sxm_files,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
