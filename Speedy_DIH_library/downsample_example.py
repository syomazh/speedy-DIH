"""
Example demonstrating the downsampling feature for reducing compute time.

The downsample_factor parameter allows you to reduce image resolution:
- downsample_factor=1 (default): No downsampling, full resolution
- downsample_factor=2: Half resolution (1/4 the pixels)
- downsample_factor=4: Quarter resolution (1/16 the pixels)
- downsample_factor=8: 1/8 resolution (1/64 the pixels)

Lower resolution = faster computation but less detail.
"""

from speedyDIH import SpeedyDIH
import time

# Initialize with your parameters
dih = SpeedyDIH(wavelength=0.532, pixel_size=3.45)

# File paths
ref_path = "../test_files/refDat.tiff"
raw_path = "../test_files/rawDat.tiff"

# Example 1: Full resolution (default)
print("=" * 60)
print("Example 1: Full Resolution")
print("=" * 60)
start = time.time()
ref_full, raw_full = dih.load_images(ref_path, raw_path, downsample_factor=1)
print(f"Time taken: {time.time() - start:.3f} seconds\n")

# Example 2: Half resolution (2x faster)
print("=" * 60)
print("Example 2: Half Resolution (2x downsample)")
print("=" * 60)
start = time.time()
ref_half, raw_half = dih.load_images(ref_path, raw_path, downsample_factor=2)
print(f"Time taken: {time.time() - start:.3f} seconds\n")

# Example 3: Quarter resolution (4x faster)
print("=" * 60)
print("Example 3: Quarter Resolution (4x downsample)")
print("=" * 60)
start = time.time()
ref_quarter, raw_quarter = dih.load_images(ref_path, raw_path, downsample_factor=4)
print(f"Time taken: {time.time() - start:.3f} seconds\n")

# Example 4: Using downsampling with focus finding
print("=" * 60)
print("Example 4: Focus Finding with Downsampling")
print("=" * 60)
distance_range = [15.0, 20.0, 25.0, 30.0, 35.0]

# Fast preliminary search with downsampled images
print("\nFast search with 4x downsampling:")
start = time.time()
focus_distance = dih.find_focus(ref_path, raw_path, distance_range, downsample_factor=4)
print(f"Optimal focus found at: {focus_distance:.2f} mm")
print(f"Time taken: {time.time() - start:.3f} seconds")

# Example 5: Using downsampling with hierarchical search
print("\n" + "=" * 60)
print("Example 5: Hierarchical Search with Downsampling")
print("=" * 60)
start = time.time()
focus_hierarchical = dih.find_focus_hierarchical(
    ref_path, raw_path, 
    min_distance=15.0, 
    max_distance=35.0,
    n_points=10,
    downsample_factor=2  # Use half resolution for faster search
)
print(f"Optimal focus found at: {focus_hierarchical:.2f} mm")
print(f"Time taken: {time.time() - start:.3f} seconds")

print("\n" + "=" * 60)
print("Downsampling reduces computation time significantly!")
print("Use it for quick previews or initial focus searches.")
print("=" * 60)
