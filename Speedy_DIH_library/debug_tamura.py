import speedyDIH
import cupy as cp

# Test with your actual images
refImagePath = "/home/berg/Documents/git/speedy-DIH/test_files/dust_hologram_blank.tiff"
rawImagePath = "/home/berg/Documents/git/speedy-DIH/test_files/dust_hologram.tiff"

dih = speedyDIH.SpeedyDIH()

# Load images
ref_image, raw_image = dih.load_images(refImagePath, rawImagePath)

# Compute contrast
contrast = raw_image / (ref_image**2)

# Test at a few distances
test_distances = [10, 100, 1000, 5000, 10000]

print("Debugging Tamura calculation:")
print("="*60)

for distance in test_distances:
    # Reconstruct
    cached_coords = dih.processor.coord_cache.get_coordinates(contrast.shape)
    reconstructed_field = dih.processor.fresnel_propagation(contrast, distance, cached_coords)
    intensity = cp.abs(reconstructed_field)**2
    
    # Calculate statistics
    mean_val = float(cp.mean(intensity).get())
    std_val = float(cp.std(intensity).get())
    min_val = float(cp.min(intensity).get())
    max_val = float(cp.max(intensity).get())
    
    # Calculate Tamura
    tamura = dih.calculate_tamura(intensity)
    
    print(f"\nDistance: {distance} µm")
    print(f"  Mean: {mean_val:.6e}")
    print(f"  Std:  {std_val:.6e}")
    print(f"  Min:  {min_val:.6e}")
    print(f"  Max:  {max_val:.6e}")
    print(f"  Tamura (std/mean): {tamura:.6f}")
    
print("\n" + "="*60)
