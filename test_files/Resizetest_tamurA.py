import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import cupy as cp
# No need for scipy.stats.kurtosis for this metric

# --- Parameters (from Mathematica code) ---
lam = 0.532  # measured in micrometers (wavelength)
pix = 3.45   # measured in micrometers (pixel size)
zf = 68900   # replace number with focus distance in micrometers
zf_values = [10000, 40000, 50000, 60000, 67225, 68900, 70000, 80000, 90000] # Sorted for better progression

# --- Function Definitions ---

# Fresnel function (from your original code - already optimized with CuPy)
def fresnel(z, lambda_val, image_array_gpu):
    """
    Computes the Fresnel propagation of an image on the GPU using CuPy.
    Equivalent to Mathematica's Fresnel function in the context of this code.
    """
    size_y, size_x = image_array_gpu.shape
    
    halfSize_x = size_x // 2
    halfSize_y = size_y // 2

    x_coords = cp.arange(-halfSize_x, size_x - halfSize_x) * pix
    y_coords = cp.arange(-halfSize_y, size_y - halfSize_y) * pix
    
    X, Y = cp.meshgrid(x_coords, y_coords, indexing='xy')
    
    exp_term = cp.exp(1j * cp.pi / (lambda_val * z) * (X**2 + Y**2))
    
    transformed_image_gpu = image_array_gpu * exp_term
    
    fft_result_gpu = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(transformed_image_gpu)))
    
    scale_factor = (1j / (lambda_val * z)) * cp.exp(1j * (2 * cp.pi / lambda_val) * z)
    
    return scale_factor * fft_result_gpu

# --- New Metric Function: Coefficient of Variation (CV) ---

def calculate_cv(image_gpu):
    """
    Calculates the Coefficient of Variation (CV) for a given image on the GPU.
    CV = standard deviation / mean
    
    Args:
        image_gpu (cupy.ndarray): Grayscale image (CuPy array) for which to calculate CV.
                                   Expected to be float type.
    
    Returns:
        float: The Coefficient of Variation value. Returns 0.0 if mean is zero to avoid division by zero.
    """
    mean_pixels = cp.mean(image_gpu)
    
    if mean_pixels == 0:
        return 0.0 # Avoid division by zero
        
    std_pixels = cp.std(image_gpu)
    
    return (std_pixels / mean_pixels).get() # .get() to move result to CPU

# --- Main Program (Modified to include CV metric) ---

start_time = time.time()

try:
    ref_image_raw = cv2.imread("/home/berg/Documents/git/speedy-DIH/test_files/dust_hologram_blank.tiff", cv2.IMREAD_GRAYSCALE)
    raw_image_raw = cv2.imread("/home/berg/Documents/git/speedy-DIH/test_files/dust_hologram.tiff", cv2.IMREAD_GRAYSCALE)

    if ref_image_raw is None or raw_image_raw is None:
        raise FileNotFoundError("Make sure 'dust_hologram_blank.tiff' and 'dust_hologram.tiff' exist in the specified path.")

    # Convert images to complex floating point for calculations
    # Transfer NumPy arrays to CuPy arrays on the GPU
    # Ensure raw_image_raw is float before converting to complex
    ref_image_gpu = cp.asarray(ref_image_raw.astype(cp.float64)) # Use float64 for better precision if required
    raw_image_gpu = cp.asarray(raw_image_raw.astype(cp.float64))

    # For Fresnel, we need complex type:
    ref_image_complex_gpu = ref_image_gpu.astype(cp.complex128)
    raw_image_complex_gpu = raw_image_gpu.astype(cp.complex128)

except Exception as e:
    print(f"Error loading images: {e}")
    print("Please ensure image files are present and valid TIFF files at the specified path.")
    exit()

# Get original image dimensions and calculate original input FOV
original_size_y, original_size_x = ref_image_raw.shape
input_fov_x = original_size_x * pix
input_fov_y = original_size_y * pix
print(f"Original Input Field of View: {input_fov_x:.2f} µm x {input_fov_y:.2f} µm")
print(f"Original Pixel Size: {pix} µm")

# 2. Perform calculations (corresponds to 'con' in Mathematica)
con = raw_image_complex_gpu / (ref_image_complex_gpu**2)

# --- Reconstruction Loop ---
print("\n--- Performing Reconstruction with Varying zf (Normalized View) ---")

# Store CV results for each zf value
cv_results = []

fig, axes = plt.subplots(1, len(zf_values), figsize=(4 * len(zf_values), 6))
if len(zf_values) == 1: # Handle case of single subplot
    axes = [axes]

for i, current_zf in enumerate(zf_values):
    print(f"Reconstructing for zf = {current_zf} µm...")
    reconstructed_field = fresnel(current_zf, lam, con)
    reconstructed_intensity = cp.abs(reconstructed_field)**2
    
    # Bring the CuPy array back to the CPU for processing and plotting
    reconstructed_intensity_cpu = cp.asnumpy(reconstructed_intensity)
    
    # --- Dynamic Cropping Calculation ---
    effective_rec_pix_size_x = (lam * current_zf) / (original_size_x * pix)
    effective_rec_pix_size_y = (lam * current_zf) / (original_size_y * pix)

    target_pixels_x = round(input_fov_x / effective_rec_pix_size_x)
    target_pixels_y = round(input_fov_y / effective_rec_pix_size_y)

    target_pixels_x = min(original_size_x, int(target_pixels_x))
    target_pixels_y = min(original_size_y, int(target_pixels_y))
    
    if target_pixels_x % 2 != 0: target_pixels_x -= 1
    if target_pixels_y % 2 != 0: target_pixels_y -= 1

    center_y, center_x = reconstructed_intensity_cpu.shape[0] // 2, reconstructed_intensity_cpu.shape[1] // 2
    
    start_x = center_x - target_pixels_x // 2
    end_x = center_x + target_pixels_x // 2
    start_y = center_y - target_pixels_y // 2
    end_y = center_y + target_pixels_y // 2
    
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(reconstructed_intensity_cpu.shape[1], end_x)
    end_y = min(reconstructed_intensity_cpu.shape[0], end_y)

    cropped_intensity_cpu = reconstructed_intensity_cpu[start_y:end_y, start_x:end_x]
    
    final_display_image = cv2.resize(cropped_intensity_cpu, 
                                     (original_size_x, original_size_y), 
                                     interpolation=cv2.INTER_LINEAR)

    # Normalize intensity for display consistency (0-1 range)
    # The CV calculation works best on the actual intensity values, not normalized to 0-1 for display.
    # We'll use the 'cropped_intensity_cpu' or its CuPy equivalent directly for CV calculation.
    # It's important that the image for CV calculation is the raw intensity, not scaled to 0-255 or 0-1.
    
    # Convert the cropped intensity back to CuPy for CV calculation
    image_for_cv_gpu = cp.asarray(cropped_intensity_cpu, dtype=cp.float32) # Using float32 for CV calc

    # --- Calculate Coefficient of Variation (CV) ---
    
    cv_calc_start = time.time()
    current_cv = calculate_cv(image_for_cv_gpu)
    cv_calc_end = time.time()
    
    cv_results.append({
        'zf': current_zf,
        'CV': current_cv
    })
    print(f"  CV (zf={current_zf}): {current_cv:.6f}")
    print(f"  CV calculation time: {cv_calc_end - cv_calc_start:.6f} seconds.")

    # Normalize final_display_image for display
    display_image_normalized = (final_display_image - final_display_image.min()) / \
                               (final_display_image.max() - final_display_image.min() + 1e-9)

    ax = axes[i]
    ax.imshow(display_image_normalized, cmap='gray')
    ax.set_title(f'zf = {current_zf} µm\nCV: {current_cv:.4f}')
    ax.axis('off')

comp_end_time = time.time()
print(f"\nGPU Total computation time (including CV metric): {comp_end_time - start_time:.2f} seconds.")

plt.suptitle("Reconstructed Images and Coefficient of Variation at Different zf values")
plt.tight_layout(rect=[0, 0.03, 1, 0.9]) # Adjust layout to prevent title overlap
plt.show()

print("\n--- Summary of Coefficient of Variation for each zf ---")
for result in cv_results:
    print(f"zf={result['zf']} µm: CV={result['CV']:.6f}")