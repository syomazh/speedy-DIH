import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import cupy as cp

# --- Parameters (from Mathematica code) ---
lam = 0.532  # measured in micrometers (wavelength)
pix = 3.45   # measured in micrometers (pixel size)
zf = 68900   # replace number with focus distance in micrometers
zf_values = [10000, 40000, 50000, 60000, 67225, 68900, 70000, 800000, 90000] # Sorted for better progression


# n = 1 # Not used in this version of fresnel, can remove or keep for reference

# --- Function Definitions ---

# Fresnel function
def fresnel(z, lambda_val, image_array_gpu):
    """
    Computes the Fresnel propagation of an image on the GPU using CuPy.
    Equivalent to Mathematica's Fresnel function in the context of this code.
    """
    size_y, size_x = image_array_gpu.shape # CuPy shape is (rows, cols)
    
    halfSize_x = size_x // 2
    halfSize_y = size_y // 2

    # Coordinate generation matching the FFT convention
    # These coordinates represent the physical space *before* the FFT
    # and define the physical extent of the input plane.
    x_coords = cp.arange(-halfSize_x, size_x - halfSize_x) * pix
    y_coords = cp.arange(-halfSize_y, size_y - halfSize_y) * pix
    
    X, Y = cp.meshgrid(x_coords, y_coords, indexing='xy') # 'xy' for Cartesian indexing (matching numpy's default)
    
    exp_term = cp.exp(1j * cp.pi / (lambda_val * z) * (X**2 + Y**2))
    
    transformed_image_gpu = image_array_gpu * exp_term
    
    # Perform 2D FFT
    fft_result_gpu = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(transformed_image_gpu)))
    
    # Scale factor for the Fresnel transform
    scale_factor = (1j / (lambda_val * z)) * cp.exp(1j * (2 * cp.pi / lambda_val) * z)
    
    return scale_factor * fft_result_gpu

# --- Main Program ---

start_time = time.time()

try:
    ref_image_raw = cv2.imread("/home/berg/Documents/git/speedy-DIH/test_files/dust_hologram_blank.tiff", cv2.IMREAD_GRAYSCALE)
    raw_image_raw = cv2.imread("/home/berg/Documents/git/speedy-DIH/test_files/dust_hologram.tiff", cv2.IMREAD_GRAYSCALE)

    if ref_image_raw is None or raw_image_raw is None:
        raise FileNotFoundError("Make sure 'dust_hologram_blank.tiff' and 'dust_hologram.tiff' exist in the specified path.")

    # Convert images to complex floating point for calculations
    # Transfer NumPy arrays to CuPy arrays on the GPU
    ref_image = cp.asarray(ref_image_raw.astype(cp.complex128))
    raw_image = cp.asarray(raw_image_raw.astype(cp.complex128))

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
con = raw_image / (ref_image**2)

# --- Reconstruction Loop ---
print("\n--- Performing Reconstruction with Varying zf (Normalized View) ---")

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
    # Calculate the effective pixel size in the reconstructed plane
    # P_rec = lambda * z / (N * original_pixel_size)
    # The term 'lambda * z' is related to the effective focal length of the free space propagation
    # and also determines the "zoom" factor.
    
    # The new effective pixel size in the reconstructed image
    # Note: In a true FFT-based Fresnel, the effective pixel size in the output
    # space domain is given by lambda * z / (N * original_pixel_size).
    # We want to know how many of these new pixels correspond to our original input FOV.
    
    # Calculate the physical size of the output plane in the FFT (reconstructed_intensity_cpu)
    # The dimensions of the frequency domain result are (lambda * z) / (N * pix) per pixel.
    # The total physical extent of the *reconstructed* image (the full FFT result) is:
    # Full_output_FOV_x = (lambda * current_zf / pix)
    # Full_output_FOV_y = (lambda * current_zf / pix)
    # This is for a N x N FFT.
    
    # Let's verify the actual "pixel size" of the reconstructed image
    # For a square FFT, the spatial frequency step is df = 1 / (N * pix).
    # The physical extent of the output in spatial domain is (N * lambda * z * df) = (N * lambda * z) / (N * pix) = (lambda * z) / pix
    
    # So, the original image (size_x * pix) is mapped into a new 'pixel grid'
    # where the full extent of the N x N output array corresponds to (lambda * z) / pix physically.
    # The "physical size per pixel" in the reconstructed image is then:
    effective_rec_pix_size_x = (lam * current_zf) / (original_size_x * pix)
    effective_rec_pix_size_y = (lam * current_zf) / (original_size_y * pix)

    # Now, calculate how many of *these new pixels* are needed to cover the original input FOV.
    target_pixels_x = round(input_fov_x / effective_rec_pix_size_x)
    target_pixels_y = round(input_fov_y / effective_rec_pix_size_y)

    # Ensure target_pixels are even for easy centering, and not larger than original image
    target_pixels_x = min(original_size_x, int(target_pixels_x))
    target_pixels_y = min(original_size_y, int(target_pixels_y))
    
    if target_pixels_x % 2 != 0: target_pixels_x -= 1
    if target_pixels_y % 2 != 0: target_pixels_y -= 1

    # Calculate the cropping region to get the central 'target_pixels' square
    center_y, center_x = reconstructed_intensity_cpu.shape[0] // 2, reconstructed_intensity_cpu.shape[1] // 2
    
    start_x = center_x - target_pixels_x // 2
    end_x = center_x + target_pixels_x // 2
    start_y = center_y - target_pixels_y // 2
    end_y = center_y + target_pixels_y // 2
    
    # Ensure crop boundaries are within image dimensions
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(reconstructed_intensity_cpu.shape[1], end_x)
    end_y = min(reconstructed_intensity_cpu.shape[0], end_y)

    cropped_intensity = reconstructed_intensity_cpu[start_y:end_y, start_x:end_x]
    
    # Finally, resize the cropped region back to the original image display size
    # This ensures each subplot has the same visual dimensions.
    # We use INTER_AREA for shrinking, INTER_LINEAR or INTER_CUBIC for enlarging.
    # Since we are usually shrinking the high-zf images, INTER_AREA is a good choice.
    final_display_image = cv2.resize(cropped_intensity, 
                                     (original_size_x, original_size_y), 
                                     interpolation=cv2.INTER_LINEAR) # Linear is fine here as we are filling the display

    # Normalize intensity for display consistency (0-1 range)
    final_display_image = (final_display_image - final_display_image.min()) / \
                          (final_display_image.max() - final_display_image.min())

    ax = axes[i]
    ax.imshow(final_display_image, cmap='gray')
    ax.set_title(f'zf = {current_zf} µm')
    ax.axis('off')

comp_end_time = time.time()
print(f"GPU Total computation time: {comp_end_time - start_time:.2f} seconds.")

plt.suptitle("Reconstructed Images at Different zf values - Cropped and Resized View")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()