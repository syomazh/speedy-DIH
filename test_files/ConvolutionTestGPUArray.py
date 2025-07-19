import numpy as np
import cv2 # For reading and writing TIFF images (OpenCV is good for this)
import matplotlib.pyplot as plt # For plotting
import time
import cupy as cp # Import CuPy

# --- Parameters (from Mathematica code) ---
lam = 0.532  # measured in micrometers (wavelength)
pix = 3.45   # measured in micrometers (pixel size)
zf = 68900   # replace number with focus distance in micrometers
zf_values = [] # Example zf values
zf_values = list(range(30000, 90001, 5000))  # Continue the list in steps of 2000 up to 200000
# zf_values = [60000,67225, 78225, ] # Example zf values


n = 1 # Number of iterations for the Fresnel transform (just for performance)

# --- Function Definitions ---

# Fresnel function
def fresnel(z, lambda_val, image_array_gpu):
    """
    Computes the Fresnel propagation of an image on the GPU using CuPy.
    Equivalent to Mathematica's Fresnel function in the context of this code.
    """
    size_x, size_y = image_array_gpu.shape
    
    # Mathematica's Range and Outer for coordinate generation
    # x values range from -halfSize to halfSize-1
    # y values range from -halfSizez to halfSizez-1
    # Note: Mathematica's Floor[size/2.0] gives an integer.
    # Python's // operator for integer division is equivalent to Floor for positive numbers.
    halfSize_x = size_x // 2
    halfSize_y = size_y // 2

    # Use cp.arange and cp.meshgrid for GPU arrays
    x_coords = cp.arange(-halfSize_x, size_x - halfSize_x) * pix
    y_coords = cp.arange(-halfSize_y, size_y - halfSize_y) * pix
    
    X, Y = cp.meshgrid(x_coords, y_coords, indexing='ij') # 'ij' for matrix indexing
    
    # The 'Exp' part of the Fresnel kernel - all operations now on GPU
    exp_term = cp.exp(1j * cp.pi / (lambda_val * z) * (X**2 + Y**2))
    
    # Element-wise multiplication of the image with the exponential term
    transformed_image_gpu = image_array_gpu * exp_term
    
    # 2D Fourier Transform using CuPy's FFT
    # cp.fft.fftshift moves zero-frequency component to center
    # cp.fft.fft2 performs 2D FFT
    # cp.fft.ifftshift moves zero-frequency component back
    fft_result_gpu = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(transformed_image_gpu)))
    
    # Scale factor from Mathematica's Fresnel - also use CuPy's exp
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

size_y, size_x = ref_image.shape # Corrected: cupy shape is (rows, cols)

# 2. Perform calculations (corresponds to 'con' in Mathematica)
# All these operations now happen on the GPU with CuPy arrays
con = raw_image / (ref_image**2)

# --- Physical Scaling Calculation ---
input_fov_x = size_x * pix
input_fov_y = size_y * pix
print(f"Input Field of View: {input_fov_x:.2f} µm x {input_fov_y:.2f} µm")
print(f"Original Pixel Size: {pix} µm")

# --- Reconstruction Loop (demonstrating the effect) ---
print("\n--- Performing Reconstruction with Varying zf ---")


fig, axes = plt.subplots(1, len(zf_values), figsize=(4 * len(zf_values), 6))
if len(zf_values) == 1: # Handle case of single subplot
    axes = [axes]

for i, current_zf in enumerate(zf_values):
    print(f"Reconstructing for zf = {current_zf} µm...")
    # Call the angular spectrum fresnel
    reconstructed_field = fresnel(current_zf, lam, con)
    reconstructed_intensity = cp.abs(reconstructed_field)**2
    
    # To display with matplotlib, you need to bring the CuPy array back to the CPU
    ax = axes[i]
    ax.imshow(cp.asnumpy(reconstructed_intensity), cmap='gray')
    ax.set_title(f'zf = {current_zf} µm')
    ax.axis('off')

comp_end_time = time.time()
print(f"GPU Total computation time: {comp_end_time - start_time:.2f} seconds.")

plt.suptitle("Reconstructed Images at Different zf values - Convolution method")
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.show()


# # --- Final Reconstruction for the specified zf ---
# print(f"\nPerforming final reconstruction for zf = {zf} µm (as per original parameter)...")
# reconstructed_field_final = fresnel(zf, lam, con, pix)
# reconstructed_intensity_final = cp.abs(reconstructed_field_final)**2

# plt.figure(figsize=(8, 8))
# # Bring the final reconstructed image back to CPU for plotting
# plt.imshow(cp.asnumpy(reconstructed_intensity_final), cmap='gray')
# plt.title(f'Final Reconstructed Image (zf={zf} µm)')
# plt.colorbar(label='Intensity')
# plt.axis('off')
# plt.show()