import cupy as cp # Import CuPy
import cv2
import matplotlib.pyplot as plt
import time

# --- Parameters (from Mathematica code) ---
lam = 0.532  # measured in micrometers (wavelength)
pix = 3.45   # measured in micrometers (pixel size)
zf = 67225   # replace number with focus distance in micrometers

zf_values = [40000, 50000, 60000, 70000, 80000,90000,10000,67225 ,68900] # Example zf values
#zf_values = [67225,68000, 68900, 68925, 68950,] # Example zf values
n = 1 # Number of iterations for the Fresnel transform (rn just for performace)

# --- Function Definitions ---

# Fresnel function (Angular Spectrum Method)
def fresnel(z, lambda_val, image_array, pix_size_orig):
    """
    Computes the Fresnel propagation of an image using the Angular Spectrum Method.
    The 'image_array' here is assumed to be the complex field immediately after the hologram plane.
    """
    size_y, size_x = image_array.shape # Note: cupy shape is (rows, cols) -> (y, x)

    # 1. FFT of the input field
    # Use cp.fft for GPU-accelerated FFT
    U0_fft = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(image_array)))

    # 2. Define spatial frequencies
    # These are in cycles/micrometer. Multiply by 2*pi to get radians/micrometer.
    # Use cp.fft.fftfreq and cp.meshgrid for GPU arrays
    fx = cp.fft.fftfreq(size_x, d=pix_size_orig)
    fy = cp.fft.fftfreq(size_y, d=pix_size_orig)
    FX, FY = cp.meshgrid(fx, fy, indexing='xy') # 'xy' for (x,y) grid matching image

    # 3. Calculate the propagation kernel H(fx, fy)
    k = 2 * cp.pi / lambda_val
    
    # Argument of the square root: k_z^2 = k^2 - k_x^2 - k_y^2
    # k_x = 2*pi*fx, k_y = 2*pi*fy
    arg_sqrt = k**2 - (2*cp.pi*FX)**2 - (2*cp.pi*FY)**2
    
    # Handle evanescent waves: where arg_sqrt < 0, set k_z to 0 (or filter out)
    # cp.maximum(0, arg_sqrt) ensures that we don't take the sqrt of negative numbers
    # for the propagating part.
    k_z = cp.sqrt(cp.maximum(0, arg_sqrt))
    
    H = cp.exp(1j * k_z * z)
    
    # 4. Multiply by kernel in Fourier space
    U_fft = U0_fft * H
    
    # 5. Inverse FFT to get the propagated field
    propagated_field = cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(U_fft)))
    
    return propagated_field


# --- Main Program ---

start_time = time.time()

try:
    ref_image_raw = cv2.imread("/home/berg/Documents/git/speedy-DIH/python_port/dust_hologram_blank.tiff", cv2.IMREAD_GRAYSCALE)
    raw_image_raw = cv2.imread("/home/berg/Documents/git/speedy-DIH/python_port/dust_hologram.tiff", cv2.IMREAD_GRAYSCALE)

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
    reconstructed_field = fresnel(current_zf, lam, con, pix)
    reconstructed_intensity = cp.abs(reconstructed_field)**2
    
    # To display with matplotlib, you need to bring the CuPy array back to the CPU
    ax = axes[i]
    ax.imshow(cp.asnumpy(reconstructed_intensity), cmap='gray')
    ax.set_title(f'zf = {current_zf} µm')
    ax.axis('off')

comp_end_time = time.time()
print(f"GPU Total computation time: {comp_end_time - start_time:.2f} seconds.")

plt.suptitle("Reconstructed Images at Different zf values - Angular Spectrum Method")
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