import numpy as np
import cv2  # For reading and writing TIFF images (OpenCV is good for this)
import matplotlib.pyplot as plt # For plotting
import time

# --- Parameters (from Mathematica code) ---
lam = 0.532  # measured in micrometers (wavelength)
pix = 3.45   # measured in micrometers (pixel size)
zf = 67225   # replace number with focus distance in micrometers
zf_values = [65225, 66225, 67225, 68225, 69225,70000,80000] # Example zf values

n = 1 # Number of iterations for the Fresnel transform (rn just for performace)

# --- Function Definitions ---

# Fresnel function (Angular Spectrum Method)
def fresnel(z, lambda_val, image_array, pix_size_orig):
    """
    Computes the Fresnel propagation of an image using the Angular Spectrum Method.
    The 'image_array' here is assumed to be the complex field immediately after the hologram plane.
    """
    size_y, size_x = image_array.shape # Note: numpy shape is (rows, cols) -> (y, x)

    # 1. FFT of the input field
    U0_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image_array)))

    # 2. Define spatial frequencies
    # These are in cycles/micrometer. Multiply by 2*pi to get radians/micrometer.
    fx = np.fft.fftfreq(size_x, d=pix_size_orig)
    fy = np.fft.fftfreq(size_y, d=pix_size_orig)
    FX, FY = np.meshgrid(fx, fy, indexing='xy') # 'xy' for (x,y) grid matching image

    # 3. Calculate the propagation kernel H(fx, fy)
    k = 2 * np.pi / lambda_val
    
    # Argument of the square root: k_z^2 = k^2 - k_x^2 - k_y^2
    # k_x = 2*pi*fx, k_y = 2*pi*fy
    arg_sqrt = k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2
    
    # Handle evanescent waves: where arg_sqrt < 0, set k_z to 0 (or filter out)
    # np.maximum(0, arg_sqrt) ensures that we don't take the sqrt of negative numbers
    # for the propagating part.
    k_z = np.sqrt(np.maximum(0, arg_sqrt))
    
    H = np.exp(1j * k_z * z)
    
    # 4. Multiply by kernel in Fourier space
    U_fft = U0_fft * H
    
    # 5. Inverse FFT to get the propagated field
    propagated_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U_fft)))
    
    return propagated_field


# --- Main Program ---

start_time = time.time()

try:
    ref_image_raw = cv2.imread("/home/berg/Documents/git/speedy-DIH/python_port/dust_hologram_blank.tiff", cv2.IMREAD_GRAYSCALE)
    raw_image_raw = cv2.imread("/home/berg/Documents/git/speedy-DIH/python_port/dust_hologram.tiff", cv2.IMREAD_GRAYSCALE)

    if ref_image_raw is None or raw_image_raw is None:
        raise FileNotFoundError("Make sure 'dust_hologram_blank.tiff' and 'dust_hologram.tiff' exist in the specified path.")

    # Convert images to complex floating point for calculations
    # Ensure they are float and, if they are intensity images, they might need
    # to be converted to complex if they represent complex-valued holograms.
    # Assuming they are intensity images (real values) initially, we'll convert them to complex
    # as the Fresnel transform operates on complex fields.
    ref_image = ref_image_raw.astype(np.complex128)
    raw_image = raw_image_raw.astype(np.complex128)

except Exception as e:
    print(f"Error loading images: {e}")
    print("Please ensure image files are present and valid TIFF files at the specified path.")
    exit()

size_y, size_x = ref_image.shape # Corrected: numpy shape is (rows, cols)

# 2. Perform calculations (corresponds to 'con' in Mathematica)
# The 'con' variable in Mathematica: rawImage / refImage^2
# This is element-wise division.
con = raw_image / (ref_image**2)

# --- Physical Scaling Calculation ---
# This is where we understand why the object appears smaller.
# The reconstructed pixel size in the angular spectrum method is the same as the input pixel size.
# However, the physical extent of the *propagated field* remains the same, but the object itself
# disperses, so it appears "smaller" relative to the total fixed field of view.

# Let's consider the input field of view:
input_fov_x = size_x * pix
input_fov_y = size_y * pix
print(f"Input Field of View: {input_fov_x:.2f} µm x {input_fov_y:.2f} µm")
print(f"Original Pixel Size: {pix} µm")

# In the angular spectrum method, the output pixel size is the same as the input pixel size.
# "shrinking" observed is due to the diffraction spreading the light over the *fixed*
# physical dimensions of the output plane.
# The *object* itself spreads out, making it occupy more physical space,
# but because the display grid is fixed, it appears "smaller" in terms of pixels.

# --- Reconstruction Loop (demonstrating the effect) ---
print("\n--- Performing Reconstruction with Varying zf ---")
#try a few zf values to observe the effect


fig, axes = plt.subplots(1, len(zf_values), figsize=(4 * len(zf_values), 6))
if len(zf_values) == 1: # Handle case of single subplot
    axes = [axes]

for i, current_zf in enumerate(zf_values):
    print(f"Reconstructing for zf = {current_zf} µm...")
    # Call the angular spectrum fresnel
    reconstructed_field = fresnel(current_zf, lam, con, pix)
    reconstructed_intensity = np.abs(reconstructed_field)**2

    # Display the image
    ax = axes[i]
    ax.imshow(reconstructed_intensity, cmap='gray')
    ax.set_title(f'zf = {current_zf} µm')
    ax.axis('off')

end_time = time.time()
print(f"CPU Total computation time: {end_time - start_time:.2f} seconds.")

    # Optionally, print some stats
    # You'll notice the object gets 'fainter' and spreads out, taking up less distinct pixel area.
    # This is the "shrinking" you perceive.

plt.suptitle("Reconstructed Images at Different zf values")
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.show()




# # --- Final Reconstruction for the specified zf ---
# print(f"\nPerforming final reconstruction for zf = {zf} µm (as per original parameter)...")
# reconstructed_field_final = fresnel(zf, lam, con, pix)
# reconstructed_intensity_final = np.abs(reconstructed_field_final)**2

# plt.figure(figsize=(8, 8))
# plt.imshow(reconstructed_intensity_final, cmap='gray')
# plt.title(f'Final Reconstructed Image (zf={zf} µm)')
# plt.colorbar(label='Intensity')
# plt.axis('off')
# plt.show()

# Optional: Save the reconstructed image
# Normalization might be needed if you want to save it as an 8-bit or 16-bit image
# max_val = np.max(reconstructed_intensity_final)
# if max_val > 0:
#     normalized_image = (reconstructed_intensity_final / max_val * 255).astype(np.uint8)
#     cv2.imwrite("reconstructed_image_final.tiff", normalized_image)
# else:
#     print("Warning: Reconstructed image is all zeros, not saving.")