import numpy as np
import cv2  # For reading and writing TIFF images (OpenCV is good for this)
import matplotlib.pyplot as plt # For plotting
import time

# --- Parameters ---
lam = 0.532  # measured in micrometers (wavelength)
pix = 3.45   # measured in micrometers (pixel size)
initial_zf = 72225  # Initial focus distance in micrometers, now used as a reference
z_step = 10000  # Step size for iterating through zf values
z_range_min = initial_zf - 5 * z_step # Start of the zf range for plotting
z_range_max = initial_zf + 5 * z_step # End of the zf range for plotting
num_z_steps = 20 # Number of steps in the zf range for plotting

# --- Function Definitions ---

def fresnel(z, lambda_val, image_array):
    """
    Computes the Fresnel propagation of an image.
    Equivalent to Mathematica's Fresnel function in the context of this code.
    """
    size_x, size_y = image_array.shape
    
    halfSize_x = size_x // 2
    halfSize_y = size_y // 2

    x_coords = np.arange(-halfSize_x, size_x - halfSize_x) * pix
    y_coords = np.arange(-halfSize_y, size_y - halfSize_y) * pix
    
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij') # 'ij' for matrix indexing
    
    exp_term = np.exp(1j * np.pi / (lambda_val * z) * (X**2 + Y**2))
    
    transformed_image = image_array * exp_term
    
    fft_result = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(transformed_image)))
    
    scale_factor = (1j / (lambda_val * z)) * np.exp(1j * (2 * np.pi / lambda_val) * z)
    
    return scale_factor * fft_result

def tamura_coefficient(image_array):
    """
    Calculates the Tamura coefficient for a given grayscale image.
    C = (standard deviation of pixels) / (mean of pixels)
    """
    if image_array.dtype != np.float64 and image_array.dtype != np.float32:
        image_array = image_array.astype(np.float64) # Ensure float type for calculations

    mean_pixels = np.mean(image_array)
    std_pixels = np.std(image_array)

    if mean_pixels == 0:
        return 0 # Avoid division by zero, though unlikely for an intensity image

    return std_pixels / mean_pixels

# --- Main Program ---

start_time = time.time()  # Start timing

# 1. Import Images
try:
    # Ensure 'dust_hologram_blank.tiff' and 'dust_hologram.tiff' are in the script directory
    ref_image_raw = cv2.imread("/home/berg/Documents/git/speedy-DIH/python_port/dust_hologram_blank.tiff", cv2.IMREAD_GRAYSCALE)
    raw_image_raw = cv2.imread("/home/berg/Documents/git/speedy-DIH/python_port/dust_hologram.tiff", cv2.IMREAD_GRAYSCALE)

    if ref_image_raw is None or raw_image_raw is None:
        raise FileNotFoundError("Make sure 'dust_hologram_blank.tiff' and 'dust_hologram.tiff' exist in the script directory.")

    ref_image = ref_image_raw.astype(np.complex128)
    raw_image = raw_image_raw.astype(np.complex128)

except Exception as e:
    print(f"Error loading images: {e}")
    print("Please ensure 'dust_hologram_blank.tiff' and 'dust_hologram.tiff' are present and valid TIFF files.")
    exit()

# Get image dimensions (assuming both images have the same dimensions)
size_x, size_y = ref_image.shape

# Perform calculations (corresponds to 'con' in Mathematica)
con = raw_image / (ref_image**2)

# --- Calculate Tamura Coefficient over a range of zf values ---
z_values = np.linspace(z_range_min, z_range_max, num_z_steps)
tamura_coefficients = []

print("\nCalculating Tamura Coefficients for various zf values...")
for i, zf_val in enumerate(z_values):
    print(f"Processing zf = {zf_val:.2f} µm ({i+1}/{num_z_steps})...")
    # Apply Fresnel transform for the current zf_val
    reconstructed_image_at_z = np.abs(fresnel(zf_val, lam, con))**2
    
    # Calculate Tamura Coefficient for the reconstructed image
    tamura_c = tamura_coefficient(reconstructed_image_at_z)
    tamura_coefficients.append(tamura_c)

end_time = time.time()  # End timing

print(f"\nAll computations done. Total time taken: {end_time - start_time:.2f} seconds.")

# --- Plotting the Tamura Coefficient vs zf ---
plt.figure(figsize=(10, 6))
plt.plot(z_values, tamura_coefficients, marker='o', linestyle='-', color='blue')
plt.title('Tamura Coefficient vs. Focus Distance (zf)')
plt.xlabel('Focus Distance, zf (µm)')
plt.ylabel('Tamura Coefficient (C)')
plt.grid(True)
plt.axvline(x=initial_zf, color='red', linestyle='--', label=f'Initial zf = {initial_zf} µm')
plt.legend()
plt.show()

# Display the reconstructed image at the initial zf (as a final example)
# You can uncomment this section if you still want to see one specific hologram image
print(f"\nDisplaying hologram at initial zf = {initial_zf} µm...")
reconstructed_image_initial_zf = np.abs(fresnel(initial_zf, lam, con))**2
tamura_c_initial_zf = tamura_coefficient(reconstructed_image_initial_zf)

plt.figure(figsize=(8, 8))
plt.imshow(reconstructed_image_initial_zf, cmap='gray')
plt.title(f'Reconstructed Image (zf={initial_zf} µm)\nTamura Coefficient: {tamura_c_initial_zf:.4f}')
plt.colorbar(label='Intensity')
plt.axis('off')
plt.show()