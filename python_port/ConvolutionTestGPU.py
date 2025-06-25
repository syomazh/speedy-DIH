import numpy as np
import cv2 # For reading and writing TIFF images (OpenCV is good for this)
import matplotlib.pyplot as plt # For plotting
import time
import cupy as cp # Import CuPy

# --- Parameters (from Mathematica code) ---
lam = 0.532  # measured in micrometers (wavelength)
pix = 3.45   # measured in micrometers (pixel size)
zf = 67225   # replace number with focus distance in micrometers
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

start_time = time.time()  # Start timing

# 1. Import Images
# Use OpenCV (cv2) to import TIFF files. It reads images as NumPy arrays.
try:
    # Use absolute paths for Linux compatibility
    ref_image_raw = cv2.imread("/home/berg/Documents/git/speedy-DIH/python_port/dust_hologram_blank.tiff", cv2.IMREAD_GRAYSCALE)
    raw_image_raw = cv2.imread("/home/berg/Documents/git/speedy-DIH/python_port/dust_hologram.tiff", cv2.IMREAD_GRAYSCALE)

    if ref_image_raw is None or raw_image_raw is None:
        raise FileNotFoundError("Make sure 'refDat.tiff' and 'rawDat.tiff' exist in the script directory.")

    # Convert images to complex floating point for calculations
    # and then transfer them to the GPU using cp.asarray
    ref_image_cpu = ref_image_raw.astype(np.complex128)
    raw_image_cpu = raw_image_raw.astype(np.complex128)

    ref_image_gpu = cp.asarray(ref_image_cpu)
    raw_image_gpu = cp.asarray(raw_image_cpu)

except Exception as e:
    print(f"Error loading images: {e}")
    print("Please ensure 'refDat.tiff' and 'rawDat.tiff' are present and valid TIFF files.")
    exit()

# Get image dimensions (assuming both images have the same dimensions)
size_x, size_y = ref_image_gpu.shape # Now getting shape from GPU array

# 2. Perform calculations (corresponds to 'con' in Mathematica) on GPU
con_gpu = raw_image_gpu / (ref_image_gpu**2)

# 3. Apply Fresnel transform and display result on GPU
# reconstructed_image_gpu = cp.abs(fresnel(zf, lam, con_gpu))**2
for i in range(n):
    print(f"Iteration {i+1}: Processing Fresnel transform...")
    reconstructed_image_gpu = cp.abs(fresnel((i+1)*10000, lam, con_gpu))**2




# Transfer the final result back to CPU for plotting with matplotlib
reconstructed_image_cpu = cp.asnumpy(reconstructed_image_gpu)

end_time = time.time()  # End timing

print(f"Computation done. Time taken: {end_time - start_time:.2f} seconds.")

# Display the image using matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(reconstructed_image_cpu, cmap='gray') # Use 'gray' colormap for intensity images
plt.title(f'Reconstructed Image (zf={zf} µm)')
plt.colorbar(label='Intensity')
plt.axis('off') # Turn off axis ticks and labels
plt.show()

# Optional: Save the reconstructed image (save from CPU array)
# cv2.imwrite("reconstructed_image.tiff", reconstructed_image_cpu.astype(np.float32))