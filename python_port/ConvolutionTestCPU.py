import numpy as np
import cv2  # For reading and writing TIFF images (OpenCV is good for this)
import matplotlib.pyplot as plt # For plotting
import time

# --- Parameters (from Mathematica code) ---
lam = 0.532  # measured in micrometers (wavelength)
pix = 3.45   # measured in micrometers (pixel size)
zf = 67225   # replace number with focus distance in micrometers
n= 10# Number of iterations for the Fresnel transform (rn just for performace)

# --- Function Definitions ---

# Fresnel function
def fresnel(z, lambda_val, image_array):
    """
    Computes the Fresnel propagation of an image.
    Equivalent to Mathematica's Fresnel function in the context of this code.
    """
    size_x, size_y = image_array.shape
    
    # Mathematica's Range and Outer for coordinate generation
    # x values range from -halfSize to halfSize-1
    # y values range from -halfSizez to halfSizez-1
    # Note: Mathematica's Floor[size/2.0] gives an integer.
    # Python's // operator for integer division is equivalent to Floor for positive numbers.
    halfSize_x = size_x // 2
    halfSize_y = size_y // 2

    x_coords = np.arange(-halfSize_x, size_x - halfSize_x) * pix
    y_coords = np.arange(-halfSize_y, size_y - halfSize_y) * pix
    
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij') # 'ij' for matrix indexing
    
    # The 'Exp' part of the Fresnel kernel
    exp_term = np.exp(1j * np.pi / (lambda_val * z) * (X**2 + Y**2))
    
    # Element-wise multiplication of the image with the exponential term
    transformed_image = image_array * exp_term
    
    # 2D Fourier Transform
    # np.fft.fftshift moves zero-frequency component to center
    # np.fft.fft2 performs 2D FFT
    # np.fft.ifftshift moves zero-frequency component back
    fft_result = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(transformed_image)))
    
    # Scale factor from Mathematica's Fresnel
    scale_factor = (1j / (lambda_val * z)) * np.exp(1j * (2 * np.pi / lambda_val) * z)
    
    return scale_factor * fft_result

# --- Main Program ---

start_time = time.time()  # Start timing

# 1. Import Images
# Use OpenCV (cv2) to import TIFF files. It reads images as NumPy arrays.
# Ensure 'refDat.tiff' and 'rawDat.tiff' are in the same directory as your Python script
# or provide their full paths.
try:
    ref_image_raw = cv2.imread("/home/berg/Documents/git/speedy-DIH/python_port/dust_hologram_blank.tiff", cv2.IMREAD_GRAYSCALE)
    raw_image_raw = cv2.imread("/home/berg/Documents/git/speedy-DIH/python_port/dust_hologram.tiff", cv2.IMREAD_GRAYSCALE)

    if ref_image_raw is None or raw_image_raw is None:
        raise FileNotFoundError("Make sure 'refDat.tiff' and 'rawDat.tiff' exist in the script directory.")

    # Convert images to complex floating point for calculations
    # Mathematica often treats image data as numbers directly.
    # Ensure they are float and, if they are intensity images, they might need
    # to be converted to complex if they represent complex-valued holograms.
    # Assuming they are intensity images (real values) initially, we'll convert them to complex
    # as the Fresnel transform operates on complex fields.
    ref_image = ref_image_raw.astype(np.complex128)
    raw_image = raw_image_raw.astype(np.complex128)

except Exception as e:
    print(f"Error loading images: {e}")
    print("Please ensure 'refDat.tiff' and 'rawDat.tiff' are present and valid TIFF files.")
    exit()

# Get image dimensions (assuming both images have the same dimensions)
size_x, size_y = ref_image.shape

# 2. Perform calculations (corresponds to 'con' in Mathematica)
# The 'con' variable in Mathematica: rawImage / refImage^2
# This is element-wise division.
con = raw_image / (ref_image**2)

# 3. Apply Fresnel transform and display result
# Evaluate Abs[fresnel[zf, lam, con]^2]
for i in range(n):
    print(f"Iteration {i+1}: Processing Fresnel transform...")
    reconstructed_image = np.abs(fresnel((i+1)*10000, lam, con))**2

reconstructed_image = np.abs(fresnel(zf, lam, con))**2


end_time = time.time()  # End timing

print(f"Computation done. Time taken: {end_time - start_time:.2f} seconds.")

# Display the image using matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(reconstructed_image, cmap='gray') # Use 'gray' colormap for intensity images
plt.title(f'Reconstructed Image (zf={zf} µm)')
plt.colorbar(label='Intensity')
plt.axis('off') # Turn off axis ticks and labels
plt.show()

# Optional: Save the reconstructed image
# cv2.imwrite("reconstructed_image.tiff", reconstructed_image.astype(np.float32))
# Note: Saving complex data or displaying specific magnitudes might require careful handling
# of data types and normalization if the values are outside typical image ranges (0-255).
# Here, we're displaying the absolute square, which is intensity, so a grayscale map is appropriate.