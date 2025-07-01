import numpy as np
import cv2 # For reading and writing TIFF images (OpenCV is good for this)
import matplotlib.pyplot as plt # For plotting
import matplotlib.animation as animation # For creating animations
import time
import cupy as cp # Import CuPy

# --- Parameters (from Mathematica code) ---
lam = 0.532  # measured in micrometers (wavelength)
pix = 3.45   # measured in micrometers (pixel size)
# zf = 67225   # This will now be a range of Z values
n = 0 # Number of iterations for the Fresnel transform (just for performance)

# Define the range of z values for the animation
z_start = 60000  # Starting focus distance in micrometers
z_end = 70000    # Ending focus distance in micrometers
num_frames = 20  # Number of different z values to animate
z_values = np.linspace(z_start, z_end, num_frames)

# --- Function Definitions ---

# Fresnel function
def fresnel(z, lambda_val, image_array_gpu):
    """
    Computes the Fresnel propagation of an image on the GPU using CuPy.
    Equivalent to Mathematica's Fresnel function in the context of this code.
    """
    size_x, size_y = image_array_gpu.shape
    
    halfSize_x = size_x // 2
    halfSize_y = size_y // 2

    x_coords = cp.arange(-halfSize_x, size_x - halfSize_x) * pix
    y_coords = cp.arange(-halfSize_y, size_y - halfSize_y) * pix
    
    X, Y = cp.meshgrid(x_coords, y_coords, indexing='ij') # 'ij' for matrix indexing
    
    exp_term = cp.exp(1j * cp.pi / (lambda_val * z) * (X**2 + Y**2))
    
    transformed_image_gpu = image_array_gpu * exp_term
    
    fft_result_gpu = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(transformed_image_gpu)))
    
    scale_factor = (1j / (lambda_val * z)) * cp.exp(1j * (2 * cp.pi / lambda_val) * z)
    
    return scale_factor * fft_result_gpu

# --- Main Program ---

start_time = time.time()  # Start timing

# 1. Import Images
try:
    ref_image_raw = cv2.imread("/home/berg/Documents/git/speedy-DIH/python_port/dust_hologram_blank.tiff", cv2.IMREAD_GRAYSCALE)
    raw_image_raw = cv2.imread("/home/berg/Documents/git/speedy-DIH/python_port/dust_hologram.tiff", cv2.IMREAD_GRAYSCALE)

    if ref_image_raw is None or raw_image_raw is None:
        raise FileNotFoundError("Make sure 'dust_hologram_blank.tiff' and 'dust_hologram.tiff' exist in the script directory.")

    ref_image_cpu = ref_image_raw.astype(np.complex128)
    raw_image_cpu = raw_image_raw.astype(np.complex128)

    ref_image_gpu = cp.asarray(ref_image_cpu)
    raw_image_gpu = cp.asarray(raw_image_cpu)

except Exception as e:
    print(f"Error loading images: {e}")
    print("Please ensure 'dust_hologram_blank.tiff' and 'dust_hologram.tiff' are present and valid TIFF files.")
    exit()

# Get image dimensions (assuming both images have the same dimensions)
size_x, size_y = ref_image_gpu.shape # Now getting shape from GPU array

# 2. Perform calculations (corresponds to 'con' in Mathematica) on GPU
con_gpu = raw_image_gpu / (ref_image_gpu**2)

# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(np.zeros((size_x, size_y)), cmap='gray') # Initialize with a black image
plt.title('Reconstructed Image')
plt.colorbar(im,label='Intensity')
plt.axis('off')

# Function to update the image for each frame
def update(frame):
    current_z = z_values[frame]
    print(f"Processing frame {frame+1}/{num_frames} at z={current_z:.2f} µm...")
    
    reconstructed_image_gpu = cp.abs(fresnel(current_z, lam, con_gpu))**2
    reconstructed_image_cpu = cp.asnumpy(reconstructed_image_gpu)
    
    im.set_data(reconstructed_image_cpu)
    ax.set_title(f'Reconstructed Image (z={current_z:.2f} µm)')
    return [im]

# Create the animation
# interval: Delay between frames in milliseconds.
# blit: Whether to use blitting for performance. Setting to True can speed up rendering.
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=200, blit=True)

end_time = time.time()  # End timing (for initial setup)
print(f"Initial setup done. Time taken: {end_time - start_time:.2f} seconds.")

# To save the animation, you might need ffmpeg installed and available in your PATH.
# You can install ffmpeg using your system's package manager (e.g., sudo apt-get install ffmpeg on Ubuntu).
# Or, if you're in an environment like Anaconda, it might already be available or installable via conda.
print("\nSaving animation. This might take a while...")
animation_start_time = time.time()

# Save the animation as an MP4 file
try:
    ani.save('reconstruction_animation.mp4', writer='ffmpeg', fps=5) # 5 frames per second
    print(f"Animation saved as 'reconstruction_animation.mp4'. Time taken for saving: {time.time() - animation_start_time:.2f} seconds.")
except ValueError as e:
    print(f"Error saving animation: {e}")
    print("Please ensure FFmpeg is installed and accessible in your system's PATH.")
    print("You can try installing it with: sudo apt-get install ffmpeg (Debian/Ubuntu) or brew install ffmpeg (macOS with Homebrew).")
except Exception as e:
    print(f"An unexpected error occurred during animation saving: {e}")

# If you prefer to display the animation interactively (might not work in all environments, e.g., headless servers)
# plt.show()

print("Script finished.")