import numpy as np
from speedyDIH import SpeedyDIH
import time

down_sample_factor = 1  # Change this factor to test different downsampling levels
# Initialize the holography processor with your optical parameters
wavelength = 0.532/down_sample_factor  # Green laser wavelength in micrometers
pixel_size = 3.45/down_sample_factor   # Camera pixel size in micrometers

dih = SpeedyDIH(wavelength=wavelength, pixel_size=pixel_size)

# Define your image file paths
# ref_image_path = "/home/berg/Documents/git/speedy-DIH/Data_acquistion/Cpp_Save_Tiff/Images/Cpp_Save/white_ref.tiff"
# raw_image_path = "/home/berg/Documents/git/speedy-DIH/Data_acquistion/Cpp_Save_Tiff/Images/Cpp_Save/white_ref.tiff"

ref_image_path = "/home/berg/Documents/git/speedy-DIH/test_files/dust_hologram_blank.tiff"
raw_image_path = "/home/berg/Documents/git/speedy-DIH/test_files/dust_hologram.tiff"

# Define propagation distances to test (in micrometers)
z_distances = np.arange(100, 140001, 1000)  # From 0 to 30 mm in steps of 1 mm

# # 1. Display reconstructions at different distances with downsampling
# print("Displaying hologram reconstructions (with 2x downsampling for speed)...")
# dih.display_reconstructions(ref_image_path, raw_image_path, z_distances, downsample_factor=2)

# # 2. Find optimal focus distance using simple grid search with downsampling
# print("Finding optimal focus distance (with 4x downsampling for speed)...")
# optimal_distance = dih.find_focus(ref_image_path, raw_image_path, z_distances, downsample_factor=4)
# print(f"Optimal focus distance: {optimal_distance} μm")

# 3. Find focus using hierarchical search with downsampling (more efficient)
print("Finding focus using hierarchical search (with 4x downsampling)...")
min_distance = 34000
max_distance = 140000  

start_time = time.time()  # Start timing
optimal_distance_hier = dih.find_focus_hierarchical(
    ref_image_path, raw_image_path, 
    min_distance, max_distance, 
    n_points=8,
    downsample_factor=down_sample_factor  # 2x downsampling for faster search
)

end_time = time.time()  # End timing

elapsed_time = end_time - start_time
print(f"Optimal focus distance (hierarchical): {optimal_distance_hier} μm")
print(f"Time taken: {elapsed_time:.4f} seconds")

# # 4. Display Tamura coefficient graph with downsampling
# print("Displaying focus quality graph (with 2x downsampling)...")
dih.display_tamura_graph(ref_image_path, raw_image_path, z_distances, downsample_factor=down_sample_factor)

# # 5. Display reconstructions at the found optimal distance (full resolution)
# dih.display_reconstructions(ref_image_path, raw_image_path, [optimal_distance_hier], downsample_factor=1)
