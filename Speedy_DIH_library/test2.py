import numpy as np
from speedyDIH import SpeedyDIH
import time

# Initialize the holography processor with your optical parameters
wavelength = 0.532  # Green laser wavelength in micrometers
pixel_size = 3.45   # Camera pixel size in micrometers

dih = SpeedyDIH(wavelength=wavelength, pixel_size=pixel_size)

# Define your image file paths
ref_image_path = "/home/berg/Documents/git/speedy-DIH/Data_acquistion/Cpp_Save_Tiff/Images/Cpp_Save/white_ref.tiff"
raw_image_path = "/home/berg/Documents/git/speedy-DIH/Data_acquistion/Cpp_Save_Tiff/Images/Cpp_Save/white_ref.tiff"
# ref_image_path = "/home/berg/Documents/git/speedy-DIH/test_files/dust_hologram_blank.tiff"
# raw_image_path = "/home/berg/Documents/git/speedy-DIH/test_files/sphere3_hologram.tiff"

# # Define propagation distances to test (in micrometers)
# z_distances = [100, 150, 200, 250, 300, 350, 400]

# # 1. Display reconstructions at different distances
# print("Displaying hologram reconstructions...")
# dih.display_reconstructions(ref_image_path, raw_image_path, z_distances)

# # 2. Find optimal focus distance using simple grid search
# print("Finding optimal focus distance...")
# optimal_distance = dih.find_focus(ref_image_path, raw_image_path, z_distances)
# print(f"Optimal focus distance: {optimal_distance} μm")

# 3. Find focus using hierarchical search (more efficient for large ranges)
print("Finding focus using hierarchical search...")
min_distance = 30000
max_distance = 150000

start_time = time.time()  # Start timing
optimal_distance_hier = dih.find_focus_hierarchical(
    ref_image_path, raw_image_path, 
    min_distance, max_distance, 
    n_points=8
)
end_time = time.time()  # End timing

elapsed_time = end_time - start_time
print(f"Optimal focus distance (hierarchical): {optimal_distance_hier} μm")
print(f"Time taken: {elapsed_time:.4f} seconds")

# # 4. Display Tamura coefficient graph to visualize focus quality
# print("Displaying focus quality graph...")
# dih.display_tamura_graph(ref_image_path, raw_image_path, z_distances)