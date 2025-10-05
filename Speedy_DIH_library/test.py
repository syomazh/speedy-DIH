import speedyDIH
import HelperFunctions
import time
import subprocess
import os

print("Starting test script...")
# Change to the directory containing the executable
cpp_save_dir = "/home/berg/Documents/git/speedy-DIH/Data_acquistion/Cpp_Save_Tiff"
os.chdir(cpp_save_dir)


zf_values = list(range(30000, 150001, 1200))    # Continue the list in steps of 2000 up to 200000
#HelperFunctions.generate_intervals(65830, 500, 5)  # Generate intervals around 65000 with a step of 2000


# refImagePath = "/home/berg/Documents/git/speedy-DIH/test_files/dust_hologram_blank.tiff"
# rawImagePath = "/home/berg/Documents/git/speedy-DIH/test_files/sphere3_hologram.tiff"

refImagePath = "/home/berg/Documents/git/speedy-DIH/Data_acquistion/Cpp_Save_Tiff/Images/Cpp_Save/white_ref.tiff"
rawImagePath = "/home/berg/Documents/git/speedy-DIH/Data_acquistion/Cpp_Save_Tiff/Images/Cpp_Save/white_ref.tiff"


# speedyDIH.display_Tamura_graph(refImagePath, rawImagePath, zf_values)

zf_hologram_values = [speedyDIH.find_focus_hierarchical(refImagePath, rawImagePath, 34000, 140000, 8)]
speedyDIH.display_Holograms(refImagePath, rawImagePath, zf_hologram_values)