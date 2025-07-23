import speedyDIH
import HelperFunctions
import time


zf_values = list(range(30000, 90001, 1200))    # Continue the list in steps of 2000 up to 200000
zf_hologram_values = [67185]
#HelperFunctions.generate_intervals(65830, 500, 5)  # Generate intervals around 65000 with a step of 2000 
refImagePath = "/home/berg/Documents/git/speedy-DIH/test_files/dust_hologram_blank.tiff"
rawImagePath = "/home/berg/Documents/git/speedy-DIH/test_files/dust_hologram.tiff"


#speedyDIH.display_Tamura_graph(refImagePath, rawImagePath, zf_values)
start_time = time.time()

speedyDIH.find_focus_hierarchical(refImagePath, rawImagePath, 34000, 90000, n_points=10, lam=0.532, pix=3.45)
speedyDIH.display_Holograms(refImagePath, rawImagePath, zf_hologram_values)

end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")


