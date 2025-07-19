import speedyDIH
import HelperFunctions


zf_values = list(range(30000, 90001, 1200))    # Continue the list in steps of 2000 up to 200000
zf_hologram_values = list(range(30000, 90001, 5000))
#HelperFunctions.generate_intervals(65830, 500, 5)  # Generate intervals around 65000 with a step of 2000 
refImagePath = "/home/berg/Documents/git/speedy-DIH/test_files/dust_hologram_blank.tiff"
rawImagePath = "/home/berg/Documents/git/speedy-DIH/test_files/sphere2_hologram.tiff"

speedyDIH.display_Tamura_graph(refImagePath, rawImagePath, zf_values)
#speedyDIH.display_Holograms(refImagePath, rawImagePath, zf_hologram_values)

