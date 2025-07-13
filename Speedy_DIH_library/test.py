import speedyDIH

zf_values = list(range(10000, 100000, 2000))  # Continue the list in steps of 2000 up to 200000

refImagePath = "/home/berg/Documents/git/speedy-DIH/test_files/dust_hologram_blank.tiff"
rawImagePath = "/home/berg/Documents/git/speedy-DIH/test_files/sphere2_hologram.tiff"

speedyDIH.display_Tamura_graph(refImagePath, rawImagePath, zf_values)