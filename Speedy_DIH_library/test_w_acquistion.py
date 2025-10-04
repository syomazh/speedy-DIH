import speedyDIH
import HelperFunctions
import time
import subprocess
import os

# Change to the directory containing the executable
cpp_save_dir = "/home/berg/Documents/git/speedy-DIH/Data_acquistion/Cpp_Save_Tiff"
os.chdir(cpp_save_dir)

refImagePath = "/home/berg/Documents/git/speedy-DIH/Data_acquistion/Cpp_Save_Tiff/Images/Cpp_Save/image_ref.tiff"
counter_file = "/home/berg/Documents/git/speedy-DIH/Data_acquistion/Cpp_Save_Tiff/image_counter.txt"

# Run the loop 20 times
for i in range(20):
    print(f"Iteration {i+1}/20")
    
    # Run the executable to generate new image
    result = subprocess.run("./Cpp_Save_Tiff", shell=True, capture_output=True, text=True)
    
    # Check if the command was successful
    if result.returncode == 0:
        print("Cpp_Save_Tiff executed successfully")
        
        # Read the current image counter to get the image number
        try:
            with open(counter_file, 'r') as f:
                image_number = int(f.read().strip()) -1  # Subtract 1 to get the last generated image number    
            
            # Construct the path to the newly generated image
            rawImagePath = f"/home/berg/Documents/git/speedy-DIH/Data_acquistion/Cpp_Save_Tiff/Images/Cpp_Save/image_{image_number}.tiff"
            
            print(f"Using image: {rawImagePath}")
            
            # Find focus and display hologram for 1 second
            zf_hologram_values = [speedyDIH.find_focus_hierarchical(refImagePath, rawImagePath, 34000, 140000, 8)]
            speedyDIH.display_Holograms_1second(refImagePath, rawImagePath, zf_hologram_values)
            
        except Exception as e:
            print(f"Error reading counter file or processing image: {e}")
    else:
        print("Error executing Cpp_Save_Tiff")
        print("Error:", result.stderr)
        break  # Exit loop if cpp execution fails

print("Loop completed")

# Change back to your original directory
os.chdir("/home/berg/Documents/git/speedy-DIH/Speedy_DIH_library")