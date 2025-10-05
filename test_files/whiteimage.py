from PIL import Image
import numpy as np

# Create a white image with specified dimensions
width = 4024
height = 3036

# Create a white image (RGB mode, all pixels set to 255 for white)
white_image = Image.new('RGB', (width, height), (255, 255, 255))

# Save as TIFF file
white_image.save('white_image_4024x3036.tiff', 'TIFF')

print(f"Generated white TIFF image: {width}x{height} pixels")
print("File saved as: white_image_4024x3036.tiff")