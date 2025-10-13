"""
Downsample and Save Image

This script loads an image, downsamples it by a specified factor, and saves the result.
Supports both TIFF and standard image formats (PNG, JPG, etc.).

Usage:
    python downsample_and_save.py
    
Or modify the parameters in the script below.
"""

import cv2
import tifffile
import numpy as np
import os


def downsample_image(image: np.ndarray, downsample_factor: int) -> np.ndarray:
    """
    Downsample an image by the specified factor using area interpolation.
    
    Args:
        image: Input image as numpy array
        downsample_factor: Factor to downsample by (e.g., 2 = half size, 4 = quarter size)
    
    Returns:
        Downsampled image as numpy array
    """
    if downsample_factor == 1:
        return image
    
    height, width = image.shape[:2]
    new_height = height // downsample_factor
    new_width = width // downsample_factor
    
    # Use INTER_AREA for downsampling (best quality)
    downsampled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    print(f"Original size: {width}x{height}")
    print(f"Downsampled size: {new_width}x{new_height}")
    print(f"Reduction factor: {downsample_factor}x")
    
    return downsampled


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file. Supports TIFF and standard formats.
    
    Args:
        image_path: Path to input image
    
    Returns:
        Image as numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Use tifffile for TIFF files
    if image_path.lower().endswith(('.tif', '.tiff')):
        image = tifffile.imread(image_path)
        print(f"Loaded TIFF image from: {image_path}")
    else:
        # Use OpenCV for other formats
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        print(f"Loaded image from: {image_path}")
    
    # Handle color images - convert to grayscale if needed
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print("Converted color image to grayscale")
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            print("Converted RGBA image to grayscale")
    
    return image


def save_image(image: np.ndarray, output_path: str):
    """
    Save an image to file. Automatically chooses format based on extension.
    
    Args:
        image: Image to save as numpy array
        output_path: Path where to save the image
    """
    # Use tifffile for TIFF files
    if output_path.lower().endswith(('.tif', '.tiff')):
        tifffile.imwrite(output_path, image)
        print(f"Saved TIFF image to: {output_path}")
    else:
        # Use OpenCV for other formats
        success = cv2.imwrite(output_path, image)
        if not success:
            raise ValueError(f"Failed to save image to: {output_path}")
        print(f"Saved image to: {output_path}")


def main():
    """Main function to downsample and save an image."""
    
    # ========================
    # CONFIGURATION
    # ========================
    
    # Input image path
    input_image_path = "test_files/dust_hologram_blank.tiff"
    
    # Output image path
    output_image_path = "test_files/dust_hologram_blank_downsampled.tiff"

    # Downsample factor (1 = no downsampling, 2 = half size, 4 = quarter size, etc.)
    downsample_factor = 2
    
    # ========================
    # PROCESSING
    # ========================
    
    print("=" * 60)
    print("Image Downsampling and Save")
    print("=" * 60)
    
    # Load the image
    print("\n[1] Loading image...")
    image = load_image(input_image_path)
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    
    # Downsample the image
    print(f"\n[2] Downsampling by factor of {downsample_factor}...")
    downsampled_image = downsample_image(image, downsample_factor)
    
    # Save the downsampled image
    print("\n[3] Saving downsampled image...")
    save_image(downsampled_image, output_image_path)
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    
    # Print summary
    original_pixels = image.shape[0] * image.shape[1]
    downsampled_pixels = downsampled_image.shape[0] * downsampled_image.shape[1]
    reduction_percent = (1 - downsampled_pixels / original_pixels) * 100
    
    print(f"\nSummary:")
    print(f"  Input: {input_image_path}")
    print(f"  Output: {output_image_path}")
    print(f"  Downsample factor: {downsample_factor}x")
    print(f"  Pixel reduction: {reduction_percent:.1f}%")


if __name__ == "__main__":
    main()
