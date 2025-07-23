"""
Speedy Digital In-line Holography (DIH) Library
Provides efficient GPU-accelerated functions for holographic reconstruction
Author: Syoma Zharkov
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import cupy as cp
from typing import List, Dict, Tuple, Optional, Union


class SpeedyDIH:
    def __init__(self, wavelength: float = 0.532, pixel_size: float = 3.45):
        """
        Initialize the SpeedyDIH object with optical parameters.
        
        Args:
            wavelength: Light wavelength in micrometers (default: 0.532)
            pixel_size: Camera pixel size in micrometers (default: 3.45)
        """
        self.wavelength = wavelength  # µm
        self.pixel_size = pixel_size  # µm
        
    def load_images(self, ref_path: str, raw_path: str, use_high_precision: bool = False) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Load and prepare reference and raw hologram images for processing.
        
        Args:
            ref_path: Path to reference image file
            raw_path: Path to raw hologram image file
            use_high_precision: If True, use complex128 for higher precision (slower),
                               if False, use complex64 for faster computation (default)
                
        Returns:
            Tuple of (reference_image, raw_image) as CuPy complex arrays
        
        Raises:
            FileNotFoundError: If images cannot be loaded
        """
        try:
            ref_image_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            raw_image_raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)

            if ref_image_raw is None or raw_image_raw is None:
                raise FileNotFoundError(f"Failed to load images from {ref_path} or {raw_path}")

            # Choose dtype based on precision parameter
            dtype = cp.complex128 if use_high_precision else cp.complex64
            
            # Transfer to GPU as complex arrays
            ref_image = cp.asarray(ref_image_raw, dtype=dtype)
            raw_image = cp.asarray(raw_image_raw, dtype=dtype)
            
            return ref_image, raw_image
            
        except Exception as e:
            raise FileNotFoundError(f"Error loading images: {e}")

    def fresnel_propagation(self, 
                           image_array: cp.ndarray, 
                           propagation_distance: float,
                           cached_coords=None) -> cp.ndarray:
        """
        Compute Fresnel propagation of an image using the angular spectrum method.
        
        Args:
            image_array: Complex input image array on GPU
            propagation_distance: Propagation distance in micrometers
            
        Returns:
            Propagated complex field as CuPy array
        """
        size_y, size_x = image_array.shape
        
        if cached_coords is None:
            # Create coordinate grids (cached for repeated calls)
            half_x = size_x // 2
            half_y = size_y // 2
            x_coords = cp.arange(-half_x, size_x - half_x, dtype=cp.float32) * self.pixel_size
            y_coords = cp.arange(-half_y, size_y - half_y, dtype=cp.float32) * self.pixel_size
            
            X, Y = cp.meshgrid(x_coords, y_coords, indexing='xy')
            X2Y2 = X**2 + Y**2
            cached_coords = X2Y2
        else:
            X2Y2 = cached_coords
        
        # Use more efficient combined calculation
        k = cp.pi / (self.wavelength * propagation_distance)
        phase_factor = cp.exp(1j * k * X2Y2)
        
        # Use in-place operations where possible
        transformed = image_array * phase_factor
        
        # Use the plan_fft for better performance on repeated FFT operations
        fft_result = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(transformed)))
        
        # Final scaling factor
        scale = (1j / (self.wavelength * propagation_distance)) * cp.exp(1j * (2 * cp.pi / self.wavelength) * propagation_distance)
        
        return scale * fft_result

    @staticmethod
    def calculate_tamura(image: cp.ndarray) -> float:
        """
        Calculate Tamura coefficient (focus metric) for an image.
        Tamura = sqrt(standard_deviation / mean)
        
        Args:
            image: Input image as GPU array
            
        Returns:
            Tamura coefficient as float (on CPU)
        """
        mean_val = cp.mean(image)
        
        if mean_val == 0:
            return 0.0  # Avoid division by zero
            
        std_val = cp.std(image)
        return float(cp.sqrt(std_val / mean_val).get())  # Convert to CPU

    def _calculate_crop_dimensions(self, 
                                  original_size: Tuple[int, int], 
                                  propagation_distance: float) -> Tuple[int, int, int, int]:
        """
        Calculate cropping dimensions based on propagation physics.
        
        Args:
            original_size: Original image dimensions (height, width)
            propagation_distance: Propagation distance in micrometers
            
        Returns:
            Tuple of (start_y, end_y, start_x, end_x) for cropping
        """
        original_height, original_width = original_size
        
        # Calculate effective reconstruction pixel sizes
        effective_pix_x = (self.wavelength * propagation_distance) / (original_width * self.pixel_size)
        effective_pix_y = (self.wavelength * propagation_distance) / (original_height * self.pixel_size)
        
        # Calculate target dimensions
        input_fov_x = original_width * self.pixel_size
        input_fov_y = original_height * self.pixel_size
        
        target_width = min(original_width, int(round(input_fov_x / effective_pix_x)))
        target_height = min(original_height, int(round(input_fov_y / effective_pix_y)))
        
        # Ensure even dimensions
        if target_width % 2 != 0:
            target_width -= 1
        if target_height % 2 != 0:
            target_height -= 1
        
        # Calculate crop coordinates
        center_y, center_x = original_height // 2, original_width // 2
        
        start_x = max(0, center_x - target_width // 2)
        end_x = min(original_width, center_x + target_width // 2)
        start_y = max(0, center_y - target_height // 2)
        end_y = min(original_height, center_y + target_height // 2)
        
        return start_y, end_y, start_x, end_x

    def reconstruct_at_distance(self, 
                              ref_image: cp.ndarray, 
                              raw_image: cp.ndarray, 
                              distance: float) -> cp.ndarray:
        """
        Reconstruct hologram at a specific propagation distance.
        
        Args:
            ref_image: Reference image as CuPy array
            raw_image: Raw hologram image as CuPy array
            distance: Propagation distance in micrometers
            
        Returns:
            Intensity of reconstructed field as CuPy array
        """
        # Calculate contrast hologram
        contrast = raw_image / (ref_image**2)
        
        # Propagate using Fresnel
        reconstructed_field = self.fresnel_propagation(contrast, distance)
        
        # Return intensity
        return cp.abs(reconstructed_field)**2

    def find_focus(self, 
                  ref_path: str, 
                  raw_path: str, 
                  distance_range: List[float]) -> float:
        """
        Find optimal focus distance using the Tamura coefficient.
        
        Args:
            ref_path: Path to reference image
            raw_path: Path to raw hologram image
            distance_range: List of distances to evaluate
            
        Returns:
            Optimal focus distance in micrometers
        """
        tamura_results = self.calculate_focus_metrics(ref_path, raw_path, distance_range)
        
        # Find distance with maximum Tamura value
        best_distance = max(tamura_results, key=lambda x: x['tamura'])['distance']
        return best_distance
        
    def calculate_focus_metrics(self, 
                         ref_path: str, 
                         raw_path: str, 
                         distance_range: List[float],
                         use_high_precision: bool = False) -> List[Dict]:
        """
        Calculate focus metrics with batch processing and coordinate caching for improved performance.
        
        Args:
            ref_path: Path to reference image
            raw_path: Path to raw hologram image  
            distance_range: List of distances to evaluate
            use_high_precision: If True, use complex128 instead of complex64 (slower but more precise)
                
        Returns:
            List of dictionaries with distance and tamura values
        """
        start_time = time.time()
        
        # Load images
        ref_image, raw_image = self.load_images(ref_path, raw_path)
        
        # Calculate contrast (once)
        contrast = raw_image / (ref_image**2)
        
        # Store results
        tamura_results = []
        
        # Original image dimensions for cropping calculations
        original_size = ref_image.shape
        
        print(f"Calculating focus metrics across {len(distance_range)} distances...")
        
        # Pre-compute coordinate grid once (this is the key caching step)
        size_y, size_x = contrast.shape
        half_x = size_x // 2
        half_y = size_y // 2
        x_coords = cp.arange(-half_x, size_x - half_x, dtype=cp.float32) * self.pixel_size
        y_coords = cp.arange(-half_y, size_y - half_y, dtype=cp.float32) * self.pixel_size
        
        X, Y = cp.meshgrid(x_coords, y_coords, indexing='xy')
        cached_coords = X**2 + Y**2
        
        # Process in batches to improve GPU utilization
        batch_size = min(10, len(distance_range))  # Adjust based on GPU memory
        
        for i in range(0, len(distance_range), batch_size):
            batch_distances = distance_range[i:i+batch_size]
            batch_results = []
            
            for distance in batch_distances:
                # Reconstruct at current distance - pass cached coordinates
                reconstructed_field = self.fresnel_propagation(contrast, distance, cached_coords)
                intensity = cp.abs(reconstructed_field)**2
                
                # Get cropping dimensions
                start_y, end_y, start_x, end_x = self._calculate_crop_dimensions(original_size, distance)
                
                # Perform cropping directly on GPU for better performance
                if start_y > 0 or end_y < intensity.shape[0] or start_x > 0 or end_x < intensity.shape[1]:
                    cropped_gpu = intensity[start_y:end_y, start_x:end_x]
                else:
                    cropped_gpu = intensity
                
                # Calculate Tamura coefficient
                tamura = self.calculate_tamura(cropped_gpu)
                
                # Print the Tamura coefficient (CV) for each distance
                #print(f"zf={distance} µm: CV={tamura:.6f}")
                
                batch_results.append({
                    'distance': distance,
                    'tamura': tamura
                })
                
            tamura_results.extend(batch_results)
            
            # Explicitly synchronize and free memory
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            
        elapsed_time = time.time() - start_time
        print(f"Focus metrics calculated in {elapsed_time:.2f} seconds")
        
        return tamura_results

    def display_tamura_graph(self, 
                           ref_path: str, 
                           raw_path: str, 
                           distance_range: List[float],
                           save_path: Optional[str] = None) -> None:
        """
        Calculate and display a graph of Tamura coefficients vs distance.
        
        Args:
            ref_path: Path to reference image
            raw_path: Path to raw hologram image
            distance_range: List of distances to evaluate
            save_path: Optional path to save the graph
        """
        tamura_results = self.calculate_focus_metrics(ref_path, raw_path, distance_range)
        
        # Extract values for plotting
        distances = [result['distance'] for result in tamura_results]
        tamura_values = [result['tamura'] for result in tamura_results]
        
        # Find best focus distance
        best_idx = tamura_values.index(max(tamura_values))
        best_distance = distances[best_idx]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(distances, tamura_values, marker='o', linestyle='-', color='blue')
        plt.axvline(x=best_distance, color='red', linestyle='--', 
                    label=f'Best focus: {best_distance:.2f} µm')
        
        # Add labels and formatting
        plt.xlabel('Propagation Distance (µm)')
        plt.ylabel('Tamura Coefficient')
        plt.title('Focus Quality vs Propagation Distance')
        plt.grid(True)
        plt.legend()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            #print(f"Graph saved to {save_path}")
            
        plt.show()
        
        print(f"Best focus distance: {best_distance:.2f} µm with Tamura value: {max(tamura_values):.6f}")
        
    def display_reconstructions(self, 
                          ref_path: str, 
                          raw_path: str, 
                          distance_range: List[float],
                          use_high_precision: bool = False) -> None:
        """
        Display reconstructed holograms at multiple propagation distances with coordinate caching.
        
        Args:
            ref_path: Path to reference image
            raw_path: Path to raw hologram image
            distance_range: List of distances to reconstruct
            use_high_precision: If True, use complex128 instead of complex64 (slower but more precise)
        """
        start_time = time.time()
        
        # Load images
        ref_image, raw_image = self.load_images(ref_path, raw_path)
        
        # Calculate contrast
        contrast = raw_image / (ref_image**2)
        
        # Pre-compute coordinate grid once (this is the key caching step)
        size_y, size_x = contrast.shape
        half_x = size_x // 2
        half_y = size_y // 2
        x_coords = cp.arange(-half_x, size_x - half_x, dtype=cp.float32) * self.pixel_size
        y_coords = cp.arange(-half_y, size_y - half_y, dtype=cp.float32) * self.pixel_size
        
        X, Y = cp.meshgrid(x_coords, y_coords, indexing='xy')
        cached_coords = X**2 + Y**2
        
        # Calculate physical parameters
        input_fov_x = size_x * self.pixel_size
        input_fov_y = size_y * self.pixel_size
        print(f"Input Field of View: {input_fov_x:.2f} µm x {input_fov_y:.2f} µm")
        print(f"Pixel Size: {self.pixel_size} µm")

        # Create figure for visualizations
        fig, axes = plt.subplots(1, len(distance_range), figsize=(4 * len(distance_range), 6))
        if len(distance_range) == 1:
            axes = [axes]  # Handle single subplot case

        # Process each distance
        for i, distance in enumerate(distance_range):
            #print(f"Reconstructing at distance = {distance} µm...")
            
            # Reconstruct field and calculate intensity - pass cached coordinates
            reconstructed_field = self.fresnel_propagation(contrast, distance, cached_coords)
            intensity = cp.abs(reconstructed_field)**2
            
            # Display (transferring to CPU for matplotlib)
            ax = axes[i]
            ax.imshow(cp.asnumpy(intensity), cmap='gray')
            ax.set_title(f'z = {distance} µm')
            ax.axis('off')

        # Add title and adjust layout
        plt.suptitle("Hologram Reconstructions at Different Propagation Distances")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        elapsed_time = time.time() - start_time
        print(f"Total reconstruction time: {elapsed_time:.2f} seconds")
        
        plt.show()


# Example usage functions
def display_Holograms(refImage_filepath, rawImage_filepath, zf_values, lam=0.532, pix=3.45):
    """Legacy function maintained for backwards compatibility"""
    dih = SpeedyDIH(wavelength=lam, pixel_size=pix)
    dih.display_reconstructions(refImage_filepath, rawImage_filepath, zf_values)

def display_Tamura_graph(refImage_filepath, rawImage_filepath, zf_values, lam=0.532, pix=3.45):
    """Legacy function maintained for backwards compatibility"""
    dih = SpeedyDIH(wavelength=lam, pixel_size=pix)
    dih.display_tamura_graph(refImage_filepath, rawImage_filepath, zf_values)

def find_focus(refImage_filepath, rawImage_filepath, zf_values, lam=0.532, pix=3.45):
    """Find optimal focus distance using the Tamura method"""
    dih = SpeedyDIH(wavelength=lam, pixel_size=pix)
    return dih.find_focus(refImage_filepath, rawImage_filepath, zf_values)