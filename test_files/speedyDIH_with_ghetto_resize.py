"""
Speedy Digital In-line Holography (DIH) Library
Provides efficient GPU-accelerated functions for holographic reconstruction
Author: Syoma Zharkov
"""
import numpy as np
import cupy as cp
import cv2
import tifffile
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from abc import ABC, abstractmethod


class ImageLoader:
    """Handles image loading and preprocessing operations."""
    
    @staticmethod
    def load_image_pair(ref_path: str, raw_path: str, use_high_precision: bool = False, 
                       downsample_factor: int = 1) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Load and prepare reference and raw hologram images for processing.
        
        Args:
            ref_path: Path to reference image
            raw_path: Path to raw hologram image
            use_high_precision: Use complex128 instead of complex64
            downsample_factor: Factor to downsample images (1=no downsampling, 2=half resolution, etc.)
        
        Returns:
            Tuple of (ref_image, raw_image) as CuPy arrays
        """
        start_time = time.time()
        
        try:
            # Time the file loading
            load_start = time.time()
            
            # Use tifffile for .tif/.tiff files, otherwise use OpenCV
            if ref_path.lower().endswith(('.tif', '.tiff')):
                ref_image_raw = tifffile.imread(ref_path)
                raw_image_raw = tifffile.imread(raw_path)
                
                # Handle color images efficiently
                if ref_image_raw.ndim == 3:
                    ref_image_raw = ref_image_raw[:, :, 0]  # Take first channel
                if raw_image_raw.ndim == 3:
                    raw_image_raw = raw_image_raw[:, :, 0]  # Take first channel
            else:
                # OpenCV automatically loads as grayscale
                ref_image_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
                raw_image_raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
            
            load_time = time.time() - load_start

            if ref_image_raw is None or raw_image_raw is None:
                raise FileNotFoundError(f"Failed to load images from {ref_path} or {raw_path}")

            # Downsample images if requested
            if downsample_factor > 1:
                downsample_start = time.time()
                original_shape = ref_image_raw.shape
                
                # Use cv2.resize with INTER_AREA for best quality downsampling
                new_width = ref_image_raw.shape[1] // downsample_factor
                new_height = ref_image_raw.shape[0] // downsample_factor
                
                ref_image_raw = cv2.resize(ref_image_raw, (new_width, new_height), 
                                          interpolation=cv2.INTER_AREA)
                raw_image_raw = cv2.resize(raw_image_raw, (new_width, new_height), 
                                          interpolation=cv2.INTER_AREA)
                
                downsample_time = time.time() - downsample_start
                print(f"Downsampled from {original_shape} to {ref_image_raw.shape} "
                      f"(factor: {downsample_factor}x) in {downsample_time:.3f} seconds")

            # Choose dtype based on precision parameter
            dtype = cp.complex128 if use_high_precision else cp.complex64
            
            # Time the GPU transfer
            transfer_start = time.time()
            # Direct transfer to GPU as complex arrays
            ref_image = cp.asarray(ref_image_raw, dtype=dtype)
            raw_image = cp.asarray(raw_image_raw, dtype=dtype)
            transfer_time = time.time() - transfer_start
            
            total_time = time.time() - start_time
            
            print(f"Image loading completed:")
            print(f"  File I/O time: {load_time:.3f} seconds")
            print(f"  GPU transfer time: {transfer_time:.3f} seconds")
            print(f"  Total loading time: {total_time:.3f} seconds")
            print(f"  Image dimensions: {ref_image.shape}")
            print(f"  Precision: {'High (complex128)' if use_high_precision else 'Standard (complex64)'}")
            
            return ref_image, raw_image
            
        except Exception as e:
            raise FileNotFoundError(f"Error loading images: {e}")


class CoordinateCache:
    """Manages coordinate grid caching for improved performance."""
    
    def __init__(self, pixel_size: float):
        self.pixel_size = pixel_size
        self._cache = {}
    
    def get_coordinates(self, shape: Tuple[int, int]) -> cp.ndarray:
        """Get cached coordinate grid for given shape."""
        if shape not in self._cache:
            self._cache[shape] = self._compute_coordinates(shape)
        return self._cache[shape]
    
    def _compute_coordinates(self, shape: Tuple[int, int]) -> cp.ndarray:
        """Compute coordinate grid for given shape."""
        size_y, size_x = shape
        half_x = size_x // 2
        half_y = size_y // 2
        
        x_coords = cp.arange(-half_x, size_x - half_x, dtype=cp.float32) * self.pixel_size
        y_coords = cp.arange(-half_y, size_y - half_y, dtype=cp.float32) * self.pixel_size
        
        X, Y = cp.meshgrid(x_coords, y_coords, indexing='xy')
        return X**2 + Y**2
    
    def clear_cache(self):
        """Clear the coordinate cache."""
        self._cache.clear()


class FocusMetric(ABC):
    """Abstract base class for focus quality metrics."""
    
    @abstractmethod
    def calculate(self, image: cp.ndarray) -> float:
        """Calculate focus metric for an image."""
        pass


class TamuraMetric(FocusMetric):
    """Tamura coefficient focus metric implementation."""
    
    def calculate(self, image: cp.ndarray) -> float:
        """
        Calculate Tamura coefficient (focus metric) for an image.
        Tamura = sqrt(standard_deviation / mean)
        """
        mean_val = cp.mean(image)
        
        if mean_val == 0:
            return 0.0
            
        std_val = cp.std(image)
        return float(cp.sqrt(std_val / mean_val).get())


class HologramProcessor:
    """Core hologram processing operations."""
    
    def __init__(self, wavelength: float, pixel_size: float):
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.coord_cache = CoordinateCache(pixel_size)
    
    def fresnel_propagation(self, 
                           image_array: cp.ndarray, 
                           propagation_distance: float,
                           cached_coords: Optional[cp.ndarray] = None) -> cp.ndarray:
        """
        Compute Fresnel propagation using the angular spectrum method.
        """
        if cached_coords is None:
            cached_coords = self.coord_cache.get_coordinates(image_array.shape)
        
        # Use more efficient combined calculation
        k = cp.pi / (self.wavelength * propagation_distance)
        phase_factor = cp.exp(1j * k * cached_coords)
        
        # Use in-place operations where possible
        transformed = image_array * phase_factor
        
        # FFT operations
        fft_result = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(transformed)))
        
        # Final scaling factor
        scale = (1j / (self.wavelength * propagation_distance)) * cp.exp(1j * (2 * cp.pi / self.wavelength) * propagation_distance)
        
        return scale * fft_result
    
    def calculate_crop_dimensions(self, 
                                 original_size: Tuple[int, int], 
                                 propagation_distance: float) -> Tuple[int, int, int, int]:
        """Calculate cropping dimensions based on propagation physics."""
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


class FocusFinder:
    """Handles focus finding algorithms."""
    
    def __init__(self, processor: HologramProcessor, focus_metric: FocusMetric):
        self.processor = processor
        self.focus_metric = focus_metric
    
    def find_focus_simple(self, 
                         ref_image: cp.ndarray,
                         raw_image: cp.ndarray,
                         distance_range: List[float]) -> float:
        """Find optimal focus using simple grid search."""
        results = self._calculate_focus_metrics(ref_image, raw_image, distance_range)
        best_result = max(results, key=lambda x: x['metric_value'])
        return best_result['distance']
    
    def find_focus_hierarchical(self, 
                               ref_image: cp.ndarray,
                               raw_image: cp.ndarray,
                               min_distance: float,
                               max_distance: float,
                               n_points: int = 10) -> float:
        """Find optimal focus using hierarchical two-phase grid search."""
        # Phase 1: Coarse search
        step = (max_distance - min_distance) / (n_points - 1)
        coarse_distances = [min_distance + step * i for i in range(n_points)]
        
        coarse_results = self._calculate_focus_metrics(ref_image, raw_image, coarse_distances)
        
        # Find best point
        best_idx = max(range(len(coarse_results)), key=lambda i: coarse_results[i]['metric_value'])
        
        # Phase 2: Refined search
        lower_idx = max(0, best_idx - 1)
        upper_idx = min(len(coarse_distances) - 1, best_idx + 1)
        
        refined_min = coarse_distances[lower_idx]
        refined_max = coarse_distances[upper_idx]
        
        refined_step = (refined_max - refined_min) / (n_points - 1)
        refined_distances = [refined_min + refined_step * j for j in range(n_points)]
        
        refined_results = self._calculate_focus_metrics(ref_image, raw_image, refined_distances)
        
        best_refined = max(refined_results, key=lambda x: x['metric_value'])
        return best_refined['distance']
    
    def _calculate_focus_metrics(self, 
                                ref_image: cp.ndarray,
                                raw_image: cp.ndarray,
                                distance_range: List[float]) -> List[Dict]:
        """Calculate focus metrics for a range of distances."""
        contrast = raw_image / (ref_image**2)
        cached_coords = self.processor.coord_cache.get_coordinates(contrast.shape)
        
        results = []
        batch_size = min(10, len(distance_range))
        
        for i in range(0, len(distance_range), batch_size):
            batch_distances = distance_range[i:i+batch_size]
            
            for distance in batch_distances:
                # Reconstruct at current distance
                reconstructed_field = self.processor.fresnel_propagation(contrast, distance, cached_coords)
                intensity = cp.abs(reconstructed_field)**2
                
                # Apply cropping if needed
                crop_dims = self.processor.calculate_crop_dimensions(ref_image.shape, distance)
                start_y, end_y, start_x, end_x = crop_dims
                
                if start_y > 0 or end_y < intensity.shape[0] or start_x > 0 or end_x < intensity.shape[1]:
                    cropped_intensity = intensity[start_y:end_y, start_x:end_x]
                else:
                    cropped_intensity = intensity
                
                # Calculate focus metric
                metric_value = self.focus_metric.calculate(cropped_intensity)
                
                results.append({
                    'distance': distance,
                    'metric_value': metric_value
                })
                
            # Memory management
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
        
        return results


class HologramVisualizer:
    """Handles visualization and plotting operations."""
    
    def __init__(self, processor: HologramProcessor):
        self.processor = processor
    
    def display_reconstructions(self, 
                          data_images: List[cp.ndarray],
                          distance_range: List[float],
                          pixel_size: Optional[float] = None,
                          show_duration: Optional[float] = None) -> None:
        """Display pre-computed hologram reconstructions at multiple distances."""
        start_time = time.time()
        
        if len(data_images) != len(distance_range):
            raise ValueError(f"Number of data images ({len(data_images)}) must match number of distances ({len(distance_range)})")
        
        # Use provided pixel_size or fall back to processor's pixel_size
        pix_size = pixel_size if pixel_size is not None else self.processor.pixel_size
        
        # Calculate physical parameters from first image
        size_y, size_x = data_images[0].shape
        input_fov_x = size_x * pix_size
        input_fov_y = size_y * pix_size
        print(f"Display Field of View: {input_fov_x:.2f} µm x {input_fov_y:.2f} µm")
        print(f"Pixel Size: {pix_size} µm")

        # Create figure
        fig, axes = plt.subplots(1, len(distance_range), figsize=(4 * len(distance_range), 6))
        if len(distance_range) == 1:
            axes = [axes]

        # Display each pre-computed image
        for i, (data_image, distance) in enumerate(zip(data_images, distance_range)):
            # Handle both intensity and complex field data
            if cp.iscomplexobj(data_image):
                intensity = cp.abs(data_image)**2
            else:
                intensity = data_image
            
            ax = axes[i]
            ax.imshow(cp.asnumpy(intensity), cmap='gray')
            ax.set_title(f'z = {distance} µm')
            ax.axis('off')

        plt.suptitle("Hologram Reconstructions at Different Propagation Distances")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        elapsed_time = time.time() - start_time
        print(f"Total display time: {elapsed_time:.2f} seconds")
        
        if show_duration is not None:
            plt.ion()
            plt.show()
            plt.pause(show_duration)
            plt.ioff()
            plt.close(fig)
        else:
            plt.show()
    
    def display_focus_graph(self, 
                           results: List[Dict],
                           save_path: Optional[str] = None) -> None:
        """Display focus metric graph."""
        distances = [result['distance'] for result in results]
        metric_values = [result['metric_value'] for result in results]
        
        best_idx = metric_values.index(max(metric_values))
        best_distance = distances[best_idx]
        
        plt.figure(figsize=(10, 6))
        plt.plot(distances, metric_values, marker='o', linestyle='-', color='blue')
        plt.axvline(x=best_distance, color='red', linestyle='--', 
                    label=f'Best focus: {best_distance:.2f} µm')
        
        plt.xlabel('Propagation Distance (µm)')
        plt.ylabel('Focus Metric')
        plt.title('Focus Quality vs Propagation Distance')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
        print(f"Best focus distance: {best_distance:.2f} µm with metric value: {max(metric_values):.6f}")


class SpeedyDIH:
    """Main interface for Digital In-line Holography operations."""
    
    def __init__(self, wavelength: float = 0.532, pixel_size: float = 3.45):
        """
        Initialize the SpeedyDIH object with optical parameters.
        
        Args:
            wavelength: Light wavelength in micrometers (default: 0.532)
            pixel_size: Camera pixel size in micrometers (default: 3.45)
        """
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.processor = HologramProcessor(wavelength, pixel_size)
        self.focus_finder = FocusFinder(self.processor, TamuraMetric())
        self.visualizer = HologramVisualizer(self.processor)
        self.image_loader = ImageLoader()
    
    # Legacy method names for backwards compatibility
    def load_images(self, ref_path: str, raw_path: str, use_high_precision: bool = False,
                   downsample_factor: int = 1) -> Tuple[cp.ndarray, cp.ndarray]:
        """Load and prepare images for processing."""
        return self.image_loader.load_image_pair(ref_path, raw_path, use_high_precision, downsample_factor)
    
    def fresnel_propagation(self, image_array: cp.ndarray, propagation_distance: float, cached_coords=None) -> cp.ndarray:
        """Compute Fresnel propagation."""
        return self.processor.fresnel_propagation(image_array, propagation_distance, cached_coords)
    
    @staticmethod
    def calculate_tamura(image: cp.ndarray) -> float:
        """Calculate Tamura coefficient."""
        return TamuraMetric().calculate(image)
    
    def find_focus(self, ref_path: str, raw_path: str, distance_range: List[float],
                  downsample_factor: int = 1) -> float:
        """Find optimal focus distance using simple grid search."""
        ref_image, raw_image = self.load_images(ref_path, raw_path, downsample_factor=downsample_factor)
    
        
        return self.focus_finder.find_focus_simple(ref_image, raw_image, distance_range)
    
    def find_focus_hierarchical(self, ref_path: str, raw_path: str, min_distance: float, 
                               max_distance: float, n_points: int = 10, use_high_precision: bool = False,
                               downsample_factor: int = 1) -> float:
        """Find optimal focus using hierarchical search."""
        ref_image, raw_image = self.load_images(ref_path, raw_path, use_high_precision, downsample_factor)
        
        return self.focus_finder.find_focus_hierarchical(ref_image, raw_image, min_distance, max_distance, n_points)
    
    def display_reconstructions(self, ref_path: str, raw_path: str, distance_range: List[float], 
                              use_high_precision: bool = False, downsample_factor: int = 1) -> None:
        """Display reconstructed holograms."""
        ref_image, raw_image = self.load_images(ref_path, raw_path, use_high_precision, downsample_factor)
        contrast = raw_image / (ref_image**2)
        
        cached_coords = self.processor.coord_cache.get_coordinates(contrast.shape)
        
        # Pre-compute all reconstructions
        data_images = []
        for distance in distance_range:
            reconstructed_field = self.processor.fresnel_propagation(contrast, distance, cached_coords)
            data_images.append(reconstructed_field)
        
        self.visualizer.display_reconstructions(data_images, distance_range)
    
    def display_reconstructions_1second(self, ref_path: str, raw_path: str, distance_range: List[float], 
                                      use_high_precision: bool = False, downsample_factor: int = 1) -> None:
        """Display reconstructed holograms for 1 second."""
        ref_image, raw_image = self.load_images(ref_path, raw_path, use_high_precision, downsample_factor)
        contrast = raw_image / (ref_image**2)
        
        cached_coords = self.processor.coord_cache.get_coordinates(contrast.shape)
        
        # Pre-compute all reconstructions
        data_images = []
        for distance in distance_range:
            reconstructed_field = self.processor.fresnel_propagation(contrast, distance, cached_coords)
            data_images.append(reconstructed_field)
        
        self.visualizer.display_reconstructions(data_images, distance_range, show_duration=1.0)
    
    def display_tamura_graph(self, ref_path: str, raw_path: str, distance_range: List[float], 
                           save_path: Optional[str] = None, downsample_factor: int = 1) -> None:
        """Display Tamura coefficient graph."""
        ref_image, raw_image = self.load_images(ref_path, raw_path, downsample_factor=downsample_factor)
        
        results = self.focus_finder._calculate_focus_metrics(ref_image, raw_image, distance_range)
        # Convert to expected format
        tamura_results = [{'distance': r['distance'], 'metric_value': r['metric_value']} for r in results]
        self.visualizer.display_focus_graph(tamura_results, save_path)


# Legacy functions for backwards compatibility
def display_Holograms(refImage_filepath, rawImage_filepath, zf_values, lam=0.532, pix=3.45):
    """Legacy function maintained for backwards compatibility"""
    dih = SpeedyDIH(wavelength=lam, pixel_size=pix)
    dih.display_reconstructions(refImage_filepath, rawImage_filepath, zf_values)

def display_Holograms_1second(refImage_filepath, rawImage_filepath, zf_values, lam=0.532, pix=3.45):
    """Legacy function maintained for backwards compatibility"""
    dih = SpeedyDIH(wavelength=lam, pixel_size=pix)
    dih.display_reconstructions_1second(refImage_filepath, rawImage_filepath, zf_values)

def display_Tamura_graph(refImage_filepath, rawImage_filepath, zf_values, lam=0.532, pix=3.45):
    """Legacy function maintained for backwards compatibility"""
    dih = SpeedyDIH(wavelength=lam, pixel_size=pix)
    dih.display_tamura_graph(refImage_filepath, rawImage_filepath, zf_values)

def find_focus(refImage_filepath, rawImage_filepath, zf_values, lam=0.532, pix=3.45):
    """Find optimal focus distance using the Tamura method"""
    dih = SpeedyDIH(wavelength=lam, pixel_size=pix)
    return dih.find_focus(refImage_filepath, rawImage_filepath, zf_values)

def find_focus_hierarchical(refImage_filepath, rawImage_filepath, min_zf, max_zf, n_points=10, lam=0.532, pix=3.45):
    """Hierarchical search for optimal focus distance"""
    dih = SpeedyDIH(wavelength=lam, pixel_size=pix)
    return dih.find_focus_hierarchical(refImage_filepath, rawImage_filepath, min_zf, max_zf, n_points)