# Downsampling Feature Documentation

## Overview
The downsampling feature allows you to reduce image resolution when loading images, significantly reducing computation time at the cost of some detail.

## Usage

### Basic Image Loading
```python
from speedyDIH import SpeedyDIH

dih = SpeedyDIH(wavelength=0.532, pixel_size=3.45)

# Full resolution (default)
ref, raw = dih.load_images("ref.tiff", "raw.tiff", downsample_factor=1)

# Half resolution (4x fewer pixels, ~4x faster)
ref, raw = dih.load_images("ref.tiff", "raw.tiff", downsample_factor=2)

# Quarter resolution (16x fewer pixels, ~16x faster)
ref, raw = dih.load_images("ref.tiff", "raw.tiff", downsample_factor=4)
```

### Focus Finding with Downsampling
```python
# Fast focus search with downsampled images
focus = dih.find_focus(
    "ref.tiff", 
    "raw.tiff", 
    distance_range=[10, 20, 30],
    downsample_factor=4  # 4x downsampling for speed
)
```

### Hierarchical Search with Downsampling
```python
# Hierarchical search with half resolution
focus = dih.find_focus_hierarchical(
    "ref.tiff",
    "raw.tiff",
    min_distance=10.0,
    max_distance=40.0,
    n_points=15,
    downsample_factor=2  # 2x downsampling
)
```

### Display with Downsampling
```python
# Display reconstructions at lower resolution for speed
dih.display_reconstructions(
    "ref.tiff",
    "raw.tiff",
    distance_range=[15, 20, 25],
    downsample_factor=2
)

# Display Tamura graph with downsampling
dih.display_tamura_graph(
    "ref.tiff",
    "raw.tiff",
    distance_range=[15, 20, 25],
    downsample_factor=4
)
```

## Downsample Factor Guidelines

| Factor | Resolution | Speedup | Use Case |
|--------|-----------|---------|----------|
| 1 | Full (100%) | 1x | Final results, high quality |
| 2 | Half (50%) | ~4x | Good balance of speed/quality |
| 4 | Quarter (25%) | ~16x | Quick previews, initial searches |
| 8 | 1/8 (12.5%) | ~64x | Very fast rough estimates |

## Performance Impact

Downsampling reduces:
- **Memory usage**: By factor²
- **Computation time**: By approximately factor²
- **GPU memory**: By factor²

Example: `downsample_factor=4` means:
- 4000x3000 image → 1000x750 image
- 12MP → 0.75MP (16x fewer pixels)
- ~16x faster processing

## Important Notes

1. **Pixel Size Adjustment**: When downsampling, the effective pixel size increases by the downsample factor. For example:
   - Original pixel size: 3.45 µm, downsample_factor=2 → Effective pixel size: 6.9 µm
   - Original pixel size: 3.45 µm, downsample_factor=4 → Effective pixel size: 13.8 µm
   
   **This is automatically handled internally** - the library creates temporary processors with the correct effective pixel size, ensuring that the physics calculations (Fresnel propagation, coordinate grids, etc.) remain accurate regardless of downsampling level.

2. **Consistent Results Across Resolutions**: The Tamura graphs and focus finding should produce very similar results across different downsampling levels because the effective pixel size is properly adjusted. The peak positions should align, though:
   - Lower resolutions may have slightly smoother curves (less noise)
   - Very high downsampling (8x+) may lose some fine detail

3. **Quality Trade-off**: Higher downsampling = faster processing but less detail. Choose based on your needs:
   - **Initial exploration**: Use factor 4-8
   - **Refinement**: Use factor 2
   - **Final results**: Use factor 1 (no downsampling)

3. **Interpolation Method**: Uses OpenCV's INTER_AREA for high-quality downsampling that prevents aliasing.

4. **Two-stage Workflow**: 
   ```python
   # Stage 1: Fast search with downsampling
   rough_focus = dih.find_focus(..., downsample_factor=4)
   
   # Stage 2: Refine around rough_focus at full resolution
   fine_focus = dih.find_focus_hierarchical(
       ...,
       min_distance=rough_focus - 5,
       max_distance=rough_focus + 5,
       downsample_factor=1  # Full resolution
   )
   ```

## Compatibility

The `downsample_factor` parameter is optional and defaults to 1 (no downsampling), so all existing code continues to work without changes.
