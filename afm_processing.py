# -*- coding: utf-8 -*-
"""
AFM Particle Analysis Script

This script processes AFM height data (from .tiff files with embedded metadata) to:
1. Load and preprocess AFM images
2. Segment particles using watershed segmentation
3. Subtract background and compute morphological statistics
4. Generate visualizations and statistical reports

Author: Johannes
"""

# === IMPORTS ===
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter, binary_erosion
from scipy.interpolate import griddata
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries


def get_particle_areas(mask, exclude_boundary=True):
    """Calculate areas of particles in a binary mask.
    
    Args:
        mask: 2D numpy array representing the binary mask
        exclude_boundary: Whether to exclude particles touching image boundaries
        
    Returns:
        numpy array of particle areas in pixels
    """
    if exclude_boundary:
        exclude_boundary = 1
        newmask = np.ones(np.array(np.shape(mask)) + [2, 2])
        newmask[1:-1, 1:-1] = mask
    else:
        exclude_boundary = 0
        newmask = mask

    # Label connected components
    labeled_mask, num_features = ndimage.label(newmask)
    
    # Calculate sizes of each particle (skipping background and boundary if needed)
    particle_sizes = ndimage.sum_labels(
        newmask,
        labeled_mask,
        range(1 + exclude_boundary, num_features + 1)
    )
    
    return particle_sizes


def count_particles(mask, exclude_boundary=True):
    """Count particles in a binary mask.
    
    Args:
        mask: 2D numpy array representing the binary mask
        exclude_boundary: Whether to exclude particles touching image boundaries
        
    Returns:
        Integer count of particles
    """
    if exclude_boundary:
        exclude_boundary = 1
        newmask = np.ones(np.array(np.shape(mask)) + [2, 2])
        newmask[1:-1, 1:-1] = mask
    else:
        exclude_boundary = 0
        newmask = mask

    # Label connected components
    _, num_features = ndimage.label(newmask)
    
    return num_features - exclude_boundary


def get_particle_number_std(mask, patches=4):
    """Analyze particle number statistics across image patches.
    
    Args:
        mask: 2D numpy array representing the binary mask
        patches: Number of divisions along each axis (creates patches^2 total patches)
        
    Returns:
        tuple: (mean number of particles, standard deviation of particle count)
    """
    # Calculate patch dimensions
    height, width = mask.shape
    patch_height = height // patches
    patch_width = width // patches
    patch_area_factor = (height * width) / (patch_height * patch_width)
    
    # Initialize array to store particle counts
    particle_counts = []
    
    # Iterate through patches
    for i in range(patches):
        for j in range(patches):
            # Extract patch
            patch = mask[
                i * patch_height: (i + 1) * patch_height,
                j * patch_width: (j + 1) * patch_width
            ]
            
            p_count_exbound = count_particles(patch, exclude_boundary=True)
            p_count_wbound = count_particles(patch, exclude_boundary=False)
            particle_counts.append((p_count_exbound + p_count_wbound) / 2)
    
    particle_counts = np.array(particle_counts) * patch_area_factor
    
    return np.mean(particle_counts), np.std(particle_counts)


def get_particle_heights(image, local_maxima, floor=None):
    """Extract particle heights from image at given maxima locations.
    
    Args:
        image: 2D numpy array of height values
        local_maxima: List of (row, col) coordinates of particle centers
        floor: Baseline height to subtract (if None, uses image minimum)
        
    Returns:
        numpy array of particle heights
    """
    if floor is None:
        floor = np.min(image)
    
    heights = [image[r, c] for r, c in local_maxima]
    return np.array(heights) - floor


def interpolate_nan(arr, method='nearest'):
    """Interpolate NaN values in 2D array using specified method.
    
    Args:
        arr: 2D numpy array with NaN values
        method: Interpolation method ('nearest', 'linear', 'cubic')
        
    Returns:
        2D numpy array with NaN values filled
    """
    # Create grid of coordinates
    x = np.arange(arr.shape[1])
    y = np.arange(arr.shape[0])
    xx, yy = np.meshgrid(x, y)
    
    # Mask for valid points
    mask = ~np.isnan(arr)
    
    # Interpolate using griddata
    interpolated_arr = griddata(
        (xx[mask], yy[mask]),
        arr[mask],
        (xx, yy),
        method=method
    )
    
    return interpolated_arr


def find_local_maxima(image, min_distance=2):
    """Identify local maxima in image for watershed segmentation.
    
    Args:
        image: 2D numpy array
        min_distance: Minimum distance between maxima
        
    Returns:
        tuple: (coordinates of maxima, marker image)
    """
    local_maxima = peak_local_max(image, min_distance=min_distance, exclude_border=0)
    markers = np.zeros_like(image, dtype=np.int32)
    
    for i, (r, c) in enumerate(local_maxima, 1):
        markers[r, c] = i
        
    return local_maxima, markers


def individual_threshold(image, mask, bpratio=1):
    """Apply adaptive thresholding to distinguish individual particles.
    
    Args:
        image: 2D numpy array of height values
        mask: Initial segmentation mask
        bpratio: Weighting between border and peak values (0=peak only, 1=border only)
        
    Returns:
        Refined binary mask
    """
    newmask = np.zeros_like(mask)
    
    for i in range(1, np.max(mask) + 1):
        segment = mask == i
        peak = np.max(image * segment)
        bordermask = np.bitwise_xor(
            segment,
            binary_erosion(segment, iterations=3, border_value=1)
        )
        bordermax = np.max(image * bordermask)
        threshold = bpratio * bordermax + (1 - bpratio) * peak
        newmask += (image * segment) > threshold
        
    return newmask


def perform_watershed(image, markers):
    """Perform watershed segmentation on image using given markers.
    
    Args:
        image: 2D numpy array
        markers: Marker image for watershed
        
    Returns:
        tuple: (segmentation labels, boundary mask)
    """
    segmentation = watershed(-image, markers)
    watershed_boundaries = find_boundaries(segmentation, mode='inner')
    
    return segmentation, watershed_boundaries


def interpolate_background(image, watershed_boundaries):
    """Interpolate background using watershed boundaries.
    
    Args:
        image: 2D numpy array
        watershed_boundaries: Binary mask of watershed boundaries
        
    Returns:
        2D numpy array of interpolated background
    """
    watershed_boundaries_nan = watershed_boundaries * image
    watershed_boundaries_nan[~watershed_boundaries] = np.nan
    
    # Two-pass interpolation for smooth results
    interpolated = interpolate_nan(watershed_boundaries_nan, method='linear')
    interpolated = interpolate_nan(interpolated)
    
    return interpolated


def area2diam(area):
    """Convert area to equivalent circular diameter.
    
    Args:
        area: Area value(s)
        
    Returns:
        Equivalent diameter(s)
    """
    return 2 * np.sqrt(area / np.pi)


def plot_save_image(image, filename, cmap='afmhot', vmin=0, vmax=10):
    """Helper function to plot and save images.
    
    Args:
        image: 2D numpy array to plot
        filename: Base filename (without extension)
        cmap: Colormap to use
        vmin/vmax: Color scale limits
    """
    
    plt.figure(figsize=(5, 4))
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    
    if cmap != 'gray':
        plt.colorbar()
    
    plt.xticks([])
    plt.yticks([])
    
    # plt.savefig(f'{filename}.pdf', dpi=400, bbox_inches='tight')
    plt.savefig(f'{filename}.png', dpi=400, bbox_inches='tight')
    plt.show()
    
    
def calculate_and_save_statistics(image, mask, local_maxima, NM_PER_PIXEL, filename="stats"):
    """Calculate particle statistics and save results to file.
    
    Args:
        image: Original AFM image (2D numpy array)
        mask: Segmented particle mask (2D binary array)
        local_maxima: Coordinates of particle centers
        filename_prefix: Prefix for output filename
        
    Returns:
        Dictionary containing all calculated statistics
    """
    # Calculate basic metrics
    pixel_area = NM_PER_PIXEL ** 2
    image_area_um = image.size * pixel_area / 1e6
    
    particle_number = count_particles(mask)
    mean_particles, std_particles = get_particle_number_std(mask)
    
    mean_density = mean_particles / image_area_um
    std_density = std_particles / image_area_um
    
    particle_heights = get_particle_heights(image, local_maxima)
    particle_areas = get_particle_areas(mask)
    particle_areas_nm = particle_areas * pixel_area
    particle_diameters = area2diam(particle_areas_nm)
    
    # Compile statistics into dictionary
    stats = {
        'particle_number': particle_number,
        'mean_density': mean_density,
        'std_density': std_density,
        'mean_area_px': np.mean(particle_areas),
        'std_area_px': np.std(particle_areas),
        'mean_area_nm': np.mean(particle_areas_nm),
        'std_area_nm': np.std(particle_areas_nm),
        'mean_diameter': np.mean(particle_diameters),
        'std_diameter': np.std(particle_diameters),
        'mean_height': np.mean(particle_heights),
        'std_height': np.std(particle_heights)
    }
    
    # Create output strings
    header = "=== AFM Particle Analysis Statistics ==="
    summary = [
        f"\nParticle number (particles/image): {stats['particle_number']}",
        f"Particle density (1/µm²): {stats['mean_density']:.0f} ± {stats['std_density']:.0f}",
        f"Particle area (pixels): {stats['mean_area_px']:.1f} ± {stats['std_area_px']:.1f}",
        f"Particle area (nm²): {stats['mean_area_nm']:.1f} ± {stats['std_area_nm']:.1f}",
        f"Particle diameter (nm): {stats['mean_diameter']:.1f} ± {stats['std_diameter']:.1f}",
        f"Particle height (nm): {stats['mean_height']:.1f} ± {stats['std_height']:.1f}",
    ]
    
    # Print to console
    print(header)
    print("\n".join(summary))
    
    # Save to file
    with open(f"{filename}.txt", 'w') as f:
        f.write(header + "\n")
        f.write("\n".join(summary))
        f.write("\n\n=== Detailed Measurements ===")
        f.write("\nParticle heights (nm): " + ", ".join(f"{h:.2f}" for h in particle_heights))
        f.write("\nParticle areas (nm²): " + ", ".join(f"{a:.1f}" for a in particle_areas_nm))
        f.write("\nParticle diameters (nm): " + ", ".join(f"{d:.1f}" for d in particle_diameters))
    
    return stats


def main():
    """Main processing pipeline."""
    
    NM_PER_PIXEL = 500 / 2048  # Physical pixel size in nanometers
    
    # Load the AFM image
    file_path = "AFM/AFM_FEcatalyst_1p7nm.tiff"
    image = imread(file_path)
    
    # Scale the image (optional)
    average_height = 2 * np.std(image)
    image_scaled = image - np.mean(image) + average_height
    image_scaled = (image_scaled > 0) * image_scaled
    
    # Smooth the image
    image_smoothed = gaussian_filter(image_scaled, sigma=10)
    
    # Boundary finding with watershed segmentation
    local_maxima, markers = find_local_maxima(image_smoothed)
    watershed_seg, watershed_boundaries = perform_watershed(image_scaled, markers)
    interpolated_background = interpolate_background(image_smoothed, watershed_boundaries)
    
    # Background subtraction
    image_without_background = image_scaled - interpolated_background
    image_without_background = gaussian_filter(image_without_background, sigma=10)
    
    # Final segmentation
    segmented_particle_mask = individual_threshold(image_without_background, watershed_seg, 0.8)
    
    # Visualization
    plot_save_image(image_scaled, 'AFMmap', vmin=0, vmax=10)
    plot_save_image(segmented_particle_mask, 'segmented', cmap='gray', vmin=0, vmax=1)
    
    # Statistics calculation
    calculate_and_save_statistics(image_smoothed, segmented_particle_mask, local_maxima, NM_PER_PIXEL)


if __name__ == "__main__":
    main()