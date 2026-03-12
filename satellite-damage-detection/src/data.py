"""Data loading and preprocessing utilities."""

import numpy as np
import pandas as pd
import rasterio
from pathlib import Path


class SentinelDataLoader:
    """Load Sentinel-2 imagery from files."""
    
    def __init__(self, data_dir: str):
        """Initialize data loader.
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = Path(data_dir)
    
    def load_scene(self, scene_path: str) -> np.ndarray:
        """Load a Sentinel-2 scene from file.
        
        Args:
            scene_path: Path to scene file
            
        Returns:
            numpy array of shape (H, W, bands)
        """
        with rasterio.open(scene_path) as src:
            data = src.read()
        return np.transpose(data, (1, 2, 0))
    
    def load_metadata(self, csv_path: str) -> pd.DataFrame:
        """Load scene metadata.
        
        Args:
            csv_path: Path to metadata CSV
            
        Returns:
            DataFrame with scene information
        """
        return pd.read_csv(csv_path)


def apply_cloud_mask(image: np.ndarray, scl_band: np.ndarray) -> np.ndarray:
    """Apply cloud masking using Scene Classification band.
    
    Args:
        image: Image array (H, W, bands)
        scl_band: Scene classification band (H, W)
        
    Returns:
        Masked image with clouds set to NaN
    """
    # Valid pixels: 4 (vegetation), 5 (not vegetated), 6 (water)
    valid_pixels = np.isin(scl_band, [4, 5, 6])
    
    image_masked = image.copy().astype(float)
    image_masked[~valid_pixels] = np.nan
    
    return image_masked


def normalize_bands(image: np.ndarray) -> np.ndarray:
    """Normalize image bands to [0, 1].
    
    Args:
        image: Image array with values [0, 10000]
        
    Returns:
        Normalized image [0, 1]
    """
    return np.clip(image / 10000.0, 0, 1)


def create_patches(image: np.ndarray, patch_size: int = 128, stride: int = 64) -> list:
    """Extract overlapping patches from image.
    
    Args:
        image: Image array (H, W, bands)
        patch_size: Size of patches (patch_size x patch_size)
        stride: Stride for patch extraction
        
    Returns:
        List of patches
    """
    h, w, c = image.shape
    patches = []
    
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
    
    return patches
