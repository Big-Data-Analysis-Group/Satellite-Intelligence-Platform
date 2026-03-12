"""Spectral index computation and feature engineering."""

import numpy as np


class SpectralIndices:
    """Compute spectral indices from Sentinel-2 bands."""
    
    # Sentinel-2 band indices
    BANDS = {
        'B02': 0,  # Blue
        'B03': 1,  # Green
        'B04': 2,  # Red
        'B08': 3,  # NIR
        'B11': 4,  # SWIR
        'B12': 5,  # SWIR2
        'SCL': 6   # Scene Classification
    }
    
    @staticmethod
    def ndvi(image: np.ndarray) -> np.ndarray:
        """Compute Normalized Difference Vegetation Index.
        
        NDVI = (NIR - RED) / (NIR + RED)
        
        Args:
            image: Image array where bands in order: B02, B03, B04, B08, B11, B12
            
        Returns:
            NDVI array
        """
        red = image[..., 2].astype(float)
        nir = image[..., 3].astype(float)
        return (nir - red) / (nir + red + 1e-8)
    
    @staticmethod
    def ndbi(image: np.ndarray) -> np.ndarray:
        """Compute Normalized Difference Built-up Index.
        
        NDBI = (SWIR - NIR) / (SWIR + NIR)
        
        Args:
            image: Image array where bands in order: B02, B03, B04, B08, B11, B12
            
        Returns:
            NDBI array
        """
        swir = image[..., 4].astype(float)
        nir = image[..., 3].astype(float)
        return (swir - nir) / (swir + nir + 1e-8)
    
    @staticmethod
    def bsi(image: np.ndarray) -> np.ndarray:
        """Compute Bare Soil Index.
        
        BSI = (SWIR + RED - NIR - BLUE) / (SWIR + RED + NIR + BLUE)
        
        Args:
            image: Image array where bands in order: B02, B03, B04, B08, B11, B12
            
        Returns:
            BSI array
        """
        blue = image[..., 0].astype(float)
        red = image[..., 2].astype(float)
        nir = image[..., 3].astype(float)
        swir = image[..., 4].astype(float)
        
        numerator = swir + red - nir - blue
        denominator = swir + red + nir + blue
        return numerator / (denominator + 1e-8)
    
    @staticmethod
    def mndwi(image: np.ndarray) -> np.ndarray:
        """Compute Modified Normalized Difference Water Index.
        
        MNDWI = (GREEN - SWIR) / (GREEN + SWIR)
        
        Args:
            image: Image array where bands in order: B02, B03, B04, B08, B11, B12
            
        Returns:
            MNDWI array
        """
        green = image[..., 1].astype(float)
        swir = image[..., 4].astype(float)
        return (green - swir) / (green + swir + 1e-8)


def compute_patch_features(patch: np.ndarray) -> dict:
    """Compute statistical features for a patch.
    
    Args:
        patch: Image patch (patch_size, patch_size, bands)
        
    Returns:
        Dictionary of computed features
    """
    si = SpectralIndices()
    
    ndvi = si.ndvi(patch)
    ndbi = si.ndbi(patch)
    bsi = si.bsi(patch)
    mndwi = si.mndwi(patch)
    
    # Remove NaN values for statistics
    ndvi_valid = ndvi[~np.isnan(ndvi)]
    ndbi_valid = ndbi[~np.isnan(ndbi)]
    bsi_valid = bsi[~np.isnan(bsi)]
    
    features = {
        'ndvi_mean': np.nanmean(ndvi),
        'ndvi_std': np.nanstd(ndvi),
        'ndvi_min': np.nanmin(ndvi),
        'ndvi_max': np.nanmax(ndvi),
        
        'ndbi_mean': np.nanmean(ndbi),
        'ndbi_std': np.nanstd(ndbi),
        'ndbi_min': np.nanmin(ndbi),
        'ndbi_max': np.nanmax(ndbi),
        
        'bsi_mean': np.nanmean(bsi),
        'bsi_std': np.nanstd(bsi),
        
        'mndwi_mean': np.nanmean(mndwi),
        
        'valid_pixel_ratio': len(ndvi_valid) / ndvi.size
    }
    
    return features


def compute_change_index(image_t1: np.ndarray, image_t2: np.ndarray) -> np.ndarray:
    """Compute temporal change index between two images.
    
    Args:
        image_t1: Image at time t1
        image_t2: Image at time t2
        
    Returns:
        Change index (higher = more damage likely)
    """
    si = SpectralIndices()
    
    ndvi_t1 = si.ndvi(image_t1)
    ndvi_t2 = si.ndvi(image_t2)
    
    # Negative NDVI change indicates vegetation loss
    ndvi_change = ndvi_t1 - ndvi_t2
    
    return np.nan_to_num(ndvi_change, 0)
