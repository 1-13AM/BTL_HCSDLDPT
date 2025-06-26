import numpy as np
from typing import Callable
from scipy.spatial.distance import cosine, cityblock, chebyshev
from scipy.stats import wasserstein_distance

def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes the Euclidean (L2) distance between two vectors.
    Lower value means more similar, so we're gonna return the negative of the distance
    """
    return -((vec1 - vec2)**2).sum()

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes cosine similarity between two vectors.
    """
    # Handle zero vectors
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    return 1 - cosine(vec1, vec2)

def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes the manhattan_distance distance between two vectors.
    Lower value means more similar, so we're gonna return the negative of the distance
    """
    return - np.abs(vec1 - vec2).sum()

def weighted_combination(vec1: np.ndarray, vec2: np.ndarray, 
                         color_weight: float = 0.5, 
                         lbp_weight: float = 0.5,
                         hsv_bins: int = 144,  # 16*3*3 
                         hsv_similarity_metric: Callable = euclidean_distance,
                         lbp_similarity_metric: Callable = euclidean_distance) -> float:
    """
    Uses weighted combination of distances for different feature types.
    
    Args:
        vec1, vec2: Feature vectors (concatenated HSV and LBP)
        color_weight: Weight for color histogram distance (0 to 1)
        hsv_bins: Number of bins in the HSV histogram
        lbp_weight: Weight for LBP histogram distance (0 to 1)
        
    Returns:
        Weighted distance (lower is more similar)
    """
    hsv1, lbp1 = vec1[:hsv_bins], vec1[hsv_bins:]
    hsv2, lbp2 = vec2[:hsv_bins], vec2[hsv_bins:]
    
    color_dist = hsv_similarity_metric(hsv1, hsv2)
    texture_dist = lbp_similarity_metric(lbp1, lbp2)
    
    # combine with weights
    return color_weight * color_dist + lbp_weight * texture_dist
