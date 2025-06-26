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
    return -np.abs(vec1 - vec2).sum()

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

def chi2_distance(h1, h2, eps=1e-10):
    h1, h2 = np.asarray(h1), np.asarray(h2)
    return 0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + eps))

def scd_distance(tokens1, tokens2, alpha=0.5, N=5):
    def extract_vec(tokens):
        return np.array([[t['max_curvature'], t['orientation']] for t in tokens[:N]])
    
    v1 = extract_vec(tokens1)
    v2 = extract_vec(tokens2)
    
    if len(v1) != len(v2):
        min_len = min(len(v1), len(v2))
        v1 = v1[:min_len]
        v2 = v2[:min_len]
    diff = v1 - v2
    return np.mean(np.sqrt(diff[:, 0]**2 + alpha * diff[:, 1]**2))

def cosine_distance(v1, v2):
    v1, v2 = np.asarray(v1), np.asarray(v2)
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0: return 1.0
    return 1.0 - dot / norm

def clf_distance(clf1, clf2):
    def flatten(clf_dict):
        return np.concatenate([clf_dict[k] for k in sorted(clf_dict.keys())])
    
    v1 = flatten(clf1)
    v2 = flatten(clf2)
    return np.linalg.norm(v1 - v2)

def shape_similarity_metric(feat1, feat2, weights=None):
    """
    Calculates a NORMALIZED similarity score between two shape feature dictionaries.
    Each internal distance is converted to a similarity score in the range [0, 1]
    before being combined. A higher return value means more similar.
    """
    if weights is None:
        weights = {'cch': 0.25, 'scd': 0.25, 'bas': 0.25, 'clf': 0.25}

    similarities = {}
    
    # --- CCH Similarity ---
    try:
        d_cch = chi2_distance(feat1['cch'], feat2['cch'])
        similarities['cch'] = 1 / (1 + d_cch)  # Normalize
    except (KeyError, TypeError, ValueError):
        similarities['cch'] = 0  # Features not comparable or missing

    # --- SCD Similarity ---
    try:
        d_scd = scd_distance(feat1['scd_tokens'], feat2['scd_tokens'])
        similarities['scd'] = 1 / (1 + d_scd)  # Normalize
    except (KeyError, TypeError, ValueError):
        similarities['scd'] = 0

    # --- BAS Similarity ---
    try:
        # Cosine distance is already in [0, 2]. Similarity = 1 - distance
        d_bas = cosine_distance(feat1['bas_vector_fd'], feat2['bas_vector_fd'])
        similarities['bas'] = 1 - (d_bas / 2) # Normalize to [0, 1]
    except (KeyError, TypeError, ValueError):
        similarities['bas'] = 0

    # --- CLF Similarity ---
    try:
        d_clf = clf_distance(feat1['clf'], feat2['clf'])
        similarities['clf'] = 1 / (1 + d_clf)  # Normalize
    except (KeyError, TypeError, ValueError):
        similarities['clf'] = 0

    # --- Weighted Combination of Similarities ---
    total_similarity = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        if key in similarities:
            total_similarity += similarities[key] * weight
            total_weight += weight
    
    # Avoid division by zero if no features were matched
    if total_weight == 0:
        return 0.0
        
    return total_similarity / total_weight