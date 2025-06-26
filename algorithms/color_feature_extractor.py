from utils.edge_detection import extract_object_mask_canny, extract_object_mask_thresholding
from utils.image_processing import load_image, convert_rgb_to_hsv
import cv2
import numpy as np
from typing import Optional

def compute_hsv_histogram(image_hsv: np.ndarray, mask: Optional[np.ndarray] = None, h_bins: int = 8, s_bins: int = 3, v_bins: int = 3, normalize: bool = True):
    """Computes a histogram of HSV values from an image.
    We're gonna allocate more bins to the hue dimension, and the least bins for the value dimension (this dimension is sensitive to lighting conditions)
    Args:
        image_hsv (np.ndarray): HSV image as a NumPy array.
        h_bins (int): Number of bins for the Hue channel (0-179 in OpenCV).
        s_bins (int): Number of bins for the Saturation channel (0-255 in OpenCV).
        v_bins (int): Number of bins for the Value channel (0-255 in OpenCV).
        normalize (bool): Whether to normalize the histogram.
        
    Returns:
        np.ndarray: Flattened histogram of HSV values of shape [h_bins * s_bins * v_bins]
    """
    
    # define the ranges for each channel
    h_range = [0, 180]  # OpenCV uses 0-179 for H
    s_range = [0, 256]  # OpenCV uses 0-255 for S
    v_range = [0, 256]  # OpenCV uses 0-255 for V
    
    # define the bin sizes for each channel
    bins = [h_bins, s_bins, v_bins]
    ranges = [h_range, s_range, v_range]
    
    # compute the histogram
    hist = cv2.calcHist([image_hsv], [0, 1, 2], mask, bins, h_range + s_range + v_range)
    
    if normalize:
        hist /= hist.sum()
    hist_flat = hist.flatten()
    return hist_flat

def extract_hsv_histogram(image_path: str, edge_detection_strategy: str = 'canny', h_bins: int = 16, s_bins: int = 3, v_bins: int = 3, normalize: bool = True, **edge_detection_kwargs) -> np.ndarray:
    """
    """
    mask = None
    rgb_img = load_image(image_path)
    if edge_detection_strategy == 'canny':
        mask = extract_object_mask_canny(rgb_img, **edge_detection_kwargs)
    elif edge_detection_strategy == 'thresholding':
        mask = extract_object_mask_thresholding(rgb_img, **edge_detection_kwargs)
    # else:
    #     raise ValueError("edge_detection_strategy must either be 'canny' or 'thresholding'")
    
    hsv_img = convert_rgb_to_hsv(rgb_img)
    
    hsv_hist = compute_hsv_histogram(hsv_img, 
                                     mask, 
                                     h_bins=h_bins, 
                                     s_bins=s_bins, 
                                     v_bins=v_bins, 
                                     normalize=True)

    if normalize:
        hsv_hist /= hsv_hist.sum()
        
    return hsv_hist # array of shape [h_bins x s_bins x v_bins]

def extract_hsv_histogram_features(image_path: str, h_bins: int = 16, s_bins: int = 3, v_bins: int = 3, normalize: bool = True, **edge_detection_kwargs) -> np.ndarray:
    """
    """
    mask = None
    rgb_img = load_image(image_path)
    
    edge_detection_strategy = edge_detection_kwargs.get('edge_detection_strategy', 'canny')
    if edge_detection_strategy == 'canny':
        mask = extract_object_mask_canny(rgb_img, **edge_detection_kwargs)
    elif edge_detection_strategy == 'thresholding':
        mask = extract_object_mask_thresholding(rgb_img, **edge_detection_kwargs)
    # else:
    #     raise ValueError("edge_detection_strategy must either be 'canny' or 'thresholding'")
    
    hsv_img = convert_rgb_to_hsv(rgb_img)
    
    hsv_hist = compute_hsv_histogram(hsv_img, 
                                     mask, 
                                     h_bins=h_bins, 
                                     s_bins=s_bins, 
                                     v_bins=v_bins, 
                                     normalize=True)

    return hsv_hist # array of shape [h_bins x s_bins x v_bins]