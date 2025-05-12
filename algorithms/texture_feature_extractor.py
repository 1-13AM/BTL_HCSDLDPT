from utils.edge_detection import extract_object_mask_canny, extract_object_mask_thresholding
from utils.image_processing import load_image
import numpy as np
import cv2

def compute_uniform_lbp(image, mode='grayscale', radius=1, neighbors=8):
    """
    Compute uniform Local Binary Pattern on an image
    
    Args:
        image: Input image as numpy array
        mode: 'grayscale' or 'color' to process image
        radius: Radius around each pixel to consider neighbors
        neighbors: Number of neighbors to consider (typically 8)
    
    Returns:
        LBP image as numpy array
    """

    uniform_patterns = get_uniform_patterns(neighbors)
    
    if mode == 'grayscale':
        # convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # compute LBP
        return uniform_lbp(gray, radius, neighbors, uniform_patterns)
    
    elif mode == 'color':
        # process each channel separately
        if len(image.shape) != 3:
            raise ValueError("Expected color image for 'color' mode")
        
        lbp_channels = []
        for channel in range(image.shape[2]):
            lbp_channel = uniform_lbp(image[:,:,channel], radius, neighbors, uniform_patterns, n_bins)
            lbp_channels.append(lbp_channel)
        
        # concatenate histograms from each channel
        return np.concatenate(lbp_channels, axis=1) if len(lbp_channels) > 0 else np.array([])
    
    else:
        raise ValueError("Mode must be 'grayscale' or 'color'")

def uniform_lbp(image, radius, neighbors, uniform_patterns):
    """Compute LBP for a single channel image"""
    rows, cols = image.shape
    result = np.zeros((rows, cols), dtype=np.uint8)
    
    for y in range(radius, rows - radius):
        for x in range(radius, cols - radius):
            center = image[y, x]
            pattern = 0
            
            for n in range(neighbors):
                # calculate neighbor coordinates
                theta = 2 * np.pi * n / neighbors
                x_n = x + int(round(radius * np.cos(theta)))
                y_n = y + int(round(radius * np.sin(theta)))
                
                # compare neighbor with center
                if image[y_n, x_n] >= center:
                    pattern |= (1 << n)
            
            # map to uniform pattern
            result[y, x] = uniform_patterns.get(pattern, neighbors * (neighbors - 1) + 2)
    
    return result

def compute_regular_lbp(image, radius, neighbors):
    """Compute regular LBP (non-uniform) for a grayscale image"""
    rows, cols = image.shape
    result = np.zeros((rows, cols), dtype=np.uint8)
    
    for y in range(radius, rows - radius):
        for x in range(radius, cols - radius):
            center = image[y, x]
            pattern = 0
            
            for n in range(neighbors):
                # calculate neighbor coordinates
                # we move in a counter-clockwise fashion (starting from the eastmost point)
                theta = 2 * np.pi * n / neighbors
                x_n = x + int(round(radius * np.cos(theta)))
                y_n = y + int(round(radius * np.sin(theta)))
                
                # compare neighbor with center
                if image[y_n, x_n] >= center:
                    pattern |= (1 << n)
            
            result[y, x] = pattern
    
    return result

def get_uniform_patterns(neighbors):
    """Generate mapping of patterns to uniform LBP values"""
    uniform_patterns = {}
    
    def bit_transitions(pattern):
        binary = bin(pattern)[2:].zfill(neighbors)
        binary_circular = binary + binary[0]
        return sum(b1 != b2 for b1, b2 in zip(binary_circular, binary_circular[1:]))
    
    # uniform patterns have at most 2 bit transitions
    uniform_val = 0
    for pattern in range(2**neighbors):
        if bit_transitions(pattern) <= 2:
            uniform_patterns[pattern] = uniform_val
            uniform_val += 1
    
    return uniform_patterns

def compute_lbp_on_object(image: np.ndarray, mask: np.ndarray, radius=1, neighbors=8, method='uniform') -> np.ndarray:
    """
    Compute LBP only on the object region defined by the mask.
    
    Args:
        image (np.ndarray): Input image
        mask (np.ndarray): Binary mask where object pixels are 255
        radius (int): LBP radius
        neighbors (int): Number of neighbors
        method (str): 'uniform' for uniform LBP
        
    Returns:
        np.ndarray: Histogram of LBP features for the object region
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # apply LBP to the entire image
    if method == 'uniform':
        # for uniform patterns, number of possible output values is neighbors*(neighbors-1)+3
        lbp = compute_uniform_lbp(gray, mode='grayscale', radius=radius, neighbors=neighbors)
        n_bins = neighbors*(neighbors-1)+3
    else:
        # for regular LBP, number of possible output values is 2^neighbors
        lbp = compute_regular_lbp(gray, radius, neighbors)
        n_bins = 2**neighbors
    
    # take out the background pixels
    object_pixels = lbp[mask > 0]
    
    # calculate histogram
    histogram, _ = np.histogram(object_pixels, bins=n_bins, range=(0, n_bins-1))
    
    return histogram

def extract_lbp_features(image_path: str, radius: int = 1, neighbors: int = 8, method: str = 'uniform', normalize: bool = True, **edge_detection_kwargs):
    """
    """
    mask = None
    rgb_img = load_image(image_path)
    
    edge_detection_strategy = edge_detection_kwargs.get('edge_detection_strategy', 'canny')
    if edge_detection_strategy == 'canny':
        mask = extract_object_mask_canny(rgb_img, **edge_detection_kwargs)
    elif edge_detection_strategy == 'thresholding':
        mask = extract_object_mask_thresholding(rgb_img, **edge_detection_kwargs)
    
    # extract the object mask
    mask = extract_object_mask_canny(rgb_img, **edge_detection_kwargs)
    
    # compute LBP on the object region
    lbp_hist = compute_lbp_on_object(rgb_img, mask, radius=radius, neighbors=neighbors, method=method)
    
    if normalize:
        lbp_hist = lbp_hist.astype('float32') / lbp_hist.sum()
    
    return lbp_hist # array of shape [neighbors * (neighbors - 1) + 3] if method=='uniform' else 2**neighbors