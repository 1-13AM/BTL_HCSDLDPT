import cv2
import numpy as np
from utils.image_processing import load_image
from utils.edge_detection import extract_object_mask_canny, extract_object_mask_thresholding

def compute_uniform_ltp(image, mode='grayscale', radius=1, neighbors=8, threshold=5):
    """
    Compute uniform Local Ternary Pattern on an image
    
    Args:
        image: Input image as numpy array
        mode: 'grayscale' or 'color' to process image
        radius: Radius around each pixel to consider neighbors
        neighbors: Number of neighbors to consider (typically 8)
        threshold: Threshold for ternary comparison (default 5)
    
    Returns:
        LTP image as numpy array
    """
    uniform_patterns = get_uniform_ternary_patterns(neighbors)
    
    if mode == 'grayscale':
        # convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # compute LTP
        return uniform_ltp(gray, radius, neighbors, uniform_patterns, threshold)
    
    elif mode == 'color':
        # process each channel separately
        if len(image.shape) != 3:
            raise ValueError("Expected color image for 'color' mode")
        
        ltp_channels = []
        for channel in range(image.shape[2]):
            ltp_channel = uniform_ltp(image[:,:,channel], radius, neighbors, uniform_patterns, threshold)
            ltp_channels.append(ltp_channel)
        
        # concatenate histograms from each channel
        return np.concatenate(ltp_channels, axis=1) if len(ltp_channels) > 0 else np.array([])
    
    else:
        raise ValueError("Mode must be 'grayscale' or 'color'")

def uniform_ltp(image, radius, neighbors, uniform_patterns, threshold):
    """Compute uniform LTP for a single channel image"""
    rows, cols = image.shape
    result = np.zeros((rows, cols), dtype=np.uint8)
    
    for y in range(radius, rows - radius):
        for x in range(radius, cols - radius):
            center = image[y, x]
            upper_pattern = 0
            lower_pattern = 0
            
            for n in range(neighbors):
                # calculate neighbor coordinates
                theta = 2 * np.pi * n / neighbors
                x_n = x + int(round(radius * np.cos(theta)))
                y_n = y + int(round(radius * np.sin(theta)))
                
                neighbor_val = image[y_n, x_n]
                
                # ternary comparison with threshold
                if neighbor_val >= center + threshold:
                    upper_pattern |= (1 << n)
                elif neighbor_val <= center - threshold:
                    lower_pattern |= (1 << n)
            
            # convert ternary to binary representation
            combined_pattern = encode_ternary_pattern(upper_pattern, lower_pattern, neighbors)
            
            # map to uniform pattern
            result[y, x] = uniform_patterns.get(combined_pattern, len(uniform_patterns))
    
    return result

def encode_ternary_pattern(upper_pattern, lower_pattern, neighbors):
    """Encode ternary pattern into a single value for uniform mapping"""
    return (upper_pattern << neighbors) | lower_pattern

def get_uniform_ternary_patterns(neighbors):
    """Generate mapping of ternary patterns to uniform LTP values"""
    uniform_patterns = {}
    
    def ternary_transitions(upper_pattern, lower_pattern, neighbors):
        """Count transitions in ternary pattern"""
        ternary_values = []
        
        for n in range(neighbors):
            if (upper_pattern >> n) & 1:
                ternary_values.append(1)
            elif (lower_pattern >> n) & 1:
                ternary_values.append(-1)
            else:
                ternary_values.append(0)
        
        ternary_circular = ternary_values + [ternary_values[0]]
        transitions = sum(v1 != v2 for v1, v2 in zip(ternary_circular, ternary_circular[1:]))
        return transitions
    
    # uniform patterns have at most 2 transitions
    uniform_val = 0
    for upper in range(2**neighbors):
        for lower in range(2**neighbors):
            # ensure upper and lower patterns don't overlap
            if upper & lower == 0:
                if ternary_transitions(upper, lower, neighbors) <= 2:
                    combined = encode_ternary_pattern(upper, lower, neighbors)
                    uniform_patterns[combined] = uniform_val
                    uniform_val += 1
    
    return uniform_patterns

def compute_ltp_on_object(image: np.ndarray, mask: np.ndarray, radius=1, neighbors=8, threshold=5, method='uniform') -> np.ndarray:
    """
    Compute LTP only on the object region defined by the mask.
    
    Args:
        image (np.ndarray): Input image
        mask (np.ndarray): Binary mask where object pixels are 255
        radius (int): LTP radius
        neighbors (int): Number of neighbors
        threshold (int): Threshold for ternary comparison
        method (str): 'uniform' for uniform LTP
        
    Returns:
        np.ndarray: Histogram of LTP features for the object region
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # apply LTP to the entire image
    if method == 'uniform':
        ltp = compute_uniform_ltp(gray, mode='grayscale', radius=radius, neighbors=neighbors, threshold=threshold)
        uniform_patterns = get_uniform_ternary_patterns(neighbors)
        n_bins = len(uniform_patterns) + 1  # +1 for non-uniform patterns
    else:
        raise ValueError("Only 'uniform' method is implemented for LTP")
    
    # extract object pixels
    object_pixels = ltp[mask > 0]
    histogram, _ = np.histogram(object_pixels, bins=n_bins, range=(0, n_bins-1))
    
    return histogram

def extract_ltp_features(image_path: str, radius: int = 1, neighbors: int = 8, threshold: int = 2, method: str = 'uniform', normalize: bool = True, **edge_detection_kwargs):
    """
    Extract Local Ternary Pattern features from an image
    
    Args:
        image_path (str): Path to the image
        radius (int): LTP radius
        neighbors (int): Number of neighbors
        threshold (int): Threshold for ternary comparison
        method (str): 'uniform' for uniform LTP
        normalize (bool): Whether to normalize the histogram
        **edge_detection_kwargs: Additional arguments for edge detection
        
    Returns:
        np.ndarray: LTP feature histogram
    """
    rgb_img = load_image(image_path)
    edge_detection_strategy = edge_detection_kwargs.pop('edge_detection_strategy', 'canny')
    
    if edge_detection_strategy == 'canny':
        mask = extract_object_mask_canny(rgb_img, **edge_detection_kwargs)
    elif edge_detection_strategy == 'thresholding':
        mask = extract_object_mask_thresholding(rgb_img, **edge_detection_kwargs)
    
    # compute LTP on the object region
    ltp_hist = compute_ltp_on_object(rgb_img, mask, radius=radius, neighbors=neighbors, threshold=threshold, method=method)
    
    if normalize:
        ltp_hist = ltp_hist.astype('float32') / ltp_hist.sum()
    
    return ltp_hist