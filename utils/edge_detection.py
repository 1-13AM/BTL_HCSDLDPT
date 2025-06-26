import cv2
import numpy as np

def extract_object_mask_thresholding(image: np.ndarray, threshold: int = 0) -> np.ndarray:
    """
    Create a binary mask of the largest non-black object.
    
    Args:
        image (np.ndarray): Input image (RGB)
        threshold (int): Pixel values below this are considered background
        
    Returns:
        np.ndarray: Binary mask where object pixels are 255 and background is 0
    """
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # pixel with values below or equal to threshold become black
    # pixel with values above threshold become white
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # find contours
    # cv2.RETR_EXTERNAL helps ignoring any holes or internal contours inside the objects
    # cv2.CHAIN_APRROX_SIMPLE helps saving memory
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank mask
    mask = np.zeros_like(gray)
    
    if contours:
        # find the largest contours, this corresponds to the only object in the image
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    
    return mask

def extract_object_mask_canny(image: np.ndarray, low_threshold: int = 30, high_threshold: int = 100) -> np.ndarray:
    """
    Create a binary mask of the largest object using Canny edge detection.
    
    Args:
        image (np.ndarray): Input image (RGB)
        low_threshold (int): Lower threshold for Canny edge detection
        high_threshold (int): Higher threshold for Canny edge detection
        
    Returns:
        np.ndarray: Binary mask where object pixels are 255 and background is 0
    """
    
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Dilate edges to ensure they connect
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # find contours from the edges
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # find the largest contour then fill the contour in the mask
    mask = np.zeros_like(gray)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    
    return mask