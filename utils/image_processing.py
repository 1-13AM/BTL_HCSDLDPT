import cv2
import numpy as np
import os
from PIL import Image

def load_image(image_path: str) -> np.ndarray:
    """Loads an image using PIL and converts to numpy array for OpenCV use.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        np.ndarray: Image as a NumPy array in RGB format.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Use PIL to load the image
    pil_image = Image.open(image_path)
    # Convert PIL image to numpy array
    image_np = np.array(pil_image)
        
    return image_np

def convert_rgb_to_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to HSV color space.
    """
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    return image_hsv