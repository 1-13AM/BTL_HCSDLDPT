import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def read_and_binarize_image(image_path, threshold=10):
    """Read grayscale image and convert to binary."""
    img = Image.open(image_path).convert('L')
    img = np.array(img)
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    binary = (img > threshold).astype(np.uint8)
    return img, binary


def erode_manual(image):
    h, w = image.shape
    eroded = np.zeros_like(image)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]
            if np.all(region == 1):
                eroded[i, j] = 1
    return eroded


def subtract_manual(a, b):
    return np.where((a == 1) & (b == 0), 1, 0)


def extract_boundary(image_path):
    gray, binary = read_and_binarize_image(image_path)
    eroded = erode_manual(binary)
    boundary = subtract_manual(binary, eroded)
    return gray, binary, eroded, boundary


def plot_boundary_stages(gray, binary, eroded, boundary):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Grayscale')

    plt.subplot(1, 4, 2)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image')

    plt.subplot(1, 4, 3)
    plt.imshow(eroded, cmap='gray')
    plt.title('Eroded Image')

    plt.subplot(1, 4, 4)
    plt.imshow(boundary, cmap='gray')
    plt.title('Extracted Boundary')

    plt.tight_layout()
    plt.show()