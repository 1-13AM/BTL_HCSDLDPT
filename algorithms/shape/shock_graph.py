import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# ===== 1. Distance Transform (Grassfire algorithm) =====

def compute_distance_transform(binary):
    """
    Fast grassfire transform using deque (BFS)
    Input: binary image (1 = object, 0 = background)
    Output: distance map (int), where each pixel is min dist to background
    """
    h, w = binary.shape
    dist = np.full((h, w), -1, dtype=int)
    frontier = deque()

    # Tất cả pixel nền = 0 → khởi tạo
    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0:
                dist[y, x] = 0
                frontier.append((y, x))

    # BFS lan truyền khoảng cách
    while frontier:
        y, x = frontier.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] == -1:
                dist[ny, nx] = dist[y, x] + 1
                frontier.append((ny, nx))

    return dist


# ===== 2. Medial Axis Detection =====
def detect_medial_axis(distance_map):
    h, w = distance_map.shape
    medial = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            center = distance_map[y, x]
            patch = distance_map[y-1:y+2, x-1:x+2]
            if np.all(center >= patch):
                medial.append((x, y))
    return medial

# ===== 3. Extract boundary points =====
def extract_boundary_points(boundary_image):
    points = np.argwhere(boundary_image == 1)
    return [(x, y) for y, x in points]

# ===== 4. Extract shock graph =====
def extract_shock_graph(binary_image, boundary_image):
    boundary_points = extract_boundary_points(boundary_image)
    distance_map = compute_distance_transform(binary_image)
    medial_points = detect_medial_axis(distance_map)
    return boundary_points, medial_points

# ===== 5. Plot Shock Graph =====
def plot_shock_graph(binary_image, boundary_points, medial_points):
    # Convert binary image to RGB
    rgb_image = np.stack([binary_image]*3, axis=-1).astype(float)

    # Invert to make background white and object black (optional but easier to view)
    rgb_image = 1 - rgb_image

    # Plot using RGB image as background
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image, vmin=0, vmax=1)

    if boundary_points:
        bp = np.array(boundary_points)
        plt.plot(bp[:, 0], bp[:, 1], 'r.', markersize=1.5, label="Boundary")

    if medial_points:
        mp = np.array(medial_points)
        plt.plot(mp[:, 0], mp[:, 1], 'bo', markersize=1.5, label="Medial Axis")

    plt.title("Shock Graph (approximate)")
    plt.axis('off')
    plt.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    plt.show()

