import numpy as np
import matplotlib.pyplot as plt


def find_start_point(boundary):
    h, w = boundary.shape
    for i in range(h):
        for j in range(w):
            if boundary[i, j] == 1:
                return i, j
    return None


def generate_chain_code(boundary, start_point):
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]
    code = []
    visited = set()
    current = start_point
    h, w = boundary.shape

    while True:
        visited.add(current)
        found = False
        for k, (dy, dx) in enumerate(directions):
            ni, nj = current[0] + dy, current[1] + dx
            if 0 <= ni < h and 0 <= nj < w and boundary[ni, nj] == 1 and (ni, nj) not in visited:
                code.append(k)
                current = (ni, nj)
                found = True
                break
        if not found:
            break
    return code


def compute_cch(chain_code):
    histogram = [0] * 8
    for direction in chain_code:
        histogram[direction] += 1
    return histogram


def plot_cch_and_image(image, cch):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Input Image")

    plt.subplot(1, 2, 2)
    plt.bar(range(8), cch)
    plt.xlabel("Direction (0-7)")
    plt.ylabel("Frequency")
    plt.title("Chain Code Histogram")

    plt.tight_layout()
    plt.show()