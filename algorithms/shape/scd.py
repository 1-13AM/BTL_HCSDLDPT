import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===== 2. Làm mượt đường biên =====
import numpy as np

def gaussian_kernel1d(sigma, radius=None):
    """Tạo kernel Gaussian 1D thủ công"""
    if radius is None:
        radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def convolve_1d_circular(data, kernel):
    """Tích chập 1D theo chu kỳ (wrap-around)"""
    radius = len(kernel) // 2
    n = len(data)
    smoothed = np.zeros_like(data, dtype=float)
    for i in range(n):
        for k in range(-radius, radius + 1):
            idx = (i + k) % n  # wrap-around index
            smoothed[i] += data[idx] * kernel[k + radius]
    return smoothed

def smooth_contour(contour, sigma=3):
    """Làm mượt contour (chuỗi điểm x,y) bằng Gaussian smoothing thủ công"""
    x = contour[:, 0]
    y = contour[:, 1]
    kernel = gaussian_kernel1d(sigma)
    x_smooth = convolve_1d_circular(x, kernel)
    y_smooth = convolve_1d_circular(y, kernel)
    return np.stack((x_smooth, y_smooth), axis=1)


# ===== 3. Tính độ cong (curvature) =====
def compute_curvature(contour):
    x = contour[:, 0]
    y = contour[:, 1]

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8)**1.5
    return curvature

# ===== 4. Tìm các điểm phân đoạn theo zero-crossings =====
def find_zero_crossings(curvature):
    signs = np.sign(curvature)
    zero_crossings = np.where(np.diff(signs))[0]
    return zero_crossings

# ===== 5. Tách token và trích đặc trưng =====
def extract_tokens(contour, curvature, zc_indices):
    tokens = []
    for i in range(len(zc_indices) - 1):
        start = zc_indices[i]
        end = zc_indices[i + 1]
        segment = contour[start:end+1]
        segment_curv = curvature[start:end+1]

        if len(segment) < 3:
            continue

        max_idx = np.argmax(np.abs(segment_curv))
        max_curv = segment_curv[max_idx]

        dx = segment[-1, 0] - segment[0, 0]
        dy = segment[-1, 1] - segment[0, 1]
        orientation = np.arctan2(dy, dx)

        tokens.append({
            'segment': segment,
            'max_curvature': max_curv,
            'orientation': orientation
        })

    return tokens

# ===== 6. Hàm chạy tổng thể SCD và hiển thị =====
def smooth_curve_decomposition(contour, sigma=3):
    contour_smooth = smooth_contour(contour, sigma)
    curvature = compute_curvature(contour_smooth)
    zc = find_zero_crossings(curvature)
    tokens = extract_tokens(contour_smooth, curvature, zc)

    return tokens

def plot_scd_stages(contour, contour_smooth, curvature, zc_indices, tokens):
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Gốc ban đầu
    axs[0, 0].plot(contour[:, 0], contour[:, 1], c='black')
    axs[0, 0].set_title('Original Contour')
    axs[0, 0].invert_yaxis()
    axs[0, 0].axis('equal')

    # 2. Sau khi làm mượt
    axs[0, 1].plot(contour_smooth[:, 0], contour_smooth[:, 1], c='blue')
    axs[0, 1].set_title('Smoothed Contour')
    axs[0, 1].invert_yaxis()
    axs[0, 1].axis('equal')

    # 3. Độ cong
    axs[0, 2].plot(curvature, c='green')
    axs[0, 2].set_title('Curvature')
    axs[0, 2].set_xlabel('Point Index')
    axs[0, 2].set_ylabel('Curvature')
    axs[0, 2].grid(True)

    # 4. Zero-crossings
    axs[1, 0].plot(curvature, c='gray')
    axs[1, 0].scatter(zc_indices, curvature[zc_indices], color='red', label='Zero Crossings')
    axs[1, 0].set_title('Zero-Crossings on Curvature')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 5. Tokens trên contour
    axs[1, 1].plot(contour_smooth[:, 0], contour_smooth[:, 1], c='lightgray')
    for token in tokens:
        seg = token['segment']
        axs[1, 1].plot(seg[:, 0], seg[:, 1])
    axs[1, 1].set_title('Segmented Tokens')
    axs[1, 1].invert_yaxis()
    axs[1, 1].axis('equal')

    # 6. Token đặc trưng
    axs[1, 2].plot(contour_smooth[:, 0], contour_smooth[:, 1], c='lightgray')
    for token in tokens:
        seg = token['segment']
        axs[1, 2].plot(seg[:, 0], seg[:, 1])
        mid = seg[len(seg)//2]
        axs[1, 2].text(mid[0], mid[1], f"{token['max_curvature']:.2f}", fontsize=8)
    axs[1, 2].set_title('Tokens with Max Curvature')
    axs[1, 2].invert_yaxis()
    axs[1, 2].axis('equal')

    plt.tight_layout()
    plt.show()
