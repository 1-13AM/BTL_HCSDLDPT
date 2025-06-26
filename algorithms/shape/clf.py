import numpy as np
import matplotlib.pyplot as plt

def compute_clf(contour, arc_length_steps):
    """
    Tính Chord Length Function (CLF) cho nhiều giá trị độ dài cung.

    Parameters:
    - contour: np.ndarray, (N, 2) danh sách điểm (x, y) tạo thành contour khép kín.
    - arc_length_steps: list[int], các giá trị độ dài cung (l) cần tính.

    Returns:
    - dict: {l: [d_0, d_1, ..., d_{N-1}]} với d_i là độ dài dây cung tại vị trí i.
    """
    N = len(contour)
    clf_results = {}

    for l in arc_length_steps:
        if l <= 0 or l >= N:
            continue
        chord_lengths = []
        for i in range(N):
            j = (i + l) % N  # điểm kết thúc cung, đảm bảo vòng tròn
            x1, y1 = contour[i]
            x2, y2 = contour[j]
            dx = x2 - x1
            dy = y2 - y1
            dist = (dx * dx + dy * dy) ** 0.5
            chord_lengths.append(dist)
        clf_results[l] = chord_lengths

    return clf_results

def plot_clf(clf_results):
    """
    Vẽ đồ thị Chord Length Function cho các độ dài cung.
    
    Parameters:
    - clf_results: dict từ hàm compute_clf
    """
    plt.figure(figsize=(10, 5))
    for l, lengths in clf_results.items():
        plt.plot(lengths, label=f'l = {l}')
    plt.title("Chord Length Function (CLF)")
    plt.xlabel("Chỉ số cung i")
    plt.ylabel("Độ dài dây cung d(i)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

