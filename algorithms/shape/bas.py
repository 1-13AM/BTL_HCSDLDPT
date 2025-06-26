import numpy as np
import matplotlib.pyplot as plt

# ===== 1. Tính góc beam cho mọi K =====
def compute_beam_angles_allK(contour, K_list):
    N = len(contour)
    beam_angles_all = []
    for i in range(N):
        CK_values = []
        for K in K_list:
            i_fwd = (i + K) % N
            i_bwd = (i - K + N) % N
            v_fwd = contour[i_fwd] - contour[i]
            v_bwd = contour[i_bwd] - contour[i]
            theta_fwd = np.arctan2(v_fwd[1], v_fwd[0])
            theta_bwd = np.arctan2(v_bwd[1], v_bwd[0])
            CK = (theta_bwd - theta_fwd) % (2 * np.pi)
            CK_values.append(CK)
        beam_angles_all.append(CK_values)
    return np.array(beam_angles_all)

# ===== 2. Tính moment từ góc beam =====
def compute_moments_from_angles(CK_matrix, m_max=2, bins=30):
    N, _ = CK_matrix.shape
    moments = np.zeros((N, m_max))
    for i in range(N):
        hist, bins_edges = np.histogram(CK_matrix[i], bins=bins, range=(0, 2*np.pi), density=True)
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        for m in range(1, m_max + 1):
            moments[i, m-1] = np.sum((bin_centers**m) * hist) * (2*np.pi / bins)
    return moments

# ===== 3. Vẽ các moment =====
def plot_BAS_moments(moments):
    for m in range(moments.shape[1]):
        plt.plot(moments[:, m], label=f'Moment {m+1}')
    plt.title("BAS Moments along Contour")
    plt.xlabel("Contour Point Index")
    plt.ylabel("Moment Value")
    plt.grid(True)
    plt.legend()
    plt.show()

# ===== 4. Hàm tổng thể =====
def beam_angle_statistics_full(contour, m_max=2, K_step=5, bins=30):
    N = len(contour)
    K_list = list(range(1, N//2, K_step))
    CK_matrix = compute_beam_angles_allK(contour, K_list)
    moments = compute_moments_from_angles(CK_matrix, m_max, bins)
    # plot_BAS_moments(moments)
    return moments, CK_matrix, K_list

# ===== 5. Nén bằng Fourier Descriptor =====
def compress_BAS_with_FD(moments, T=10):
    features = []
    for m in range(moments.shape[1]):
        gamma = moments[:, m]
        N = len(gamma)
        fourier = np.fft.fft(gamma)
        mag = np.abs(fourier)
        mag /= (mag[0] + 1e-8)
        features.extend(mag[1:T+1])
    return np.array(features)

# ===== 6. Nén bằng Sampling từ phổ Fourier =====
def compress_BAS_by_sampling(moments, T=10):
    reconstructed = []
    for m in range(moments.shape[1]):
        gamma = moments[:, m]
        N = len(gamma)
        fft_coeffs = np.fft.fft(gamma)
        low_pass = np.zeros_like(fft_coeffs)
        low_pass[:T] = fft_coeffs[:T]
        recon = np.fft.ifft(low_pass).real
        reconstructed.append(recon[::N//T])
    return np.concatenate(reconstructed)

# ===== 7. Vẽ các bước của BAS =====
def plot_BAS_stages(contour, CK_matrix, moments, K_list):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Contour
    axs[0, 0].plot(contour[:, 0], contour[:, 1], 'k-')
    axs[0, 0].set_title("Contour")
    axs[0, 0].invert_yaxis()
    axs[0, 0].axis('equal')

    # 2. CK matrix (heatmap)
    im = axs[0, 1].imshow(CK_matrix.T, aspect='auto', cmap='jet', extent=[0, len(contour), K_list[-1], K_list[0]])
    axs[0, 1].set_title("Beam Angle Matrix")
    axs[0, 1].set_ylabel("K")
    axs[0, 1].set_xlabel("Point Index")
    fig.colorbar(im, ax=axs[0, 1])

    # 3. Moment plots
    for m in range(moments.shape[1]):
        axs[1, 0].plot(moments[:, m], label=f'Moment {m+1}')
    axs[1, 0].set_title("BAS Moments")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. Moment scatter on contour
    axs[1, 1].scatter(contour[:, 0], contour[:, 1], c=moments[:, 0], cmap='viridis')
    axs[1, 1].set_title("Contour colored by 1st Moment")
    axs[1, 1].invert_yaxis()
    axs[1, 1].axis('equal')

    plt.tight_layout()
    plt.show()
