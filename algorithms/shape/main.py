from boundary_extraction import extract_boundary, plot_boundary_stages
from CCH import find_start_point, generate_chain_code, compute_cch, plot_cch_and_image
from scd import (
    smooth_contour, compute_curvature, find_zero_crossings,
    extract_tokens, plot_scd_stages
)

from bas import (
    beam_angle_statistics_full, compress_BAS_with_FD, compress_BAS_by_sampling, plot_BAS_stages
)

from clf import compute_clf, plot_clf


import numpy as np
import os
import json
from tqdm import tqdm

# 1. Chuỗi mã & CCH
def compute_and_plot_cch(image, boundary):
    start_point = find_start_point(boundary)
    if start_point is None:
        print("No boundary found.")
        return

    chain_code = generate_chain_code(boundary, start_point)
    cch = compute_cch(chain_code)
    return chain_code, cch

# 3. scd
def compute_and_plot_scd(boundary):
    # Chuyển ảnh boundary sang danh sách điểm (x, y)
    coords = np.argwhere(boundary == 1)
    if len(coords) < 5:
        print("Boundary quá ngắn để thực hiện SCD.")
        return
    contour = coords[:, [1, 0]].astype(np.float32)  # (x, y)

    # Thực hiện SCD
    sigma = 3
    contour_smooth = smooth_contour(contour, sigma)
    curvature = compute_curvature(contour_smooth)
    zc = find_zero_crossings(curvature)
    tokens = extract_tokens(contour_smooth, curvature, zc)

    return tokens, contour_smooth, curvature, zc

# 4. BAS
def compute_and_plot_bas(contour, m_max=2, K_step=5, bins=30, T_fd=10, sampling_step=5):
    """
    Thực hiện BAS (Beam Angle Statistics) từ contour và hiển thị trực quan các bước.
    Bao gồm:
    - Tính toán CK_matrix (beam angles)
    - Tính moments theo histogram
    - Nén đặc trưng bằng Fourier Descriptor và Sampling
    - Hiển thị trực quan toàn bộ pipeline

    Parameters:
    - contour: ndarray (N x 2), danh sách điểm (x, y)
    """

    moments, CK_matrix, K_list = beam_angle_statistics_full(
        contour, m_max=m_max, K_step=K_step, bins=bins
    )
    feature_fd = compress_BAS_with_FD(moments, T=T_fd)

    feature_sampled = compress_BAS_by_sampling(moments, T=sampling_step)

    return {
        "moments": moments,
        "feature_fd": feature_fd,
        "feature_sampled": feature_sampled
    }


# 5. Chord Length Function (CLF)
def compute_and_plot_clf(boundary, arc_length_steps=[5, 10, 20, 30]):
    coords = np.argwhere(boundary == 1)
    if len(coords) < 5:
        print("Boundary quá ngắn để thực hiện CLF.")
        return

    # Chuyển sang (x, y)
    contour = coords[:, [1, 0]].astype(np.float32)
    clf_results = compute_clf(contour, arc_length_steps)
    return clf_results

# 6. Shock Graph




def extract_features_from_image(img_path):
    """Trích xuất toàn bộ 5 đặc trưng hình dạng từ 1 ảnh."""

    # 1. Preprocess & extract boundary
    gray, binary, eroded, boundary = extract_boundary(img_path)

    # ====== CCH ======
    chain_code, cch_feature = compute_and_plot_cch(gray, boundary)
    
    # ====== SCD ======
    tokens, contour_smooth, curvature, zc = compute_and_plot_scd(boundary)

    # ====== BAS ======
    bas_feature = compute_and_plot_bas(contour_smooth)

    # ====== CLF ======
    clf_feature = compute_and_plot_clf(boundary)
    # ====== Shock Graph ======
    

    # Tổng hợp đặc trưng
    return {
        "label": img_path,
        "cch": cch_feature,                      # Histogram 8 hướng
        "scd_tokens": tokens,                    # Danh sách đoạn cong với max curvature & orientation
        "bas_vector_fd": bas_feature["feature_fd"],         # Vector đặc trưng Fourier
        "bas_vector_sample": bas_feature["feature_sampled"],# Vector từ sampling
        "clf": clf_feature                       # Dict {l: [d_i]} — CLF theo nhiều scale
        
        # "shock_graph": ...                    # Có thể thêm sau
    }

def numpy_to_list(obj):
    """Chuyển mọi kiểu NumPy thành kiểu Python gốc để lưu JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, list):
        return [numpy_to_list(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    else:
        return obj


def main(dataset_dir='../../animal_datasets_no_bg/', output_path='features.json'):
    features = []
    image_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # đảm bảo thứ tự cố định

    for fname in tqdm(image_files, desc="Extracting features"):
        img_path = os.path.join(dataset_dir, fname)
        try:
            feat = extract_features_from_image(img_path)
            features.append(feat)
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

    # Lưu kết quả ra file JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(numpy_to_list(features), f, indent=2, ensure_ascii=False)


    print(f"✅ Trích xuất và lưu xong {len(features)} ảnh vào {output_path}")

if __name__ == "__main__":
    main() 