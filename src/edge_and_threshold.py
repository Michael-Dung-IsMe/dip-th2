"""
Module: edge_and_threshold.py
Mục đích: Chứa các thuật toán phát hiện biên và phân ngưỡng cho Bài 3.

Các hàm chính:
- compute_sobel(img, ksize, threshold): Tính đạo hàm bậc 1, biên độ và hướng gradient.
- compute_laplacian(img, ksize, sigma): Tính đạo hàm bậc 2 và Laplacian of Gaussian (LoG).
- zero_crossing(response, threshold): Phát hiện điểm đổi dấu zero-crossing của phản hồi Laplacian/LoG.
- run_canny(img, low_thresh, high_thresh, sigma): Phát hiện biên Canny kèm làm trơn Gaussian.
- my_otsu_threshold(img): Tự lập trình thuật toán Otsu bằng NumPy (cực đại hóa phương sai liên lớp).
- run_adaptive_threshold(img, block_size, c_val, method): Phân ngưỡng thích nghi cục bộ.
- compute_iou(reference, predicted): Tính chỉ số IoU giữa mặt nạ tham chiếu và dự đoán.
"""

from typing import Tuple, Dict, Optional
import cv2
import numpy as np


def compute_sobel(
    img: np.ndarray,
    ksize: int = 3,
    threshold: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Tính gradient Sobel theo phương ngang, dọc và biên độ gradient.
    
    Args:
        img: Ảnh xám 2D uint8
        ksize: Kích thước kernel Sobel (3, 5, 7)
        threshold: Ngưỡng nhị phân hóa biên độ (nếu có)
        
    Returns:
        dict chứa 'gx', 'gy', 'magnitude', và 'edges' (nếu threshold được cung cấp)
    """
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.hypot(gx, gy)
    result = {"gx": gx, "gy": gy, "magnitude": magnitude}
    if threshold is not None:
        result["edges"] = (magnitude > threshold).astype(np.uint8) * 255
    return result


def compute_laplacian(img: np.ndarray, ksize: int = 3, sigma: float = 0.0) -> np.ndarray:
    """
    Tính toán tử Laplacian trực tiếp hoặc LoG (Laplacian of Gaussian).
    Nếu sigma > 0, ảnh sẽ được làm trơn Gaussian trước khi tính Laplacian.
    """
    source = cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img
    return cv2.Laplacian(source, cv2.CV_64F, ksize=ksize)


def zero_crossing(response: np.ndarray, threshold: float = 10.0) -> np.ndarray:
    """
    Phát hiện điểm đổi dấu (zero-crossing) của toán tử bậc 2 (Laplacian/LoG)
    bằng cách so sánh giá trị cực đại và cực tiểu trong lân cận 3x3:
    Điều kiện: low < 0 and high > 0 and (high - low) > threshold.
    """
    high = cv2.dilate(response, np.ones((3, 3), np.uint8))
    low = cv2.erode(response, np.ones((3, 3), np.uint8))
    return ((low < 0) & (high > 0) & ((high - low) > threshold)).astype(np.uint8) * 255


def run_canny(
    img: np.ndarray,
    low_thresh: float,
    high_thresh: float,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Phát hiện biên bằng thuật toán Canny, có làm trơn Gaussian tường minh theo sigma.
    """
    blur = cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img
    return cv2.Canny(blur, int(low_thresh), int(high_thresh))


def my_otsu_threshold(img: np.ndarray) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """
    Tự lập trình thuật toán phân ngưỡng Otsu bằng NumPy.
    Tối ưu hóa phương sai giữa hai lớp (between-class variance):
    sigma_B^2(k) = [m_G * P1(k) - m(k)]^2 / [P1(k) * (1 - P1(k))]
    
    Returns:
        optimal_threshold: Ngưỡng tối ưu T* (int)
        max_variance: Giá trị phương sai liên lớp cực đại
        variance_curve: Mảng phương sai liên lớp với mọi k in [0, 255]
        binary_img: Ảnh nhị phân kết quả uint8 {0, 255}
    """
    if img.dtype != np.uint8 or img.ndim != 2 or img.size == 0:
        raise ValueError("Otsu expects a nonempty 2D uint8 image.")

    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    probabilities = hist / img.size
    weights = np.cumsum(probabilities)
    means = np.cumsum(probabilities * np.arange(256))
    
    numerator = (means[-1] * weights - means) ** 2
    denominator = weights * (1.0 - weights)
    
    curve = np.zeros(256, dtype=np.float64)
    np.divide(numerator, denominator, out=curve, where=denominator > 1e-15)
    
    threshold = int(np.argmax(curve))
    binary_img = (img > threshold).astype(np.uint8) * 255
    return threshold, float(curve[threshold]), curve, binary_img


def run_adaptive_threshold(
    img: np.ndarray,
    block_size: int = 31,
    c_val: float = 5.0,
    method: str = 'gaussian'
) -> np.ndarray:
    """
    Phân ngưỡng thích nghi thích ứng với chiếu sáng không đồng đều.
    method: 'mean' hoặc 'gaussian'.
    block_size: Kích thước lân cận (lẻ, >= 3).
    c_val: Hằng số trừ khỏi giá trị trung bình/Gaussian.
    """
    methods = {"mean": cv2.ADAPTIVE_THRESH_MEAN_C, "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C}
    if method not in methods or block_size < 3 or block_size % 2 == 0:
        raise ValueError("Use mean or gaussian and an odd block_size >= 3.")
    return cv2.adaptiveThreshold(img, 255, methods[method], cv2.THRESH_BINARY, block_size, c_val)


def compute_iou(reference: np.ndarray, predicted: np.ndarray) -> float:
    """Tính chỉ số IoU (Intersection over Union) giữa 2 mặt nạ nhị phân."""
    ref_bool = reference != 0
    pred_bool = predicted != 0
    union = np.count_nonzero(ref_bool | pred_bool)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(ref_bool & pred_bool) / union)
