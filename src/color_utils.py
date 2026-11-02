"""
Module: color_utils.py
Mục đích: Chứa các thuật toán xử lý không gian màu và phân đoạn màu cho Bài 1.

Các hàm chính:
- rgb_to_hsi(rgb_img): Chuyển đổi thủ công từ RGB sang HSI bằng NumPy theo công thức Gonzalez & Woods.
- convert_color_spaces(rgb_img): Chuyển đổi và trả về các kênh của RGB, HSV, HSI, CIELAB.
- segment_color_rgb(rgb_img, lower_bound, upper_bound): Phân đoạn theo ngưỡng hộp trong không gian RGB.
- segment_color_hsv(hsv_img, h_ranges, s_min, v_min, wrap_hue): Phân đoạn góc màu trong không gian HSV.
- evaluate_segmentation_masks(mask_pred, mask_gt): Đánh giá định lượng kết quả phân đoạn (IoU, Dice, Precision, Recall).
"""

from typing import Tuple, Dict, Any, Union
import cv2
import numpy as np


def rgb_to_hsi(rgb_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chuyển đổi ảnh RGB [0, 255] sang 3 kênh H [0, 360], S [0, 1], I [0, 1] thuần NumPy
    theo công thức chuẩn Gonzalez & Woods (Digital Image Processing).
    """
    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError("Expected a 3-channel RGB image (H, W, 3).")

    rgb = rgb_img.astype(np.float64) / 255.0
    r, g, b = np.moveaxis(rgb, -1, 0)
    total = r + g + b
    intensity = total / 3.0

    saturation = np.zeros_like(total)
    np.divide(3.0 * np.minimum(np.minimum(r, g), b), total, out=saturation, where=total > 0)
    saturation = np.where(total > 0, 1.0 - saturation, 0.0)

    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    cosine = np.ones_like(denominator)
    np.divide(0.5 * ((r - g) + (r - b)), denominator, out=cosine, where=denominator > 1e-12)
    theta = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    hue = np.where(b <= g, theta, 360.0 - theta) % 360.0
    hue = np.where(denominator > 1e-12, hue, 0.0)
    return hue, saturation, intensity


def convert_color_spaces(rgb_img: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Chuyển đổi ảnh RGB sang các không gian màu HSV, HSI, CIELAB và trả về từng kênh riêng biệt.
    
    Returns:
        dict chứa:
        - 'RGB': {'R': ..., 'G': ..., 'B': ...}
        - 'HSV': {'H': ..., 'S': ..., 'V': ...}
        - 'HSI': {'H': ..., 'S': ..., 'I': ...}
        - 'LAB': {'L': ..., 'a': ..., 'b': ...}
    """
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    h_hsi, s_hsi, i_hsi = rgb_to_hsi(rgb_img)

    return {
        "RGB": {
            "R": rgb_img[:, :, 0],
            "G": rgb_img[:, :, 1],
            "B": rgb_img[:, :, 2],
        },
        "HSV": {
            "H": hsv[:, :, 0],
            "S": hsv[:, :, 1],
            "V": hsv[:, :, 2],
        },
        "HSI": {
            "H": h_hsi,
            "S": s_hsi,
            "I": i_hsi,
        },
        "LAB": {
            "L": lab[:, :, 0],
            "a": lab[:, :, 1],
            "b": lab[:, :, 2],
        },
    }


def segment_color_rgb(
    rgb_img: np.ndarray,
    lower_bound: Tuple[int, int, int],
    upper_bound: Tuple[int, int, int]
) -> np.ndarray:
    """
    Phân đoạn đối tượng theo ngưỡng dải màu trong không gian RGB.
    Trả về mặt nạ nhị phân uint8 {0, 255}.
    """
    lower = np.array(lower_bound, dtype=np.uint8)
    upper = np.array(upper_bound, dtype=np.uint8)
    return cv2.inRange(rgb_img, lower, upper)


def segment_color_hsv(
    hsv_img: np.ndarray,
    h_range: Tuple[int, int],
    s_range: Tuple[int, int] = (30, 255),
    v_range: Tuple[int, int] = (30, 255),
    wrap_hue: bool = False
) -> np.ndarray:
    """
    Phân đoạn đối tượng theo dải Hue/Saturation/Value trong không gian HSV (OpenCV H: 0-179).
    Hỗ trợ dải Hue quấn quanh mốc 0/180 (cho màu Đỏ) khi wrap_hue=True.
    Trả về mặt nạ nhị phân uint8 {0, 255}.
    """
    low_h, high_h = h_range
    s_min, s_max = s_range
    v_min, v_max = v_range

    if wrap_hue:
        # Màu đỏ nằm ở 2 dải [low_h, 179] và [0, high_h]
        first = cv2.inRange(hsv_img, (low_h, s_min, v_min), (179, s_max, v_max))
        second = cv2.inRange(hsv_img, (0, s_min, v_min), (high_h, s_max, v_max))
        return cv2.bitwise_or(first, second)
    return cv2.inRange(hsv_img, (low_h, s_min, v_min), (high_h, s_max, v_max))


def evaluate_segmentation_masks(
    mask_pred: np.ndarray,
    mask_gt: np.ndarray
) -> Dict[str, float]:
    """
    Đánh giá định lượng kết quả phân đoạn giữa mặt nạ dự đoán và mặt nạ chuẩn:
    - IoU (Intersection over Union / Jaccard Index)
    - Dice / F1-Score
    - Precision
    - Recall
    - Foreground Ratio
    """
    pred_bool = mask_pred > 0
    gt_bool = mask_gt > 0

    intersection = np.count_nonzero(pred_bool & gt_bool)
    union = np.count_nonzero(pred_bool | gt_bool)
    pred_count = np.count_nonzero(pred_bool)
    gt_count = np.count_nonzero(gt_bool)

    iou = float(intersection / union) if union > 0 else 1.0
    dice = float(2.0 * intersection / (pred_count + gt_count)) if (pred_count + gt_count) > 0 else 1.0
    precision = float(intersection / pred_count) if pred_count > 0 else 1.0
    recall = float(intersection / gt_count) if gt_count > 0 else 1.0
    fg_ratio = float(pred_count / mask_pred.size)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "foreground_ratio": fg_ratio,
    }
