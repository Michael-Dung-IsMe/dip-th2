"""
Module: region_segmentation.py
Mục đích: Chứa các thuật toán phân đoạn dựa trên vùng và đồ thị cho Bài 4.

Các hàm chính:
- my_region_growing(img, seeds, threshold, mode, connectivity): Tự lập trình Region Growing bằng BFS.
- run_kmeans_segmentation(img, k_clusters, color_space): Phân cụm điểm ảnh K-Means trên RGB / CIELAB.
- run_slic_superpixels(img, n_segments, compactness): Phân đoạn siêu điểm ảnh SLIC.
- quadtree_split_and_merge(img, std_thresh, min_size): Thuật toán tách & hợp vùng theo cây tứ phân.
- run_felzenszwalb(img, scale, sigma, min_size): Thuật toán phân đoạn đồ thị Graph-based Felzenszwalb.
"""

from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import cv2
import numpy as np
from skimage.segmentation import slic, felzenszwalb


def my_region_growing(
    img: np.ndarray,
    seeds: List[Tuple[int, int]],
    threshold: float = 25.0,
    mode: str = 'seed_diff',
    connectivity: int = 8
) -> np.ndarray:
    """
    Tự lập trình thuật toán Region Growing (BFS) bắt đầu từ tập hạt giống seeds.
    
    Args:
        img: Ảnh xám 2D
        seeds: Danh sách tọa độ hạt giống [(y0, x0), (y1, x1), ...]
        threshold: Ngưỡng chênh lệch cường độ chấp nhận (T)
        mode: 'seed_diff' (so với hạt giống) hoặc 'mean_diff' (so với trung bình vùng)
        connectivity: 4 hoặc 8 láng giềng
        
    Returns:
        Mặt nạ nhị phân uint8 {0, 255} biểu diễn vùng đã phát triển.
    """
    if img.ndim != 2 or not seeds or threshold < 0 or connectivity not in (4, 8) or mode not in ("seed_diff", "mean_diff"):
        raise ValueError("Expected a grayscale image, seeds, nonnegative threshold, valid mode and connectivity.")

    h, w = img.shape
    accepted = np.zeros((h, w), dtype=bool)
    queue = deque()

    for y, x in seeds:
        if not (0 <= y < h and 0 <= x < w):
            raise ValueError(f"Seed ({y}, {x}) outside image of shape {img.shape}.")
        if not accepted[y, x]:
            accepted[y, x] = True
            queue.append((y, x, float(img[y, x])))

    total = float(np.sum(img[accepted], dtype=np.float64))
    count = int(accepted.sum())

    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        y, x, reference = queue.popleft()
        for dy, dx in offsets:
            yy, xx = y + dy, x + dx
            if not (0 <= yy < h and 0 <= xx < w) or accepted[yy, xx]:
                continue
            value = float(img[yy, xx])
            target = reference if mode == "seed_diff" else (total / count)
            if abs(value - target) <= threshold:
                accepted[yy, xx] = True
                total += value
                count += 1
                queue.append((yy, xx, reference))

    return accepted.astype(np.uint8) * 255


def run_kmeans_segmentation(
    img: np.ndarray,
    k: int = 3,
    color_space: str = 'RGB'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phân đoạn ảnh bằng thuật toán phân cụm K-Means.
    
    Args:
        img: Ảnh xám 2D hoặc ảnh màu RGB 3D
        k: Số cụm K
        color_space: 'RGB' hoặc 'LAB' (chỉ áp dụng cho ảnh 3 kênh)
        
    Returns:
        reconstructed: Ảnh tái tạo theo màu trung tâm cụm uint8
        labels: Ma trận nhãn cụm (0 đến K-1) cùng kích thước không gian ảnh
    """
    if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] != 3):
        raise ValueError("Expected grayscale (H, W) or RGB (H, W, 3) image.")
    if color_space not in ("RGB", "LAB") or (img.ndim == 2 and color_space == "LAB"):
        raise ValueError("LAB is supported for RGB input only.")

    source = cv2.cvtColor(img, cv2.COLOR_RGB2LAB) if color_space == "LAB" else img
    features = source.reshape((-1, 1 if source.ndim == 2 else 3)).astype(np.float32)

    cv2.setRNGSeed(42)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(features, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    reconstructed = centers[labels.ravel()].reshape(source.shape).clip(0, 255).astype(np.uint8)
    if color_space == "LAB":
        reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_LAB2RGB)

    return reconstructed, labels.reshape(img.shape[:2])


def run_slic_superpixels(
    img: np.ndarray,
    n_segments: int = 100,
    compactness: float = 10.0
) -> np.ndarray:
    """
    Phân đoạn ảnh thành các siêu điểm ảnh bằng SLIC Superpixels.
    
    Args:
        img: Ảnh xám hoặc RGB
        n_segments: Số lượng siêu điểm ảnh mục tiêu
        compactness: Độ đặc không gian m
        
    Returns:
        labels: Ma trận nhãn int gán cho từng pixel (bắt đầu từ 1)
    """
    return slic(
        img,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1,
        channel_axis=None if img.ndim == 2 else -1
    )


def run_felzenszwalb(
    img: np.ndarray,
    scale: float = 200.0,
    sigma: float = 0.8,
    min_size: int = 100
) -> np.ndarray:
    """
    Phân đoạn ảnh dựa trên đồ thị bằng thuật toán Felzenszwalb & Huttenlocher.
    
    Args:
        img: Ảnh xám hoặc RGB
        scale: Tham số tỷ lệ ngưỡng gộp đồ thị
        sigma: Độ lệch chuẩn Gaussian làm trơn sơ bộ
        min_size: Kích thước tối thiểu của một thành phần
        
    Returns:
        labels: Ma trận nhãn phân đoạn
    """
    channel_axis = -1 if img.ndim == 3 else None
    return felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size, channel_axis=channel_axis)


def quadtree_split_and_merge(
    img: np.ndarray,
    std_thresh: float = 15.0,
    min_size: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thuật toán phân đoạn Tách và Hợp vùng (Quadtree Region Splitting and Merging).
    
    1. Bước Split (Cây tứ phân đệ quy):
       Nếu độ lệch chuẩn std của vùng > std_thresh và kích thước > min_size:
       chia thành 4 vùng con bằng nhau.
    2. Bước Merge / Gán giá trị:
       Mỗi khối lá được gán bằng giá trị trung bình mức xám của vùng đó.
       
    Returns:
        reconstructed: Ảnh tái tạo theo các khối tứ phân đã làm trơn
        labels: Ma trận nhãn của các khối tứ phân
    """
    if img.ndim != 2:
        raise ValueError("Quadtree split-and-merge expects a 2D grayscale image.")

    h, w = img.shape
    reconstructed = np.zeros((h, w), dtype=np.uint8)
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0

    def _split(r: int, c: int, block_h: int, block_w: int):
        nonlocal current_label
        region = img[r:r + block_h, c:c + block_w]
        std = np.std(region)

        if std > std_thresh and block_h > min_size and block_w > min_size:
            half_h = block_h // 2
            half_w = block_w // 2
            rem_h = block_h - half_h
            rem_w = block_w - half_w

            _split(r, c, half_h, half_w)
            _split(r, c + half_w, half_h, rem_w)
            _split(r + half_h, c, rem_h, half_w)
            _split(r + half_h, c + half_w, rem_h, rem_w)
        else:
            current_label += 1
            mean_val = int(np.round(np.mean(region)))
            reconstructed[r:r + block_h, c:c + block_w] = mean_val
            labels[r:r + block_h, c:c + block_w] = current_label

    _split(0, 0, h, w)
    return reconstructed, labels
