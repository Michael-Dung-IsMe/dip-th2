"""
Module: morphology.py
Mục đích: Chứa các thuật toán hình thái học tự lập trình bằng NumPy và các thuật toán nâng cao cho Bài 2.

Các hàm chính:
- create_structuring_element(shape, size): Tạo SE hình RECT, CROSS, ELLIPSE.
- my_erosion(img, se): Phép co nhị phân vector hóa bằng NumPy (sliding window/shifts).
- my_dilation(img, se): Phép giãn nhị phân vector hóa bằng NumPy.
- my_opening(img, se): Phép mở kết hợp co trước, giãn sau.
- my_closing(img, se): Phép đóng kết hợp giãn trước, co sau.
- boundary_extraction(img, se): Trích xuất đường biên trong beta(A) = A - (A erosion B).
- hole_filling(img, se, seeds): Làm đầy lỗ thủng bằng phép giãn có điều kiện (từ biên hoặc tập seed).
- conditional_dilation_hole_filling(img, se, max_iter, snapshot_iters): Lặp giãn có điều kiện từng bước kèm snapshot.
- connected_components_labeling(img, connectivity): Trích xuất và gán nhãn thành phần liên thông bằng BFS.
"""

from typing import Tuple, List, Optional, Dict, Any
from collections import deque
import cv2
import numpy as np


def create_structuring_element(shape: str, size: int) -> np.ndarray:
    """
    Tạo ma trận phần tử cấu trúc nhị phân (RECT, CROSS, ELLIPSE) với kích thước size x size (size lẻ dương).
    Trả về ma trận uint8 có giá trị {0, 1}.
    """
    shapes = {"RECT": cv2.MORPH_RECT, "CROSS": cv2.MORPH_CROSS, "ELLIPSE": cv2.MORPH_ELLIPSE}
    if shape.upper() not in shapes or size <= 0 or size % 2 == 0:
        raise ValueError("Use RECT, CROSS or ELLIPSE and a positive odd size.")
    return cv2.getStructuringElement(shapes[shape.upper()], (size, size))


def _morph(img: np.ndarray, se: np.ndarray, erosion: bool) -> np.ndarray:
    """
    Hàm lõi thực hiện biến đổi hình thái học thuần NumPy bằng cách dịch khung nhìn (spatial shifts).
    - erosion=True: Phép co (A (-) B) = AND của các khung nhìn dịch chuyển theo B.
    - erosion=False: Phép giãn (A (+) B) = OR của các khung nhìn dịch chuyển theo B lật đối xứng.
    """
    if img.ndim != 2 or se.ndim != 2 or any(n % 2 == 0 for n in se.shape) or not np.any(se):
        raise ValueError("Expected a 2D image and a nonempty odd-sized structuring element.")
    
    foreground = img != 0
    h, w = foreground.shape
    ph, pw = se.shape[0] // 2, se.shape[1] // 2
    padded = np.pad(foreground, ((ph, ph), (pw, pw)), constant_values=erosion)
    result = np.full((h, w), erosion, dtype=bool)
    kernel = se if erosion else se[::-1, ::-1]

    for y, x in np.argwhere(kernel):
        view = padded[y:y + h, x:x + w]
        if erosion:
            result &= view
        else:
            result |= view
    return result.astype(np.uint8) * 255


def my_erosion(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Phép co (Erosion) tự lập trình thuần NumPy, tối ưu vector hóa:
    A (-) B = { z | (B)_z subseteq A }.
    Trả về ảnh uint8 {0, 255}.
    """
    return _morph(img, se, True)


def my_dilation(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Phép giãn (Dilation) tự lập trình thuần NumPy, tối ưu vector hóa:
    A (+) B = { z | (B_hat)_z cap A != emptyset }.
    Trả về ảnh uint8 {0, 255}.
    """
    return _morph(img, se, False)


def my_opening(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Phép mở: A o B = (A (-) B) (+) B (Co trước, Giãn sau)."""
    return my_dilation(my_erosion(img, se), se)


def my_closing(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Phép đóng: A * B = (A (+) B) (-) B (Giãn trước, Co sau)."""
    return my_erosion(my_dilation(img, se), se)


def boundary_extraction(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Trích xuất đường viền biên trong:
    beta(A) = A - (A (-) B).
    """
    return ((img != 0) & (my_erosion(img, se) == 0)).astype(np.uint8) * 255


def hole_filling(img: np.ndarray, se: np.ndarray, seeds: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    """
    Làm đầy lỗ bằng phép lan truyền hàng đợi (BFS) từ biên ngoài trên ảnh bù A^c:
    Tất cả các pixel nền nối ra ngoài biên là background thực sự.
    Các pixel nền bị cô lập bên trong vật thể chính là lỗ thủng cần làm đầy.
    """
    if img.ndim != 2 or se.ndim != 2 or any(n % 2 == 0 for n in se.shape):
        raise ValueError("Expected a 2D image and an odd-sized structuring element.")

    foreground = img != 0
    h, w = foreground.shape
    offsets = [(int(y - se.shape[0] // 2), int(x - se.shape[1] // 2))
               for y, x in np.argwhere(se) if (y, x) != (se.shape[0] // 2, se.shape[1] // 2)]

    if seeds is None:
        starts = [(y, x) for y in range(h) for x in (0, w - 1)]
        starts += [(y, x) for x in range(w) for y in (0, h - 1)]
    else:
        starts = seeds

    reached = np.zeros((h, w), dtype=bool)
    queue = deque()
    for y, x in starts:
        if not (0 <= y < h and 0 <= x < w):
            raise ValueError("Seed outside image.")
        if not foreground[y, x] and not reached[y, x]:
            reached[y, x] = True
            queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        for dy, dx in offsets:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and not foreground[yy, xx] and not reached[yy, xx]:
                reached[yy, xx] = True
                queue.append((yy, xx))

    filled = foreground | (~foreground & ~reached if seeds is None else reached)
    return filled.astype(np.uint8) * 255


def conditional_dilation_hole_filling(
    img: np.ndarray,
    se: np.ndarray,
    max_iter: int = 2000,
    snapshot_iters: Tuple[int, ...] = (1, 5, 15)
) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
    """
    Thuật toán lặp giãn có điều kiện từng bước từ biên:
    X_k = (X_{k-1} (+) se) cap A^c
    Trả về (ảnh_đã_fill, danh_sách_snapshots_(iteration, snapshot_mask)).
    """
    foreground = img != 0
    background = ~foreground
    reached = np.zeros_like(foreground, dtype=np.uint8)
    reached[0, :] = background[0, :]
    reached[-1, :] = background[-1, :]
    reached[:, 0] = background[:, 0]
    reached[:, -1] = background[:, -1]

    snapshots = []
    converged = False

    for iteration in range(1, max_iter + 1):
        updated = cv2.dilate(reached, se, borderType=cv2.BORDER_CONSTANT, borderValue=0) & background
        if iteration in snapshot_iters:
            snapshots.append((iteration, (updated * 255).astype(np.uint8)))
        if np.array_equal(updated, reached):
            snapshots.append((iteration, (updated * 255).astype(np.uint8)))
            converged = True
            break
        reached = updated

    if not converged:
        raise RuntimeError(f"Conditional dilation did not converge after {max_iter} iterations.")

    filled = foreground | (~foreground & (reached == 0))
    return filled.astype(np.uint8) * 255, snapshots


def connected_components_labeling(
    img: np.ndarray,
    connectivity: int = 8
) -> Tuple[np.ndarray, int, List[Dict[str, Any]]]:
    """
    Gán nhãn thành phần liên thông bằng BFS thuần NumPy/Python (không phụ thuộc opencv).
    
    Args:
        img: Ảnh nhị phân 2D (foreground != 0)
        connectivity: 4 hoặc 8
        
    Returns:
        labeled_mask: Ma trận int32 cùng kích thước, mỗi thành phần có nhãn 1, 2, ..., N (nền là 0)
        num_components: Tổng số thành phần liên thông
        stats: Danh sách thông tin từng thành phần (label, area, centroid, bbox)
    """
    if img.ndim != 2:
        raise ValueError("Expected a 2D binary image.")
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8.")

    foreground = img != 0
    h, w = foreground.shape
    labeled_mask = np.zeros((h, w), dtype=np.int32)
    visited = np.zeros((h, w), dtype=bool)

    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    current_label = 0
    stats = []

    for r in range(h):
        for c in range(w):
            if foreground[r, c] and not visited[r, c]:
                current_label += 1
                queue = deque([(r, c)])
                visited[r, c] = True
                labeled_mask[r, c] = current_label

                pixels = [(r, c)]
                min_r, max_r = r, r
                min_c, max_c = c, c

                while queue:
                    curr_r, curr_c = queue.popleft()
                    for dr, dc in offsets:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if foreground[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                labeled_mask[nr, nc] = current_label
                                queue.append((nr, nc))
                                pixels.append((nr, nc))
                                if nr < min_r: min_r = nr
                                if nr > max_r: max_r = nr
                                if nc < min_c: min_c = nc
                                if nc > max_c: max_c = nc

                area = len(pixels)
                coords = np.array(pixels)
                centroid = (float(np.mean(coords[:, 0])), float(np.mean(coords[:, 1])))
                bbox = (int(min_r), int(min_c), int(max_r - min_r + 1), int(max_c - min_c + 1))

                stats.append({
                    "label": current_label,
                    "area": area,
                    "centroid": centroid,
                    "bbox": bbox
                })

    return labeled_mask, current_label, stats
