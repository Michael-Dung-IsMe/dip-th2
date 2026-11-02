"""
Module: visualization.py
Mục đích: Chuẩn hóa việc lưu và vẽ các cặp ảnh Đầu vào - Kết quả phục vụ báo cáo PDF.

Các hàm chính:
- save_figure_grid(panels, titles, folder, filename, cols, dpi, base_out): Vẽ và lưu lưới ảnh đối sánh (tương ứng hàm figure() trong notebook).
- save_image_pair(input_img, output_img, output_dir, prefix, ...): Lưu bộ 3 file ảnh (input, output, input_vs_output).
- save_single_image(img, folder, filename, cmap, dpi, base_out): Lưu ảnh đơn lẻ ra đĩa.
"""

from typing import Union, List, Optional, Tuple, Sequence
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_figure_grid(
    panels: Sequence[np.ndarray],
    titles: Sequence[str],
    folder: Union[str, Path],
    filename: str,
    cols: int = 4,
    dpi: int = 160,
    base_out: Union[str, Path] = "outputs",
    show: bool = False
) -> Path:
    """
    Vẽ và lưu lưới ảnh gồm nhiều panels và titles tương ứng (chuẩn hóa hàm figure() trong các notebook).
    
    Args:
        panels: Danh sách các ma trận ảnh (2D hoặc 3D)
        titles: Danh sách tiêu đề tương ứng cho từng panel
        folder: Tên thư mục con (ví dụ: 'bai_2/2.1_numpy_vs_opencv')
        filename: Tên tệp ảnh xuất ra (ví dụ: '2.1_wirebond_input_vs_output_4_ops.png')
        cols: Số cột hiển thị trong lưới
        dpi: Độ phân giải khi lưu ảnh (mặc định 160)
        base_out: Thư mục gốc lưu trữ (mặc định 'outputs')
        show: Có gọi plt.show() hay không (mặc định False để chạy ngầm)
        
    Returns:
        Đường dẫn Path tới tệp ảnh đã lưu.
    """
    out_dir = Path(base_out) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    n = len(panels)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.8 * rows), squeeze=False)

    for i, (ax, img, title) in enumerate(zip(axes.flat, panels, titles)):
        is_gray = (img.ndim == 2)
        vmin = 0 if img.dtype == np.uint8 else None
        vmax = 255 if img.dtype == np.uint8 else None
        ax.imshow(img, cmap="gray" if is_gray else None, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Tắt các trục thừa nếu số ảnh không lấp đầy hàng cuối
    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def save_image_pair(
    input_img: np.ndarray,
    output_img: np.ndarray,
    output_dir: Union[str, Path],
    prefix: str,
    title_in: str = "Ảnh đầu vào",
    title_out: str = "Kết quả",
    cmap_in: Optional[str] = None,
    cmap_out: Optional[str] = None,
    dpi: int = 200
) -> Tuple[Path, Path, Path]:
    """
    Lưu bộ 3 file ảnh chuẩn định dạng báo cáo:
    1. <prefix>_input.png: Ảnh đầu vào
    2. <prefix>_output.png: Ảnh kết quả
    3. <prefix>_input_vs_output.png: Ảnh ghép đối sánh trực tiếp [Đầu vào | Kết quả]
    
    Returns:
        Tuple chứa đường dẫn tới 3 file: (path_in, path_out, path_comp)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path_in = out_dir / f"{prefix}_input.png"
    path_out = out_dir / f"{prefix}_output.png"
    path_comp = out_dir / f"{prefix}_input_vs_output.png"

    # 1. Lưu ảnh input
    fig_in, ax_in = plt.subplots(figsize=(5, 5))
    ax_in.imshow(input_img, cmap=cmap_in or ("gray" if input_img.ndim == 2 else None))
    ax_in.set_title(title_in, fontsize=11)
    ax_in.axis("off")
    fig_in.tight_layout()
    fig_in.savefig(path_in, dpi=dpi, bbox_inches="tight")
    plt.close(fig_in)

    # 2. Lưu ảnh output
    fig_out, ax_out = plt.subplots(figsize=(5, 5))
    ax_out.imshow(output_img, cmap=cmap_out or ("gray" if output_img.ndim == 2 else None))
    ax_out.set_title(title_out, fontsize=11)
    ax_out.axis("off")
    fig_out.tight_layout()
    fig_out.savefig(path_out, dpi=dpi, bbox_inches="tight")
    plt.close(fig_out)

    # 3. Lưu ảnh ghép đối sánh [Input | Output]
    fig_comp, axes_comp = plt.subplots(1, 2, figsize=(10, 5))
    axes_comp[0].imshow(input_img, cmap=cmap_in or ("gray" if input_img.ndim == 2 else None))
    axes_comp[0].set_title(title_in, fontsize=11)
    axes_comp[0].axis("off")

    axes_comp[1].imshow(output_img, cmap=cmap_out or ("gray" if output_img.ndim == 2 else None))
    axes_comp[1].set_title(title_out, fontsize=11)
    axes_comp[1].axis("off")

    fig_comp.tight_layout()
    fig_comp.savefig(path_comp, dpi=dpi, bbox_inches="tight")
    plt.close(fig_comp)

    return path_in, path_out, path_comp


def save_single_image(
    img: np.ndarray,
    folder: Union[str, Path],
    filename: str,
    cmap: Optional[str] = None,
    dpi: int = 200,
    base_out: Union[str, Path] = "outputs"
) -> Path:
    """Lưu ảnh đơn lẻ ra thư mục đích."""
    out_dir = Path(base_out) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    if img.ndim == 2 and cmap is None:
        cmap = "gray"

    plt.imsave(out_path, img, cmap=cmap)
    return out_path
