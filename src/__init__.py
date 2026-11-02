"""
DIP Practical Lab 2 - Source Package
Các module thuật toán cốt lõi cho 4 bài thực hành DIP:
- color_utils: Xử lý và phân đoạn không gian màu (Bài 1)
- morphology: Các phép toán hình thái học và thuật toán nâng cao (Bài 2)
- edge_and_threshold: Phát hiện biên và các phương pháp phân ngưỡng (Bài 3)
- region_segmentation: Phân đoạn dựa trên vùng và đồ thị (Bài 4)
- visualization: Tiện ích trực quan hóa và lưu ảnh kết quả chuẩn báo cáo
"""

from src.color_utils import (
    rgb_to_hsi,
    convert_color_spaces,
    segment_color_rgb,
    segment_color_hsv,
    evaluate_segmentation_masks,
)

from src.morphology import (
    create_structuring_element,
    my_erosion,
    my_dilation,
    my_opening,
    my_closing,
    boundary_extraction,
    hole_filling,
    conditional_dilation_hole_filling,
    connected_components_labeling,
)

from src.edge_and_threshold import (
    compute_sobel,
    compute_laplacian,
    zero_crossing,
    run_canny,
    my_otsu_threshold,
    run_adaptive_threshold,
    compute_iou,
)

from src.region_segmentation import (
    my_region_growing,
    run_kmeans_segmentation,
    run_slic_superpixels,
    run_felzenszwalb,
    quadtree_split_and_merge,
)

from src.visualization import (
    save_figure_grid,
    save_image_pair,
    save_single_image,
)

__all__ = [
    # Color (Bài 1)
    "rgb_to_hsi",
    "convert_color_spaces",
    "segment_color_rgb",
    "segment_color_hsv",
    "evaluate_segmentation_masks",
    # Morphology (Bài 2)
    "create_structuring_element",
    "my_erosion",
    "my_dilation",
    "my_opening",
    "my_closing",
    "boundary_extraction",
    "hole_filling",
    "conditional_dilation_hole_filling",
    "connected_components_labeling",
    # Edge & Threshold (Bài 3)
    "compute_sobel",
    "compute_laplacian",
    "zero_crossing",
    "run_canny",
    "my_otsu_threshold",
    "run_adaptive_threshold",
    "compute_iou",
    # Region Segmentation (Bài 4)
    "my_region_growing",
    "run_kmeans_segmentation",
    "run_slic_superpixels",
    "run_felzenszwalb",
    "quadtree_split_and_merge",
    # Visualization
    "save_figure_grid",
    "save_image_pair",
    "save_single_image",
]
