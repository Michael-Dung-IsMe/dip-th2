# Thực hành 2 — Hướng dẫn chạy notebook

Các notebook sử dụng ảnh trong `data/` và lưu hình, bảng kết quả vào `outputs/bai_1/` đến `outputs/bai_4/`. Chạy theo thứ tự từ Bài 1 đến Bài 4 bằng kernel của môi trường `dip-th2`.

Từ thư mục gốc dự án, có thể chạy lại một notebook bằng:

```powershell
conda activate dip-th2
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/bai_2_hinh_thai_hoc.ipynb
```

Thay tên file để chạy bài khác. `--inplace` ghi lại output của các cell vào notebook; hãy lưu các chỉnh sửa notebook trước khi chạy. Notebook cũng xác định đường dẫn dữ liệu khi chạy từ thư mục `notebooks/`.

| Bài | Notebook | Nội dung |
| --- | --- | --- |
| 1 | `bai_1_phan_doan_anh_mau.ipynb` | RGB, HSV, HSI, Lab và phân đoạn theo ngưỡng màu |
| 2 | `bai_2_hinh_thai_hoc.ipynb` | Hình thái NumPy, đối chiếu OpenCV, trích biên và làm đầy lỗ |
| 3 | `bai_3_phat_hien_bien_phan_nguong.ipynb` | Sobel, Laplacian, Canny, Otsu, adaptive và ảnh hưởng nhiễu |
| 4 | `bai_4_phan_doan_vung.ipynb` | Region Growing, K-means, SLIC và Felzenszwalb |

Chọn ảnh và số liệu đã chạy để viết nhận xét, ghi tham số và tên ảnh đầu vào. Kiểm tra hình ở kích thước thực tế trong PDF và bổ sung ảnh chụp màn hình minh chứng cá nhân theo yêu cầu nộp bài. Quadtree Split & Merge trong kế hoạch Bài 4 là phần mở rộng.
