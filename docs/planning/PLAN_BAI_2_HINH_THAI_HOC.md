# Kế Hoạch Triển Khai Bài 2: Xử Lý Ảnh Hình Thái Học

## 1. Thông Tin Chung
- **File Notebook thực thi:** `notebooks/bai_2_hinh_thai_hoc.ipynb`
- **Thư mục dữ liệu đầu vào:** `data/DIP3E_Original_Images_CH09/`
- **Thư mục xuất kết quả:** `outputs/bai_2/`
- **Quy tắc lưu ảnh báo cáo (Đầu vào — Kết quả):**
  - Lưu PNG ở độ phân giải đủ đọc rõ trong PDF; kiểm tra kích thước chữ, SE và bản đồ sai khác khi đặt hình lên trang:
    1. `<ma_thi_nghiem>_<ten_anh>_input.png`: Ảnh nhị phân/xám đầu vào.
    2. `<ma_thi_nghiem>_<ten_anh>_output_<noi_dung>.png`: Ảnh kết quả của từng phép toán/tham số.
    3. `<ma_thi_nghiem>_<ten_anh>_input_vs_output_<noi_dung>.png`: Ảnh đối sánh trực quan **Đầu vào — Kết quả NumPy — Kết quả OpenCV — Bản đồ sai số (`|NumPy - OpenCV|`)**.

---

## 2. Danh Sách Ảnh Sử Dụng

| Mã ảnh | Tên file gốc trong `CH09` | Kích thước | Mục đích thực nghiệm |
| :-: | :--- | :---: | :--- |
| **`wirebond`** | `Fig0905(a)`<br>`(wirebond-mask).tif` | `486 × 486`<br>(Nhị phân) | Kiểm chứng tự lập trình **Erosion, Dilation, Opening, Closing**; khảo sát kích thước & hình dạng SE. |
| **`text_gaps`** | `Fig0907(a)`<br>`(text_gaps_1_and_2_pixels).tif` | `508 × 444`<br>(Nhị phân) | Khảo sát **Dilation & Closing** hàn gắn các nét chữ đứt gãy 1–2 pixels với các hình dạng SE khác nhau. |
| **`fingerprint`** | `Fig0911(a)`<br>`(noisy_fingerprint).tif` | `315 × 238`<br>(Nhị phân) | Khảo sát **Opening, Closing & Chuỗi lọc kết hợp (Opening $\rightarrow$ Closing)** để khử nhiễu dấu vân tay. |
| **`lincoln`** | `Fig0914(a)`<br>`(licoln from penny).tif` | `221 × 269`<br>(Nhị phân) | Thuật toán nâng cao 1: **Trích xuất hình bao / đường biên (Boundary Extraction)**. |
| **`reflections`** | `Fig0916(a)`<br>`(region-filling-reflections).tif` | `512 × 512`<br>(TIFF mode L, chỉ có 0/255) | Thuật toán nâng cao 2: **Làm đầy lỗ (Hole / Region Filling)**; xác định foreground trước khi xử lý. |
| **`chicken_bones`** | `Fig0918(a)`<br>`(Chickenfilet with bones).tif` | `712 × 321`<br>(Ảnh xám) | Thuật toán nâng cao 3: **Thành phần liên thông (Connected Components)** (kèm thêm **Convex Hull & Thinning**). |

---

## 3. Cấu Trúc Các Cell Trong Notebook (`bai_2_hinh_thai_hoc.ipynb`)

### Phần 0: Khởi tạo & Hàm tiện ích
- Import: `numpy`, `cv2`, `matplotlib.pyplot`, `time`, `skimage.morphology`, `scipy.ndimage`.
- Hàm đọc ảnh TIFF nhị phân (Mode 1 trong PIL / OpenCV) và chuẩn hóa về mảng `uint8` có giá trị `{0, 255}` (hoặc `{0, 1}`).
- Hàm xuất cặp file `input` — `output` & `input_vs_output` tự động vào `outputs/bai_2/`.

---

### Phần 1: Tự lập trình Erosion, Dilation, Opening, Closing bằng NumPy & So sánh với OpenCV
- **Thiết kế thuật toán bằng NumPy (Vectorized Sliding Window):**
  - Sử dụng `np.lib.stride_tricks.sliding_window_view` kết hợp `np.pad` cho SE nhỏ. Với SE 45×45 trên ảnh 486×486, tránh tạo mảng trung gian kích thước ảnh × 2025 vị trí; xử lý theo lô hoặc dùng cách cài đặt tiết kiệm bộ nhớ, rồi mới đối chiếu thư viện:
    - **`my_erosion(img, se)`:** Pad biên ảnh (giá trị `255` hoặc xử lý tương đương `cv2.BORDER_CONSTANT`), trích xuất các cửa sổ trượt kích thước $(k_h, k_w)$, kiểm tra điều kiện mọi vị trí ứng với `se == 1` đều bằng `255`: `np.all(windows[..., se == 1] == 255, axis=-1)`.
    - **`my_dilation(img, se)`:** Lật đối xứng phần tử cấu trúc $\hat{B} = \text{se}[::-1, ::-1]$, pad biên giá trị `0`, kiểm tra điều kiện tồn tại ít nhất 1 vị trí ứng với $\hat{B} == 1$ bằng `255`: `np.any(windows[..., se_flipped == 1] == 255, axis=-1)`.
    - **`my_opening(img, se)`:** `my_dilation(my_erosion(img, se), se)`.
    - **`my_closing(img, se)`:** `my_erosion(my_dilation(img, se), se)`.
- **Tiêu chí đối sánh với OpenCV (`cv2.erode`, `cv2.dilate`, `cv2.morphologyEx`):**
  1. **Độ chính xác:** Tính `diff = np.abs(my_res.astype(int) - cv_res.astype(int))`, số pixel sai khác và sai khác lớn nhất. Kỳ vọng bằng 0 sau khi thống nhất foreground, điểm neo SE, phép lật SE và quy tắc biên; nếu khác, xác định vị trí và nguyên nhân.
  2. **Tốc độ thực thi (`ms`):** Lập bảng trên cùng ảnh/SE, đo lặp vài lần sau một lượt khởi động và ghi cấu hình máy. Ưu tiên 3 ảnh `wirebond`, `text_gaps`, `fingerprint`.
- **File kết quả xuất ra (`outputs/bai_2/2.1_numpy_vs_opencv/`):**
  - `2.1_wirebond_input.png`
  - `2.1_wirebond_input_vs_output_4_ops.png` *(Đầu vào vs Erosion, Dilation, Opening, Closing của NumPy & OpenCV)*
  - `2.1_fingerprint_input.png`
  - `2.1_fingerprint_input_vs_output_denoise.png` *(Đầu vào vs Opening, Closing, và Opening+Closing)*

---

### Phần 2: Thử nghiệm các Phần tử cấu trúc (Structuring Elements - SE) có hình dạng & kích thước khác nhau
- **Thiết kế thực nghiệm:**
  - **3 Hình dạng SE:**
    1. Hình vuông / chữ nhật (`RECT`): Đẳng hướng theo chuẩn $L_\infty$, giữ các góc vuông của mạch điện.
    2. Hình chữ thập (`CROSS`): Chỉ xét 4 lân cận ngang/dọc (chuẩn $L_1$), giữ các nét thẳng đứng/ngang tốt hơn đường chéo.
    3. Hình tròn / elip (`ELLIPSE` / `DISK`): Gần đẳng hướng Euclid ($L_2$), làm trơn đều các góc cong.
  - **4 Kích thước SE:**
    - Trên ảnh `wirebond` ($486\times 486$): Khảo sát $k \in \{3\times 3, 11\times 11, 15\times 15, 45\times 45\}$ (bám sát thí nghiệm gốc trong sách Gonzalez Hình 9.5: $11\times 11$ xóa dây dẫn mảnh, $15\times 15$ xóa vòng đệm, $45\times 45$ chỉ giữ lại khối vuông trung tâm).
    - Trên ảnh `text_gaps` ($508\times 444$): Khảo sát $k \in \{3\times 3, 5\times 5, 9\times 9\}$ với 3 hình dạng `RECT`, `CROSS`, `ELLIPSE` để xem SE nào nối liền nét chữ đứt mà ít làm dính các chữ cái kề nhau nhất.
- **File kết quả xuất ra (`outputs/bai_2/2.2_structuring_elements/`):**
  - `2.2_se_shapes_and_sizes_visual.png` *(Hình minh họa trực quan các ma trận SE)*
  - `2.2_wirebond_input_vs_output_erosion_sizes.png`
  - `2.2_wirebond_input_vs_output_erosion_shapes.png`
  - `2.2_text_gaps_input_vs_output_dilation_shapes_sizes.png`

---

### Phần 3: Thuật toán hình thái học nâng cao (hoàn thành tối thiểu 2)

Ưu tiên hoàn thành **Boundary Extraction** và **Hole Filling** cùng ảnh đầu vào, tham số và nhận xét. **Connected Components** và **Thinning / Convex Hull** là phần mở rộng nếu còn thời gian; không cần làm đủ cả bốn để đáp ứng yêu cầu.

1. **Thuật toán 1 — Trích xuất hình bao / đường biên (Boundary Extraction) trên ảnh `lincoln`:**
   - Công thức tự cài đặt: $\beta(A) = A - (A \ominus B)$ sử dụng hàm `my_erosion` với SE $3\times 3$ và $5\times 5$.
   - Đo và nhận xét độ dày, độ liên tục của biên khi tăng kích thước SE hoặc đổi giữa `CROSS` và `RECT`; không mặc định biên dày 1 pixel hoặc gán sẵn loại liên thông cho từng SE.
2. **Thuật toán 2 — Làm đầy lỗ (Hole Filling) trên ảnh `reflections`:**
   - Ảnh đầu vào là mode L nhưng chỉ có 0/255; chuyển về mask Boolean và kiểm tra đối tượng là trắng hay đen trước khi áp dụng công thức.
   - Cài đặt giãn có điều kiện từ một seed nằm trong lỗ: $X_k = (X_{k-1} \oplus B) \cap A^c$ đến khi hội tụ, sau đó lấy $A \cup X_k$; để lấp tất cả lỗ, dùng seed cho từng lỗ hoặc flood-fill nền từ biên ngoài rồi lấy phần nền bị bao kín.
   - Hiển thị vài bước lặp trung gian và kết quả hội tụ; đánh dấu seed hoặc giải thích cách chọn foreground trong chú thích hình.
3. **Mở rộng — Trích xuất thành phần liên thông (Connected Components) trên ảnh `chicken_bones`:**
   - Tiền xử lý ngưỡng hóa + Erosion nhỏ để tách các mảnh xương trong miếng thịt gà.
   - Thực hiện gán nhãn liên thông (`my_connected_components` lặp $X_k = (X_{k-1} \oplus B) \cap A$ & đối chiếu `cv2.connectedComponentsWithStats`).
   - Tô màu giả (pseudo-color) cho từng mảnh xương và xuất bảng thống kê số lượng pixel (diện tích) + bounding box của từng thành phần.
4. **Mở rộng — Làm mảnh (Thinning) & Bao lồi (Convex Hull) bổ trợ:**
   - Thực hiện `skimage.morphology.thin` (trích khung xương 1 pixel) và `convex_hull_image` trên ảnh `fingerprint` / `lincoln`.

- **File kết quả xuất ra (`outputs/bai_2/2.3_advanced_algorithms/`):**
  - `2.3_lincoln_input.png` & `2.3_lincoln_input_vs_output_boundary.png`
  - `2.3_reflections_input.png` & `2.3_reflections_input_vs_output_hole_filling.png`
  - Nếu làm mở rộng: `2.3_chicken_bones_input.png` & `2.3_chicken_bones_input_vs_output_connected_components.png`
  - Nếu làm mở rộng: `2.3_lincoln_input_vs_output_convex_hull_thinning.png`

### Gợi ý chọn minh chứng cho báo cáo PDF
- Chọn một ảnh đối chiếu NumPy–OpenCV cho bốn phép cơ bản, một hình so sánh SE, và mỗi thuật toán nâng cao bắt buộc một hình đầu vào–kết quả. Giữ ma trận tham số đầy đủ trong `outputs/`.
- Bảng đối chiếu ghi số pixel sai khác, thời gian đo, SE, quy tắc biên và quy ước foreground. Với ảnh xám phải ghi bước ngưỡng hóa trước khi đánh giá hình thái hoặc thành phần liên thông.
- Đưa mã bốn phép hình thái tự viết và thuật toán làm đầy lỗ vào báo cáo hoặc phụ lục; nộp notebook đầy đủ theo yêu cầu tổng quan.
- Đưa ảnh chụp màn hình quá trình chạy/kết quả có thông tin cá nhân vào phần minh chứng theo tài liệu tổng quan.

---

## 4. Tổng Hợp Cấu Trúc Thư Mục Kết Quả Bài 2
Các file của Connected Components và Thinning / Convex Hull trong sơ đồ sau chỉ tạo khi thực hiện phần mở rộng.
```text
outputs/bai_2/
├── 2.1_numpy_vs_opencv/
│   ├── 2.1_wirebond_input.png
│   ├── 2.1_wirebond_input_vs_output_4_ops.png
│   ├── 2.1_fingerprint_input.png
│   └── 2.1_fingerprint_input_vs_output_denoise.png
├── 2.2_structuring_elements/
│   ├── 2.2_se_shapes_and_sizes_visual.png
│   ├── 2.2_wirebond_input_vs_output_erosion_sizes.png
│   ├── 2.2_wirebond_input_vs_output_erosion_shapes.png
│   ├── 2.2_text_gaps_input.png
│   └── 2.2_text_gaps_input_vs_output_dilation_shapes_sizes.png
└── 2.3_advanced_algorithms/
    ├── 2.3_lincoln_input.png
    ├── 2.3_lincoln_input_vs_output_boundary.png
    ├── 2.3_reflections_input.png
    ├── 2.3_reflections_input_vs_output_hole_filling.png
    ├── 2.3_chicken_bones_input.png
    ├── 2.3_chicken_bones_input_vs_output_connected_components.png
    └── 2.3_lincoln_input_vs_output_convex_hull_thinning.png
```

