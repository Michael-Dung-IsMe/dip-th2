# Kế Hoạch Triển Khai Bài 4: Phân Đoạn Ảnh Dựa Trên Vùng

## 1. Thông Tin Chung
- **File Notebook thực thi:** `notebooks/bai_4_phan_doan_vung.ipynb`
- **Thư mục dữ liệu đầu vào:** `data/DIP3E_Original_Images_CH10/` & `data/DIP3E_Original_Images_CH06/`
- **Thư mục xuất kết quả:** `outputs/bai_4/`
- **Quy tắc lưu ảnh báo cáo (Đầu vào — Kết quả):**
  - Lưu PNG ở độ phân giải đủ đọc rõ khi đặt trong PDF; tách lưới tham số thành nhiều hình nếu nhãn vùng hoặc seed quá nhỏ:
    1. `<ma_thi_nghiem>_<ten_anh>_input.png`: Ảnh gốc đầu vào (có thể kèm dấu cộng `+` đỏ đánh dấu vị trí Seed cho Region Growing).
    2. `<ma_thi_nghiem>_<ten_anh>_output_<noi_dung>.png`: Ảnh kết quả phân vùng / siêu điểm ảnh.
    3. `<ma_thi_nghiem>_<ten_anh>_input_vs_output_<noi_dung>.png`: Ảnh ghép đối sánh trực tiếp **Đầu vào — Kết quả các tham số/phương pháp**.

---

## 2. Danh Sách Ảnh Sử Dụng

| Mã ảnh | Tên file gốc & Nguồn | Kích thước | Mục đích thực nghiệm |
| :-: | :--- | :---: | :--- |
| **`weld`** | `[CH10]` `Fig1051(a)`<br>`(defective_weld).tif` | `566 × 482`<br>(Ảnh xám) | Tự lập trình **Region Growing (BFS)**: khảo sát vị trí điểm hạt giống (Seed) và ngưỡng đồng nhất $T$. |
| **`strawberries`** | `[CH06]` `Fig0630(01)`<br>`(strawberries_fullcolor).tif` | `666 × 666`<br>(RGB) | Phân đoạn bằng **K-Means ($K = 2, 3, 4, 5$)**, **SLIC Superpixels**, và **Graph-based (Felzenszwalb)** trên ảnh màu. |
| **`lenna`** | `[CH06]` `Fig0638(a)`<br>`(lenna_RGB).tif` | `512 × 512`<br>(RGB) | Khảo sát **SLIC Superpixels** với các cặp tham số (`n_segments`, `compactness`) và so sánh đa phương pháp. |
| **`iceberg`** | `[CH10]` `Fig1045(a)`<br>`(iceberg).tif` | `566 × 533`<br>(Ảnh xám) | Đối chiếu phân vùng trên ảnh xám; ưu tiên **Graph-based Segmentation**, Quadtree là phần mở rộng. |

---

## 3. Cấu Trúc Các Cell Trong Notebook (`bai_4_phan_doan_vung.ipynb`)

### Phần 0: Khởi tạo & Hàm tiện ích
- Import: `numpy`, `cv2`, `matplotlib.pyplot`, `collections.deque`, `skimage.segmentation` (`slic`, `felzenszwalb`, `mark_boundaries`), `skimage.color`.
- Cố định bộ sinh số ngẫu nhiên của thuật toán được chọn: nếu dùng `cv2.kmeans`, gọi `cv2.setRNGSeed(42)` và ghi số lần khởi tạo; nếu dùng scikit-learn, truyền `random_state=42`. Ghi phiên bản thư viện, không hứa kết quả giống tuyệt đối giữa các môi trường.

---

### Phần 1: Tự lập trình Region Growing & Khảo sát vị trí Seed, Ngưỡng đồng nhất $T$
- **Thiết kế thuật toán `my_region_growing(img, seeds, threshold, mode='seed_diff', connectivity=8)`:**
  - Sử dụng hàng đợi `collections.deque` (BFS) và ma trận `visited` để mở rộng vùng từ tập điểm hạt giống `seeds = [(y0, x0), ...]`.
  - Lưu tọa độ, giá trị pixel tại mỗi seed và quy tắc xử lý nhiều seed. Với `mean_diff`, ghi rõ cách cập nhật trung bình và thứ tự duyệt vì chúng có thể ảnh hưởng kết quả.
  - Hỗ trợ 2 chế độ tiêu chuẩn đồng nhất (Homogeneity Criterion):
    1. **So với cường độ hạt giống ban đầu (`seed_diff`):** $|I(y, x) - I(y_{\text{seed}}, x_{\text{seed}})| \le T$.
    2. **So với trung bình động của vùng đang phát triển (`mean_diff`):** $|I(y, x) - \mu_{\text{region}}| \le T$ (cập nhật $\mu_{\text{region}}$ mỗi khi thêm pixel mới).
- **Thiết kế khảo sát trên ảnh `weld` (`Fig1051(a)(defective_weld).tif`) và `iceberg`:**
  1. **Khảo sát vị trí Seed (3 cấu hình vị trí):**
     - *Seed 1 (trong vùng khuyết tật cần tách):* Chọn sau khi kiểm tra ảnh, ghi tọa độ và cường độ thật tại điểm đó.
     - *Seed 2 (vùng biên chuyển tiếp hoặc nền):* Dùng để đánh giá độ nhạy theo vị trí, không mặc định sẽ tách đúng khuyết tật.
     - *Seed 3 (multi-seed trong các vùng rời rạc):* Ghi từng tọa độ và kiểm tra liệu các vùng thu được có bị nối/lẫn nền.
  2. **Khảo sát ngưỡng đồng nhất $T$ (4 mức ngưỡng: $T \in \{10, 30, 65, 110\}$):**
     - Với mỗi $T$, đo diện tích mask và đánh dấu vùng bị bỏ sót hoặc tràn sang nền trên ảnh gốc. Các giá trị 10, 30, 65, 110 là điểm khảo sát ban đầu; điều chỉnh sau khi xem dải cường độ của vùng đích.
     - Chỉ gọi một ngưỡng là phù hợp hoặc có rò rỉ vùng khi hình/mask quan sát được hỗ trợ nhận xét đó.
- **File kết quả xuất ra (`outputs/bai_4/4.1_region_growing/`):**
  - `4.1_weld_input.png`
  - `4.1_weld_input_vs_output_seed_positions.png` *(Đầu vào đánh dấu vị trí Seed vs Kết quả từng vị trí & Multi-seed)*
  - `4.1_weld_input_vs_output_threshold_survey.png` *(Đầu vào vs 4 mức ngưỡng $T = 10, 30, 65, 110$ kèm đường viền đỏ overlay lên ảnh gốc)*
  - `4.1_iceberg_input_vs_output_region_growing.png`

---

### Phần 2: Phân đoạn bằng K-Means ($K = 2, 3, 4, 5$) và SLIC Superpixels
1. **Phân đoạn K-Means với $K \in \{2, 3, 4, 5\}$ (trên ảnh `strawberries` và `iceberg`):**
   - Duỗi ảnh thành ma trận đặc trưng $(H\cdot W, C)$, chạy `cv2.kmeans` (hoặc tự cài đặt K-Means++ / `sklearn`) với $K = 2, 3, 4, 5$.
   - Thực nghiệm mở rộng: So sánh K-Means trong không gian **RGB** và không gian **CIELAB** (hoặc ghép thêm tọa độ không gian $[L^*, a^*, b^*, \alpha x, \alpha y]$ để tăng tính liên kết vùng).
   - **Ý nhận xét:**
     - Với từng $K$, hiển thị tâm cụm và mask từng cụm; mô tả cụm nào chứa quả dâu, lá hoặc nền sau khi xem kết quả. Cụm màu không tự tương ứng một-một với lớp ngữ nghĩa.
2. **Phân đoạn SLIC Superpixels (trên ảnh `lenna` và `strawberries`):**
   - Sử dụng `skimage.segmentation.slic(img, n_segments=..., compactness=..., start_label=1)`.
   - Tạo cả 2 dạng hiển thị cho mỗi cấu hình:
     - **Bản đồ đường biên superpixels (`mark_boundaries`)** đè lên ảnh gốc.
     - **Ảnh tô màu trung bình của từng superpixel (`label2rgb(..., kind='avg')`)** để thấy rõ mức độ bảo toàn chi tiết.
   - **Lưới tham số khảo sát ($3 \times 3$):**
     - Số lượng siêu điểm ảnh: `n_segments` $\in \{60, 200, 500\}$.
     - Độ đặc không gian: `compactness` $m \in \{1, 10, 40\}$.
   - Ghi số superpixel thực tế và phiên bản scikit-image; `n_segments` là số mục tiêu, kết quả có thể khác. Với ảnh xám `iceberg`, cấu hình `channel_axis=None` nếu chạy SLIC.
- **File kết quả xuất ra (`outputs/bai_4/4.2_kmeans_and_slic/`):**
  - `4.2_strawberries_input.png`
  - `4.2_strawberries_input_vs_output_kmeans_k2_to_k5.png` *(Đầu vào vs $K=2, 3, 4, 5$)*
  - `4.2_iceberg_input_vs_output_kmeans_k2_to_k5.png`
  - `4.2_lenna_input.png`
  - `4.2_lenna_input_vs_output_slic_nsegments_survey.png`
  - `4.2_lenna_input_vs_output_slic_compactness_survey.png`
  - `4.2_strawberries_input_vs_output_slic_survey.png`

---

### Phần 3: Thực hiện thêm phương pháp nâng cao & So sánh tổng thể

Ưu tiên hoàn thành **một phương pháp nâng cao** là Felzenszwalb để đáp ứng yêu cầu. Quadtree Split & Merge là phần mở rộng nếu còn thời gian sau khi hoàn thiện kết quả và nhận xét của ba phương pháp chính.
1. **Phương pháp nâng cao chính — Graph-based Segmentation (Felzenszwalb & Huttenlocher):**
   - Sử dụng `skimage.segmentation.felzenszwalb(img, scale=..., sigma=..., min_size=...)`.
   - Khảo sát `scale` $\in \{50, 200, 500\}$ và `min_size` $\in \{20, 100\}$; ghi số vùng thực tế, thời gian chạy và đường biên trên ảnh gốc. Với ảnh xám, truyền `channel_axis=None` nếu phiên bản thư viện yêu cầu.
2. **Mở rộng — Region Splitting and Merging (Quadtree tự lập trình):**
   - **Split:** Chia vùng $R$ khi $\sigma_R > T_{\text{split}}$ và vùng lớn hơn $s_{\min}$; xử lý rõ các kích thước lẻ.
   - **Merge:** Tạo quan hệ kề giữa các lá; gộp các vùng kề nhau khi $|\mu_i-\mu_j|\le T_{\text{merge}}$, cập nhật đặc trưng vùng sau mỗi lần gộp và xuất bản đồ nhãn. Chỉ vẽ ô Quadtree hoặc tô mỗi lá theo giá trị trung bình là kết quả split, chưa phải split & merge.
   - Nếu làm, ưu tiên ảnh `iceberg` và khảo sát ba mức $T_{\text{split}}$; ghi thêm $T_{\text{merge}}$ và số vùng trước/sau gộp.
3. **So sánh trên cùng dữ liệu:**
   - Trên ảnh `iceberg`, đặt kết quả Region Growing, K-Means, SLIC và Felzenszwalb cạnh nhau để mô tả loại đầu ra, số vùng, độ liên tục và thời gian chạy. Nếu làm Quadtree, thêm cột riêng.
   - Region Growing tạo mask theo seed; K-Means tạo cụm màu/cường độ; SLIC tạo superpixel; Felzenszwalb tạo nhiều vùng. Muốn so sánh độ chính xác, xác định cùng đối tượng đích, chuyển mỗi kết quả thành mask của đối tượng đó và đối chiếu với một mask tham chiếu bằng IoU hoặc nhận xét vùng bỏ sót/chọn nhầm. Nếu chưa có mask tham chiếu, chỉ so sánh định tính, không xếp hạng độ chính xác.
   - Trên ảnh `strawberries`, dùng riêng K-Means, SLIC và Felzenszwalb để minh họa sự khác biệt trên ảnh màu; chỉ thêm Region Growing nếu đã định nghĩa tiêu chí tương đồng màu và seed thích hợp.
- **File kết quả xuất ra (`outputs/bai_4/4.3_advanced_and_comparison/`):**
  - Nếu làm mở rộng: `4.3_iceberg_input_vs_output_quadtree_split_merge.png` *(kèm bản đồ nhãn sau gộp)*
  - `4.3_strawberries_input_vs_output_felzenszwalb_survey.png`
  - `4.3_methods_comparison_strawberries.png` *(K-Means, SLIC, Felzenszwalb trên ảnh màu)*
  - `4.3_methods_comparison_iceberg.png` *(Region Growing, K-Means, SLIC, Felzenszwalb; thêm Quadtree nếu có)*

### Gợi ý chọn minh chứng cho báo cáo PDF
- Chọn một hình Region Growing có tọa độ seed và ngưỡng, một hình khảo sát $K$, một hình SLIC có số vùng thực tế, một hình Felzenszwalb và một hình so sánh trên `iceberg`. Giữ các lưới tham số đầy đủ trong `outputs/`.
- Chú thích hình ghi rõ đầu ra là mask đối tượng, cụm màu hay bản đồ vùng. Bảng so sánh ghi cùng ảnh, tham số, số vùng và thời gian; chỉ ghi IoU khi các phương pháp đã được quy về cùng mask đích.
- Đưa mã Region Growing tự viết và cách chọn seed vào báo cáo hoặc phụ lục; nộp notebook đầy đủ theo yêu cầu tổng quan.
- Đưa ảnh chụp màn hình quá trình chạy/kết quả có thông tin cá nhân vào phần minh chứng theo tài liệu tổng quan.

---

## 4. Tổng Hợp Cấu Trúc Thư Mục Kết Quả Bài 4
File Quadtree trong sơ đồ sau chỉ tạo nếu thực hiện phần mở rộng.
```text
outputs/bai_4/
├── 4.1_region_growing/
│   ├── 4.1_weld_input.png
│   ├── 4.1_weld_input_vs_output_seed_positions.png
│   ├── 4.1_weld_input_vs_output_threshold_survey.png
│   ├── 4.1_iceberg_input.png
│   └── 4.1_iceberg_input_vs_output_region_growing.png
├── 4.2_kmeans_and_slic/
│   ├── 4.2_strawberries_input.png
│   ├── 4.2_strawberries_input_vs_output_kmeans_k2_to_k5.png
│   ├── 4.2_iceberg_input_vs_output_kmeans_k2_to_k5.png
│   ├── 4.2_lenna_input.png
│   ├── 4.2_lenna_input_vs_output_slic_nsegments_survey.png
│   ├── 4.2_lenna_input_vs_output_slic_compactness_survey.png
│   └── 4.2_strawberries_input_vs_output_slic_survey.png
└── 4.3_advanced_and_comparison/
    ├── 4.3_iceberg_input_vs_output_quadtree_split_merge.png
    ├── 4.3_strawberries_input_vs_output_felzenszwalb_survey.png
    ├── 4.3_methods_comparison_strawberries.png
    └── 4.3_methods_comparison_iceberg.png
```

