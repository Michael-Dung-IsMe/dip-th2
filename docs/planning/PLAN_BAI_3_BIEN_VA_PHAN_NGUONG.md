# Kế Hoạch Triển Khai Bài 3: Phát Hiện Biên và Phân Ngưỡng

## 1. Thông Tin Chung
- **File Notebook thực thi:** `notebooks/bai_3_phat_hien_bien_phan_nguong.ipynb`
- **Thư mục dữ liệu đầu vào:** `data/DIP3E_Original_Images_CH10/`
- **Thư mục xuất kết quả:** `outputs/bai_3/`
- **Quy tắc lưu ảnh báo cáo (Đầu vào — Kết quả):**
  - Lưu PNG ở độ phân giải đủ đọc rõ khi đặt trong PDF; tách lưới tham số thành nhiều hình nếu nhãn và đường biên quá nhỏ:
    1. `<ma_thi_nghiem>_<ten_anh>_input.png`: Ảnh xám đầu vào.
    2. `<ma_thi_nghiem>_<ten_anh>_output_<noi_dung>.png`: Ảnh kết quả đơn hoặc biểu đồ Histogram/Phương sai.
    3. `<ma_thi_nghiem>_<ten_anh>_input_vs_output_<noi_dung>.png`: Ảnh ghép đối sánh trực tiếp **Đầu vào — Kết quả các phương pháp/tham số**.

---

## 2. Danh Sách Ảnh Sử Dụng

| Mã ảnh | Tên file gốc trong `CH10` | Kích thước | Mục đích thực nghiệm |
| :-: | :--- | :---: | :--- |
| **`building`** | `Fig1016(a)`<br>`(building_original).tif` | `1114 × 834`<br>(Ảnh xám) | Khảo sát **Sobel, Laplacian (LoG), Canny** và đánh giá ảnh hưởng của các bộ tham số (`ksize`, $\sigma$, $T_{\text{low}}, T_{\text{high}}$). |
| **`polymersomes`** | `Fig1039(a)`<br>`(polymersomes).tif` | `702 × 648`<br>(Ảnh xám) | Kiểm chứng hàm **tự lập trình Otsu (`my_otsu`)** đối chiếu với OpenCV và scikit-image trên ảnh có Histogram 2 đỉnh rõ nét. |
| **`shaded_text`** | `Fig1049(a)`<br>`(spot_shaded_text_image).tif` | `772 × 686`<br>(Ảnh xám) | So sánh **Ngưỡng cố định (Fixed) vs Otsu toàn cục vs Adaptive Thresholding (Mean & Gaussian)** trên ảnh chiếu sáng không đều. |
| **`septagon_clean`**<br>**`septagon_std10`**<br>**`septagon_std50`** | `Fig1036(a)(original_septagon).tif`<br>`Fig1036(b)(...std_10...).tif`<br>`Fig1036(c)(...std_50...).tif` | `814 × 651`<br>(Bộ 3 ảnh) | Khảo sát định lượng và trực quan **ảnh hưởng của nhiễu Gaussian ($\sigma=0, 10, 50$)** và **nhiễu Muối tiêu** lên cả **Phát hiện biên** lẫn **Phân ngưỡng Otsu**. |

---

## 3. Cấu Trúc Các Cell Trong Notebook (`bai_3_phat_hien_bien_phan_nguong.ipynb`)

### Phần 0: Khởi tạo & Hàm tiện ích
- Import: `numpy`, `cv2`, `matplotlib.pyplot`, `skimage.filters`.
- Hàm đọc ảnh xám và hàm lưu cặp `input` — `output` & `input_vs_output` vào `outputs/bai_3/`.

---

### Phần 1: Thực hiện Sobel, Laplacian, Canny & Khảo sát tham số (trên ảnh `building`)
1. **Toán tử Sobel (Đạo hàm bậc 1):**
   - Tính đạo hàm ngang $G_x = \text{cv2.Sobel}(I, \text{cv2.CV\_64F}, 1, 0, \text{ksize})$ và dọc $G_y = \text{cv2.Sobel}(I, \text{cv2.CV\_64F}, 0, 1, \text{ksize})$.
   - Biên độ gradient gốc: $M = \sqrt{G_x^2 + G_y^2}$. Khi dùng cùng các ngưỡng $T_{\text{grad}}$ để so sánh giữa các `ksize`, chuẩn hóa hoặc hiệu chỉnh theo cùng một quy tắc và ghi rõ quy tắc đó; không chuẩn hóa riêng từng ảnh nếu mục tiêu là so cường độ tuyệt đối.
   - **Tham số khảo sát:**
     - Kích thước mặt nạ `ksize` $\in \{3, 5, 7\}$.
     - Ngưỡng nhị phân hóa biên độ $T_{\text{grad}} \in \{50, 100, 150\}$.
2. **Toán tử Laplacian (Đạo hàm bậc 2) & LoG:**
   - So sánh Laplacian trực tiếp (`ksize` $\in \{1, 3, 5\}$) với **Laplacian of Gaussian (LoG)** (lọc Gaussian với $\sigma \in \{1.0, 2.0\}$ trước khi lấy Laplacian). Nếu hiển thị bản đồ biên nhị phân, quy định thêm cách phát hiện zero-crossing và ngưỡng biên độ để tránh biên yếu do nhiễu; giữ cùng quy tắc hiển thị khi so sánh.
3. **Thuật toán Canny:**
   - **Tham số khảo sát (Ma trận $3 \times 3$ bộ tham số):**
     - Độ lệch chuẩn làm trơn Gaussian trước khi chạy Canny: $\sigma \in \{0.5, 1.5, 3.0\}$.
     - 3 cặp ngưỡng trễ $(T_{\text{low}}, T_{\text{high}})$:
       1. **Ngưỡng thấp `(30, 80)`:** Giữ lại cả các vân gạch nhỏ và nhiễu kết cấu.
       2. **Ngưỡng trung bình `(80, 180)`:** Tách rõ khung nhà, cửa sổ, cột trụ, loại bỏ nhiễu gạch vụn.
       3. **Ngưỡng cao `(150, 240)`:** Chỉ giữ lại các đường biên tương phản cực mạnh, một số đường viền mái bị đứt đoạn.
- **File kết quả xuất ra (`outputs/bai_3/3.1_edge_detection/`):**
  - `3.1_building_input.png`
  - `3.1_building_input_vs_output_sobel_survey.png` *(Đầu vào vs $|G_x|, |G_y|, M$ và khảo sát `ksize`, ngưỡng)*
  - `3.1_building_input_vs_output_laplacian_log.png` *(Đầu vào vs Laplacian `ksize=1,3,5` và LoG)*
  - `3.1_building_input_vs_output_canny_survey.png` *(Đầu vào vs Lưới khảo sát $\sigma$ và $(T_{\text{low}}, T_{\text{high}})$)*
  - `3.1_building_input_vs_output_compare_3_methods.png` *(Tổng hợp Đầu vào vs Sobel vs Laplacian vs Canny)*

---

### Phần 2: Phân ngưỡng Cố định, Otsu (Tự lập trình & Thư viện) và Adaptive Thresholding
1. **Tự lập trình thuật toán Otsu bằng NumPy (`my_otsu_threshold`) & Đối chiếu:**
   - **Cài đặt toán học:**
     - Tính Histogram chuẩn hóa $p_i = n_i / N$ cho $i \in [0, 255]$.
     - Tính xác suất tích lũy $P_1(k) = \sum_{i=0}^k p_i$, trung bình tích lũy $m(k) = \sum_{i=0}^k i \cdot p_i$, trung bình toàn cục $m_G = m(255)$.
     - Tính phương sai giữa hai lớp:
       $$\sigma_B^2(k) = \frac{[m_G P_1(k) - m(k)]^2}{P_1(k)[1 - P_1(k)] + \epsilon}$$
     - Chọn $T^* = \arg\max_{0 \le k \le 255} \sigma_B^2(k)$ trên các ngưỡng có hai lớp khác rỗng; quy định chọn ngưỡng nhỏ nhất nếu nhiều giá trị cùng đạt cực đại. Chỉ tính $\eta^* = \frac{\sigma_B^2(T^*)}{\sigma_G^2}$ khi $\sigma_G^2 > 0$.
   - **Đối chiếu trên ảnh `polymersomes`:**
     - So sánh giá trị $T^*_{\text{NumPy}}$ với `cv2.threshold(..., cv2.THRESH_OTSU)` và `skimage.filters.threshold_otsu`; ghi ngưỡng của từng cách, số pixel sai khác của mask và giải thích nếu có sai khác do quy tắc chọn ngưỡng đồng hạng hoặc dấu bất đẳng thức.
     - Vẽ đồ thị kép: **Histogram mức xám + Đường cong phương sai giữa hai lớp $\sigma_B^2(k)$ + Vạch dấu ngưỡng tối ưu $T^*$**.
2. **So sánh Ngưỡng cố định, Otsu toàn cục và Adaptive Thresholding (trên ảnh `shaded_text`):**
   - **Ngưỡng cố định (Fixed Thresholding):** Khảo sát 3 mức $T \in \{60, 127, 190\}$.
   - **Otsu toàn cục:** Áp dụng `my_otsu_threshold` lên toàn ảnh `shaded_text`; đánh dấu vùng chữ bị mất hoặc nền bị chọn nhầm nếu thực tế quan sát thấy.
   - **Adaptive Thresholding (`cv2.adaptiveThreshold`):**
     - So sánh `ADAPTIVE_THRESH_MEAN_C` và `ADAPTIVE_THRESH_GAUSSIAN_C`.
     - Khảo sát kích thước vùng lân cận `blockSize` $\in \{11, 31, 61\}$ và hằng số bù $C \in \{2, 8, 15\}$.
- **File kết quả xuất ra (`outputs/bai_3/3.2_thresholding/`):**
  - `3.2_polymersomes_input.png`
  - `3.2_polymersomes_input_vs_output_my_otsu_verification.png` *(Đầu vào + Đồ thị $\sigma_B^2(k)$ + Kết quả NumPy vs OpenCV)*
  - `3.2_shaded_text_input.png`
  - `3.2_shaded_text_input_vs_output_fixed_vs_otsu_vs_adaptive.png`
  - `3.2_shaded_text_input_vs_output_adaptive_params_survey.png` *(Khảo sát `blockSize` và `C`)*

---

### Phần 3: Khảo sát ảnh hưởng của Nhiễu đến Phát hiện biên và Phân ngưỡng
- **Dữ liệu thử nghiệm:** Bộ 3 ảnh đa giác 7 cạnh `septagon_clean` ($\sigma=0$), `septagon_std10` ($\sigma=10$), `septagon_std50` ($\sigma=50$) + thêm 1 trường hợp nhiễu Muối tiêu (Salt & Pepper $p = 0.05$) tự tạo từ ảnh sạch bằng bộ sinh số ngẫu nhiên có seed cố định.
1. **Tác động của nhiễu lên Phát hiện biên (Sobel, Laplacian, Canny):**
   - Chạy cả 3 thuật toán trên 3 mức nhiễu.
   - So sánh độ liên tục của biên đa giác và lượng biên giả trên nền ở từng mức nhiễu. Ghi nhận phương pháp nào giữ biên tốt hơn sau khi xem hình; không mặc định Laplacian thất bại ở $\sigma=10$ hoặc Canny luôn tốt nhất.
2. **Tác động của nhiễu lên Phân ngưỡng (Otsu):**
   - Vẽ **Histogram** của ba ảnh và ghi nhận mức độ chồng lấn của hai lớp khi nhiễu tăng; không gọi histogram ở $\sigma=50$ là một đỉnh nếu đồ thị chưa cho thấy điều đó. Vì có ảnh sạch tương ứng, dùng mask từ ảnh sạch làm tham chiếu và báo cáo sai khác pixel hoặc IoU cho mask nhiễu.
   - **Thử nghiệm tiền xử lý:** Lọc Gaussian ($5\times 5$ hoặc $7\times 7$) cho nhiễu Gaussian và lọc trung vị (Median $5\times 5$) cho nhiễu Muối tiêu trước khi chạy Otsu. So sánh với kết quả chưa lọc theo cùng tiêu chí; chỉ kết luận mức cải thiện đo được, không mặc định khôi phục hoàn hảo.
- **File kết quả xuất ra (`outputs/bai_3/3.3_noise_impact/`):**
  - `3.3_septagon_inputs_clean_std10_std50.png`
  - `3.3_septagon_input_vs_output_edges_under_noise.png`
  - `3.3_septagon_input_vs_output_otsu_under_noise_and_denoised.png`

### Gợi ý chọn minh chứng cho báo cáo PDF
- Chọn một hình đối chiếu Sobel–Laplacian–Canny trên cùng ảnh và cùng cách hiển thị, một hình Histogram/Otsu có ngưỡng ghi rõ, một hình Fixed–Otsu–Adaptive, và một hình ảnh hưởng của nhiễu.
- Bảng tham số ghi `ksize`, cách chuẩn hóa gradient, ngưỡng, bộ lọc và tiêu chí đo. Chỉ đưa các nhận xét đã quan sát hoặc tính được vào phần kết quả; giữ lưới khảo sát đầy đủ trong `outputs/`.
- Đưa mã Otsu tự viết và cách tạo nhiễu vào báo cáo hoặc phụ lục; nộp notebook đầy đủ theo yêu cầu tổng quan.
- Đưa ảnh chụp màn hình quá trình chạy/kết quả có thông tin cá nhân vào phần minh chứng theo tài liệu tổng quan.

---

## 4. Tổng Hợp Cấu Trúc Thư Mục Kết Quả Bài 3
```text
outputs/bai_3/
├── 3.1_edge_detection/
│   ├── 3.1_building_input.png
│   ├── 3.1_building_input_vs_output_sobel_survey.png
│   ├── 3.1_building_input_vs_output_laplacian_log.png
│   ├── 3.1_building_input_vs_output_canny_survey.png
│   └── 3.1_building_input_vs_output_compare_3_methods.png
├── 3.2_thresholding/
│   ├── 3.2_polymersomes_input.png
│   ├── 3.2_polymersomes_input_vs_output_my_otsu_verification.png
│   ├── 3.2_shaded_text_input.png
│   ├── 3.2_shaded_text_input_vs_output_fixed_vs_otsu_vs_adaptive.png
│   └── 3.2_shaded_text_input_vs_output_adaptive_params_survey.png
└── 3.3_noise_impact/
    ├── 3.3_septagon_inputs_clean_std10_std50.png
    ├── 3.3_septagon_input_vs_output_edges_under_noise.png
    └── 3.3_septagon_input_vs_output_otsu_under_noise_and_denoised.png
```

