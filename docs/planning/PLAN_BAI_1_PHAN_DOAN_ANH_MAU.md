# Kế Hoạch Triển Khai Bài 1: Phân Tích và Phân Đoạn Ảnh Màu

## 1. Thông Tin Chung
- **File Notebook thực thi:** `notebooks/bai_1_phan_doan_anh_mau.ipynb`
- **Thư mục dữ liệu đầu vào:** `data/DIP3E_Original_Images_CH06/`
- **Thư mục xuất kết quả:** `outputs/bai_1/`
- **Quy tắc lưu ảnh báo cáo (Đầu vào — Kết quả):**
  - Lưu ảnh PNG ở độ phân giải đủ đọc rõ khi đặt trong PDF; 300 DPI là thiết lập xuất hình, không thay thế việc kiểm tra kích thước chữ và từng ô ảnh trên trang:
    1. `<ma_thi_nghiem>_<ten_anh>_input.png`: Ảnh gốc đầu vào.
    2. `<ma_thi_nghiem>_<ten_anh>_output_<noi_dung>.png`: Ảnh kết quả đơn lẻ hoặc lưới kênh/mặt nạ.
    3. `<ma_thi_nghiem>_<ten_anh>_input_vs_output_<noi_dung>.png`: Ảnh ghép đối sánh trực tiếp **Đầu vào (bên trái) — Kết quả (bên phải)** để chèn ngay vào báo cáo PDF.

---

## 2. Danh Sách Ảnh Sử Dụng

| Mã ảnh | Tên file gốc trong `CH06` | Kích thước | Nhóm đối tượng màu cần phân đoạn |
| :-: | :--- | :---: | :--- |
| **`strawberries`** | `Fig0630(01)`<br>`(strawberries_fullcolor).tif` | `666 × 666`<br>(RGB) | • **Nhóm 1:** Quả dâu tây (**Màu Đỏ** — có bóng râm & đốm phản xạ trắng).<br>• **Nhóm 2:** Cuống và lá dâu (**Màu Xanh lá**). |
| **`chalk`** | `Fig0635`<br>`(middle_row_left_chalk ).tif` | `1040 × 1040`<br>(RGB) | • **Nhóm 1:** Các viên phấn **Màu Vàng (Yellow)**.<br>• **Nhóm 2:** Các viên phấn **Màu Xanh dương (Blue)** (hoặc Đỏ). |

---

## 3. Cấu Trúc Các Cell Trong Notebook (`bai_1_phan_doan_anh_mau.ipynb`)

### Phần 0: Khởi tạo & Hàm tiện ích (Helper Functions)
- Import thư viện: `numpy`, `cv2`, `matplotlib.pyplot`, `pathlib`.
- Hàm `save_input_and_output(img_in, fig_out, exp_id, img_name, desc)`:
  - Tự động tạo thư mục `outputs/bai_1/<exp_id>/`.
  - Lưu ảnh gốc `<exp_id>_<img_name>_input.png`.
  - Lưu Figure kết quả `<exp_id>_<img_name>_output_<desc>.png` và ảnh ghép đối sánh `<exp_id>_<img_name>_input_vs_output_<desc>.png`.
- Hàm tự cài đặt `rgb_to_hsi(rgb_img)` bằng NumPy theo công thức chuẩn Gonzalez & Woods:
  - Chuẩn hóa $R, G, B \in [0, 1]$.
  - Tính góc $\theta = \arccos\left(\frac{0.5[(R-G)+(R-B)]}{\sqrt{(R-G)^2+(R-B)(G-B)} + \epsilon}\right)$.
  - $H = \theta$ nếu $B \le G$, ngược lại $360^\circ - \theta$; giữ Hue theo độ trong $[0, 360)$ và chỉ đổi thang đo khi so với HSV của OpenCV.
  - $S = 1 - \frac{3}{R+G+B+\epsilon}\min(R,G,B)$, $I = \frac{R+G+B}{3}$.
  - Với pixel gần xám hoặc quá tối, Hue không ổn định; xử lý riêng các trường hợp này trước khi tính góc và luôn kết hợp điều kiện $S$ khi phân đoạn.

---

### Phần 1: Chuyển đổi không gian màu RGB sang HSV, HSI và CIELAB & Hiển thị kênh màu
- **Mục tiêu:** Thực hiện trên cả 2 ảnh `strawberries` và `chalk`.
- **Các bước xử lý:**
  1. Đọc ảnh BGR bằng `cv2.imread`, chuyển sang RGB (`cv2.COLOR_BGR2RGB`).
  2. Chuyển sang **HSV** bằng `cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)` $\rightarrow$ tách $(H, S, V)$.
  3. Chuyển sang **HSI** bằng hàm `rgb_to_hsi(img_rgb)` $\rightarrow$ tách $(H, S, I)$.
  4. Chuyển sang **CIELAB** bằng `cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)`; với đầu vào `uint8`, ghi rõ thang OpenCV: $L\in[0,255]$, $a,b\in[0,255]$ và giá trị trung tính của $a,b$ xấp xỉ 128. Nếu dùng ký hiệu $L^*,a^*,b^*$ theo thang CIELAB chuẩn, chuyển từ thang OpenCV trước khi ghi số liệu.
  5. Vẽ lưới hiển thị từng kênh màu kèm thanh giá trị (`colorbar`) và ghi rõ đơn vị/thang đo để quan sát phân bố cường độ.
- **File kết quả xuất ra (`outputs/bai_1/1.1_color_spaces/`):**
  - `1.1_strawberries_input.png`
  - `1.1_strawberries_output_rgb_channels.png`
  - `1.1_strawberries_output_hsv_hsi_channels.png`
  - `1.1_strawberries_output_cielab_channels.png`
  - `1.1_strawberries_input_vs_output_all_spaces.png` *(Ảnh tổng hợp để lưu kết quả; trong PDF chọn các kênh làm rõ đối tượng và chia thành nhiều hình nếu chữ hoặc colorbar quá nhỏ)*
  - *(Tương tự bộ 5 file cho ảnh `chalk`)*
- **Ý chính cần nhận xét trong báo cáo:**
  - Trong RGB, cả 3 kênh đều chứa thông tin độ sáng lẫn cấu trúc hình học.
  - Kiểm tra mức độ ổn định của $H$ ở vùng có bóng và vùng phản xạ; với pixel tối hoặc ít bão hòa, $H$ có thể nhiễu nên cần xem cùng kênh $S$.
  - Trên thang `uint8` của OpenCV, kiểm tra liệu kênh $a$ làm nổi bật cặp Đỏ/Xanh lá và kênh $b$ làm nổi bật cặp Vàng/Xanh dương; chỉ kết luận mức độ tách biệt sau khi xem ảnh và phân bố giá trị.

---

### Phần 2: Phân đoạn 2 nhóm đối tượng theo màu — So sánh RGB và HSV/HSI
- **Mục tiêu:** Phân đoạn 2 nhóm màu trên từng ảnh và đặt cạnh nhau để đối chiếu hiệu quả giữa RGB và HSV/HSI.
- **Thiết kế thực nghiệm:**
  - Ghi rõ Hue theo độ ($[0,360)$) hay theo OpenCV `uint8` ($[0,179]$); khi dùng `cv2.inRange`, chuyển các khoảng Hue theo độ sang thang OpenCV và xử lý màu đỏ bằng hai khoảng ở hai đầu trục.
  - **Trên ảnh `strawberries`:**
    - *Nhóm đối tượng 1 (Quả dâu Đỏ):*
      - **RGB:** Ngưỡng hộp `R > 150, G < 70, B < 70` (hoặc khoảng cách Euclid tới tâm màu đỏ mẫu $\mathbf{a}_{\text{red}}$).
      - **HSV/HSI:** Ngưỡng góc màu đỏ ($H \in [0^\circ, 20^\circ] \cup [340^\circ, 360^\circ]$ kết hợp $S > 0.35$).
    - *Nhóm đối tượng 2 (Lá dâu Xanh lá):*
      - **RGB:** Ngưỡng hộp `G > R + 15` và `G > B + 15` (hoặc `inRange`).
      - **HSV/HSI:** Ngưỡng góc màu xanh lá ($H \in [65^\circ, 160^\circ], S > 0.25$).
  - **Trên ảnh `chalk`:**
    - *Nhóm đối tượng 1 (Phấn Vàng):* So sánh phân đoạn trong RGB và HSV/HSI ($H \in [40^\circ, 70^\circ]$).
    - *Nhóm đối tượng 2 (Phấn Xanh dương):* So sánh phân đoạn trong RGB và HSV/HSI ($H \in [190^\circ, 250^\circ]$).
- **Trình bày trực quan (Đầu vào — Kết quả):**
  - Mỗi nhóm màu xuất 1 Figure gồm: `[Ảnh Đầu vào] | [Mask RGB] | [Kết quả tách RGB] | [Mask HSV/HSI] | [Kết quả tách HSV/HSI]`.
- **File kết quả xuất ra (`outputs/bai_1/1.2_segmentation_rgb_vs_hsv/`):**
  - `1.2_strawberries_input.png`
  - `1.2_strawberries_input_vs_output_red_group.png`
  - `1.2_strawberries_input_vs_output_green_group.png`
  - `1.2_chalk_input.png`
  - `1.2_chalk_input_vs_output_yellow_group.png`
  - `1.2_chalk_input_vs_output_blue_group.png`

---

### Phần 3: Khảo sát ít nhất 3 bộ tham số ngưỡng & Nhận xét
- **Mục tiêu:** Khảo sát độ nhạy của kết quả phân đoạn khi thay đổi 3 bộ tham số ngưỡng: **Bộ 1 (Ngưỡng Hẹp / Strict)**, **Bộ 2 (Ngưỡng Vừa / Balanced)**, **Bộ 3 (Ngưỡng Rộng / Loose)**. Chỉ gọi một bộ là tối ưu nếu có tiêu chí đánh giá và kết quả đo hỗ trợ.
- **Bảng tham số khảo sát ban đầu (ví dụ cho nhóm Quả dâu Đỏ trong HSV & RGB; chỉnh lại sau khi xem mask):**

| Bộ tham số | Thiết lập trong RGB (`[R_min, G_max, B_max]` / Bán kính $D_0$) | Thiết lập trong HSV (`H_range`, `S_min`, `V_min`) | Điều cần kiểm tra trên kết quả |
| :--- | :--- | :--- | :--- |
| **Bộ 1: Ngưỡng hẹp (Strict)** | $R \in [190, 255]$, $G \le 45$, $B \le 45$<br>*(hoặc $D_0 = 35$)* | $H \in [0, 6] \cup [175, 179]$<br>$S \ge 160, V \ge 120$ | Kiểm tra vùng rìa tối, mắt dâu và phản xạ sáng có bị bỏ sót không. |
| **Bộ 2: Ngưỡng vừa (Balanced)** | $R \in [140, 255]$, $G \le 85$, $B \le 85$<br>*(hoặc $D_0 = 65$)* | $H \in [0, 12] \cup [168, 179]$<br>$S \ge 90, V \ge 60$ | Kiểm tra mức độ bao phủ quả dâu và số vùng nền bị chọn nhầm. |
| **Bộ 3: Ngưỡng rộng (Loose)** | $R \in [100, 255]$, $G \le 130$, $B \le 130$<br>*(hoặc $D_0 = 105$)* | $H \in [0, 25] \cup [155, 179]$<br>$S \ge 40, V \ge 30$ | Kiểm tra bóng đổ, cuống nâu và nền có bị chọn nhầm không. |

- **File kết quả xuất ra (`outputs/bai_1/1.3_threshold_survey/`):**
  - `1.3_strawberries_input_vs_output_3_thresholds_rgb.png`
  - `1.3_strawberries_input_vs_output_3_thresholds_hsv.png`
  - `1.3_chalk_input_vs_output_3_thresholds_hsv.png`
  - Bảng thống kê tỷ lệ % pixel thuộc đối tượng (Foreground Ratio %) ứng với từng bộ ngưỡng; tỷ lệ này chỉ phản ánh diện tích mask, không tự chứng minh mask chính xác. Nếu cần so sánh chất lượng, đánh dấu vài vùng tham chiếu hoặc ghi cụ thể các vùng bỏ sót/chọn nhầm trên ảnh.

### Gợi ý chọn minh chứng cho báo cáo PDF
- Với mỗi ảnh, chọn một hình kênh màu có thang đo rõ, một hình so sánh mask RGB–HSV trên cùng nhóm đối tượng và một hình ba bộ ngưỡng. Giữ các lưới đầy đủ trong `outputs/` để đối chiếu khi cần.
- Ghi trong chú thích hình: tên ảnh gốc, không gian màu, thang Hue, các ngưỡng và ý nghĩa màu trắng/đen của mask. Phần nhận xét nêu đúng vùng bỏ sót/chọn nhầm quan sát được thay vì chép sẵn dự đoán trong bảng.
- Đưa đoạn mã chuyển đổi HSI và tạo mask chính vào báo cáo; nộp notebook đầy đủ cùng bộ tham số đã dùng theo yêu cầu tổng quan.
- Đưa ảnh chụp màn hình quá trình chạy/kết quả có thông tin cá nhân vào phần minh chứng theo tài liệu tổng quan; chụp sau khi notebook đã chạy thành công.

---

## 4. Tổng Hợp Cấu Trúc Thư Mục Kết Quả Bài 1
```text
outputs/bai_1/
├── 1.1_color_spaces/
│   ├── 1.1_strawberries_input.png
│   ├── 1.1_strawberries_output_rgb_channels.png
│   ├── 1.1_strawberries_output_hsv_hsi_channels.png
│   ├── 1.1_strawberries_output_cielab_channels.png
│   ├── 1.1_strawberries_input_vs_output_all_spaces.png
│   ├── 1.1_chalk_input.png
│   ├── 1.1_chalk_output_rgb_channels.png
│   ├── 1.1_chalk_output_hsv_hsi_channels.png
│   ├── 1.1_chalk_output_cielab_channels.png
│   └── 1.1_chalk_input_vs_output_all_spaces.png
├── 1.2_segmentation_rgb_vs_hsv/
│   ├── 1.2_strawberries_input.png
│   ├── 1.2_strawberries_input_vs_output_red_group.png
│   ├── 1.2_strawberries_input_vs_output_green_group.png
│   ├── 1.2_chalk_input.png
│   ├── 1.2_chalk_input_vs_output_yellow_group.png
│   └── 1.2_chalk_input_vs_output_blue_group.png
└── 1.3_threshold_survey/
    ├── 1.3_strawberries_input_vs_output_3_thresholds_rgb.png
    ├── 1.3_strawberries_input_vs_output_3_thresholds_hsv.png
    └── 1.3_chalk_input_vs_output_3_thresholds_hsv.png
```

