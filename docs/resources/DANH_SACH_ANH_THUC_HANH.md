# Kế Hoạch & Danh Sách Ảnh Thực Nghiệm Chuẩn Cho 4 Bài Thực Hành DIP

Bộ dữ liệu được tuyển chọn trực tiếp từ các thư mục ảnh chuẩn của giáo trình **"Digital Image Processing, 3rd / Global Edition - Rafael C. Gonzalez & Richard E. Woods"** trong thư mục `data/`.

---

## 1. Bài 1: Phân Tích và Phân Đoạn Ảnh Màu

### Yêu cầu đề bài:
- Chọn ít nhất 2 ảnh màu; chuyển đổi RGB sang HSV/HSI và CIELAB, hiển thị các kênh màu.
- Phân đoạn ít nhất 2 nhóm đối tượng theo màu; so sánh kết quả trong RGB và HSV/HSI.
- Khảo sát ít nhất 3 bộ tham số ngưỡng, nhận xét kết quả.

> **Thư mục dữ liệu nguồn:** `data/DIP3E_Original_Images_CH06/`

### Danh mục ảnh lựa chọn:

| STT | Tên file ảnh | Đặc tính ảnh | Mục đích & Nội dung thực nghiệm |
| :-: | :--- | :--- | :--- |
| **1** | `Fig0630(01)`<br>`(strawberries_fullcolor).tif` | **RGB**<br>`666 × 666` | **Ảnh chính 1**: Quả dâu tây đỏ tươi trên nền lá xanh.<br>• *Nhóm 1*: Quả dâu tây (Màu đỏ - Red).<br>• *Nhóm 2*: Cuống và lá (Màu xanh lá - Green).<br>• *So sánh*: Trong RGB, bóng râm và phản xạ ánh sáng làm thủng mask; trong HSV/HSI, kênh Hue tách biệt trọn vẹn cả quả dù có bóng đổ. |
| **2** | `Fig0635`<br>`(middle_row_left_chalk ).tif` | **RGB**<br>`1040 × 1040` | **Ảnh chính 2**: Hộp phấn nhiều màu sắc.<br>• *Nhóm 1*: Các viên phấn màu vàng (Yellow).<br>• *Nhóm 2*: Các viên phấn màu xanh dương (Blue).<br>• Đánh giá phân đoạn nhiều dải màu riêng biệt với 3 bộ ngưỡng (hẹp, chuẩn, rộng). |
| **3** | `Fig0638(a)`<br>`(lenna_RGB).tif` | **RGB**<br>`512 × 512` | **Ảnh bổ trợ/dự phòng**: Chân dung kinh điển.<br>• Tách vùng màu da người (Skin tone) và mũ (Magenta/Violet). |

---

## 2. Bài 2: Xử Lý Ảnh Hình Thái Học (Morphological Processing)

### Yêu cầu đề bài:
- Tự lập trình Erosion, Dilation, Opening, Closing bằng NumPy; so sánh đối chiếu với OpenCV.
- Thử nghiệm các phần tử cấu trúc (Structuring Elements) có hình dạng và kích thước khác nhau.
- Thực hiện ít nhất 2 thuật toán: trích xuất hình bao, làm đầy lỗ, thành phần liên thông, convex hull, thinning hoặc thickening.

> **Thư mục dữ liệu nguồn:** `data/DIP3E_Original_Images_CH09/`

### Danh mục ảnh lựa chọn:

| STT | Tên file ảnh | Đặc tính ảnh | Mục đích & Nội dung thực nghiệm |
| :-: | :--- | :--- | :--- |
| **1** | `Fig0905(a)`<br>`(wirebond-mask).tif` | **Nhị phân** (Mode 1)<br>`486 × 486` | **Erosion & Dilation cơ bản**:<br>• Mặt nạ mạch bán dẫn nối dây.<br>• Khảo sát SE hình vuông, tròn, chữ thập với size $3\times3, 5\times5, 11\times11, 15\times15$.<br>• Đo sai số pixel tuyệt đối và tốc độ chạy giữa NumPy tự viết và OpenCV. |
| **2** | `Fig0907(a)`<br>`(text_gaps_1_and_2_pixels).tif` | **Nhị phân** (Mode 1)<br>`508 × 444` | **Dilation & Closing hàn gắn đứt gãy**:<br>• Văn bản có nét chữ bị đứt quãng 1 đến 2 pixel.<br>• Minh chứng Dilation/Closing nối liền nét đứt gãy mà vẫn bảo toàn cấu trúc nét chữ. |
| **3** | `Fig0911(a)`<br>`(noisy_fingerprint).tif` | **Nhị phân** (Mode 1)<br>`315 × 238` | **Opening & Closing khử nhiễu kết hợp**:<br>• Dấu vân tay có đốm nhiễu trắng ngoài nền và lỗ thủng đen trong đường vân.<br>• Minh chứng Opening loại bỏ nhiễu hạt nền, Closing lấp kín lỗ thủng trên vân. |
| **4** | `Fig0914(a)`<br>`(licoln from penny).tif` | **Nhị phân** (Mode 1)<br>`221 × 269` | **Thuật toán nâng cao 1 — Trích xuất đường biên (Boundary Extraction)**:<br>• Hình chân dung Lincoln trên đồng xu.<br>• Áp dụng công thức $\beta(A) = A - (A \ominus B)$ thu được đường viền mảnh 1 pixel sắc nét. |
| **5** | `Fig0916(a)`<br>`(region-filling-reflections).tif` | **Ảnh xám / Nhị phân**<br>`512 × 512` | **Thuật toán nâng cao 2 — Làm đầy lỗ (Hole / Region Filling)**:<br>• Các hình cầu có lỗ rỗng do phản xạ ánh sáng ở tâm.<br>• Áp dụng thuật toán lặp $X_k = (X_{k-1} \oplus B) \cap A^c$ để lấp kín toàn bộ các lỗ rỗng bên trong. |
| **6** | `Fig0918(a)`<br>`(Chickenfilet with bones).tif` | **Ảnh xám** (Mode L)<br>`712 × 321` | **Thuật toán nâng cao 3 — Thành phần liên thông (Connected Components)**:<br>• Ảnh X-ray miếng thịt có chứa các mảnh xương rời rạc.<br>• Phân đoạn, gán nhãn màu riêng cho từng mảnh xương và lập bảng thống kê diện tích. |

---

## 3. Bài 3: Phát Hiện Biên và Phân Ngưỡng

### Yêu cầu đề bài:
- Thực hiện Sobel, Laplacian, Canny; khảo sát ảnh hưởng của các tham số.
- Phân ngưỡng bằng ngưỡng cố định, Otsu và Adaptive Thresholding; tự lập trình Otsu và đối chiếu kết quả thư viện.
- Khảo sát ảnh hưởng của nhiễu đến kết quả phát hiện biên và phân ngưỡng.

> **Thư mục dữ liệu nguồn:** `data/DIP3E_Original_Images_CH10/`

### Danh mục ảnh lựa chọn:

| STT | Tên file ảnh | Đặc tính ảnh | Mục đích & Nội dung thực nghiệm |
| :-: | :--- | :--- | :--- |
| **1** | `Fig1016(a)`<br>`(building_original).tif` | **Ảnh xám** (Mode L)<br>`1114 × 834` | **Phát hiện biên (Sobel, Laplacian, Canny)**:<br>• Tòa nhà có cấu trúc gạch, cửa sổ và cạnh viền kiến trúc phong phú.<br>• Khảo sát `ksize` của Sobel/Laplacian, các cặp ngưỡng $(T_{\text{low}}, T_{\text{high}})$ của Canny. |
| **2** | `Fig1039(a)`<br>`(polymersomes).tif` | **Ảnh xám** (Mode L)<br>`702 × 648` | **Kiểm chứng thuật toán Otsu tự lập trình**:<br>• Các hạt polymersomes hình cầu trên nền sạch, histogram 2 đỉnh (bimodal) rõ nét.<br>• Vẽ Histogram, ngưỡng $T^*$ và đường cong phương sai $\sigma_B^2(k)$. Đối chiếu `my_otsu` với `cv2.THRESH_OTSU`. |
| **3** | `Fig1049(a)`<br>`(spot_shaded_text_image).tif` | **Ảnh xám** (Mode L)<br>`772 × 686` | **So sánh Fixed vs Otsu vs Adaptive Thresholding**:<br>• Văn bản bị bóng chiếu sáng dạng đốm (spot shading).<br>• Minh chứng sự thất bại của ngưỡng toàn cục (Fixed, Otsu) và sự vượt trội của Adaptive Thresholding. |
| **4** | `Fig1036(a)` *(gốc)*<br>`Fig1036(b)` *($\sigma=10$)*<br>`Fig1036(c)` *($\sigma=50$)* | **Ảnh xám** (Mode L)<br>`814 × 651`<br>*(Bộ 3 ảnh)* | **Khảo sát ảnh hưởng của Nhiễu**:<br>• Bộ đa giác 7 cạnh: Ảnh gốc, nhiễu $\sigma=10$, nhiễu nặng $\sigma=50$.<br>• *Biên*: Laplacian vỡ nát hoàn toàn $\rightarrow$ Sobel sinh biên giả $\rightarrow$ Canny bền vững nhất.<br>• *Ngưỡng*: Nhiễu làm bẹt 2 đỉnh histogram khiến Otsu sai lệch; khắc phục bằng lọc Gaussian trước khi tìm ngưỡng. |

---

## 4. Bài 4: Phân Đoạn Ảnh Dựa Trên Vùng (Region-based Segmentation)

### Yêu cầu đề bài:
- Tự lập trình Region Growing; khảo sát vị trí seed và ngưỡng đồng nhất.
- Phân đoạn bằng K-means ($K = 2, 3, 4, 5$) và SLIC Superpixels với các tham số khác nhau.
- Thực hiện thêm một phương pháp: graph-based segmentation, graph cuts hoặc region splitting and merging; so sánh kết quả.

> **Thư mục dữ liệu nguồn:** `data/DIP3E_Original_Images_CH10/` & `data/DIP3E_Original_Images_CH06/`

### Danh mục ảnh lựa chọn:

| STT | Tên file ảnh & Nguồn | Đặc tính ảnh | Mục đích & Nội dung thực nghiệm |
| :-: | :--- | :--- | :--- |
| **1** | `[CH10]`<br>`Fig1051(a)`<br>`(defective_weld).tif` | **Ảnh xám** (Mode L)<br>`566 × 482` | **Region Growing tự lập trình (BFS)**:<br>• Vết nứt khuyết tật mối hàn kéo dài ở trung tâm (ảnh chuẩn Gonzalez Chương 10).<br>• Khảo sát vị trí seed (tâm vết nứt, vùng biên, vùng nền) và ngưỡng đồng nhất $T \in \{10, 25, 45, 70\}$. |
| **2** | `[CH06]`<br>`Fig0630(01)`<br>`(strawberries_fullcolor).tif` | **RGB**<br>`666 × 666` | **K-Means Clustering ($K=2, 3, 4, 5$)**:<br>• Phân cụm điểm ảnh trong không gian màu RGB và CIELAB.<br>• Quan sát sự phân tách đối tượng theo số lượng cụm $K$ tăng dần. |
| **3** | `[CH06]`<br>`Fig0638(a)`<br>`(lenna_RGB).tif` | **RGB**<br>`512 × 512` | **SLIC Superpixels**:<br>• Khảo sát số lượng siêu điểm ảnh `n_segments` $\in \{50, 200, 500\}$ và độ đặc `compactness` $\in \{1, 10, 40\}$.<br>• Đánh giá khả năng bám biên vật thể và hình dạng các ô superpixels. |
| **4** | `[CH10]`<br>`Fig1045(a)`<br>`(iceberg).tif` | **Ảnh xám** (Mode L)<br>`566 × 533` | **Phương pháp nâng cao: Graph-based Felzenszwalb & Quadtree Split/Merge**:<br>• Khối băng trôi trên mặt biển với các vùng cường độ sáng khác biệt.<br>• Lập bảng đối sánh kết quả giữa Region Growing, K-Means, SLIC và Graph-based. |

---

## 5. Bảng Tổng Hợp Số Lượng Ảnh Cho Cả 4 Bài

| Bài thực hành | Số lượng ảnh tối thiểu | Số lượng đã chọn | Danh sách tệp được chọn |
| :--- | :---: | :---: | :--- |
| **Bài 1: Phân đoạn màu** | Ít nhất 2 ảnh màu | **3 ảnh** | `Fig0630(01)(strawberries_fullcolor).tif`<br>`Fig0635(middle_row_left_chalk ).tif`<br>`Fig0638(a)(lenna_RGB).tif` |
| **Bài 2: Hình thái học** | Đủ cho 4 phép cơ bản + 2 thuật toán | **6 ảnh** | `Fig0905(a)(wirebond-mask).tif`<br>`Fig0907(a)(text_gaps_1_and_2_pixels).tif`<br>`Fig0911(a)(noisy_fingerprint).tif`<br>`Fig0914(a)(licoln from penny).tif`<br>`Fig0916(a)(region-filling-reflections).tif`<br>`Fig0918(a)(Chickenfilet with bones).tif` |
| **Bài 3: Biên & Phân ngưỡng** | Đủ cho biên, ngưỡng & nhiễu | **6 ảnh** | `Fig1016(a)(building_original).tif`<br>`Fig1039(a)(polymersomes).tif`<br>`Fig1049(a)(spot_shaded_text_image).tif`<br>`Fig1036(a)(original_septagon).tif`<br>`Fig1036(b)(...std_10...).tif`<br>`Fig1036(c)(...std_50...).tif` |
| **Bài 4: Phân đoạn vùng** | Đủ cho Region growing, K-means, SLIC, Nâng cao | **4 ảnh** | `Fig1051(a)(defective_weld).tif`<br>`Fig0630(01)(strawberries_fullcolor).tif`<br>`Fig0638(a)(lenna_RGB).tif`<br>`Fig1045(a)(iceberg).tif` |
| **TỔNG CỘNG** | — | **19 lượt ảnh<br>(15 file duy nhất)** | Đảm bảo tính bao phủ 100% mục tiêu của đề bài. |
