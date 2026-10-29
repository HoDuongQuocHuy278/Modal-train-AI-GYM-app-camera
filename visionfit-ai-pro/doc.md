# 📘 TÀI LIỆU TOÀN DIỆN KIẾN TRÚC & HỆ THỐNG VISIONFIT AI PRO

Tài liệu này giải thích chi tiết cấu trúc toàn bộ tệp tin trong dự án, cơ sở khoa học của công nghệ Computer Vision nhận dạng khớp cơ thể người, luồng dữ liệu (Data Pipeline Flow) đi qua từng tệp tin, cùng thông số Đầu vào (Input) và Đầu ra (Output) của từng thành phần.

---

## 📑 MỤC LỤC
1. [Sơ đồ Cấu trúc Tệp tin Dự án](#1-sơ-đồ-cấu-trúc-tệp-tin-dự-án)
2. [Cơ sở Khoa học: Vì sao máy tính nhận dạng được các khớp cơ thể người?](#2-cơ-sở-khoa-học-vì-sao-máy-tính-nhận-dạng-được-các-khớp-cơ-thể-người)
3. [Luồng Chạy Chi Tiết Của Toàn Bộ Hệ Thống (Pipeline Flows)](#3-luồng-chạy-chi-tiết-của-toàn-bộ-hệ-thống-pipeline-flows)
4. [Bảng Đặc Tả Chi Tiết: Vai Trò, Đầu Vào (Input) & Đầu Ra (Output) Của Từng Tệp Tin](#4-bảng-đặc-tả-chi-tiết-vai-trò-đầu-vào-input--đầu-ra-output-của-từng-tệp-tin)
5. [Hướng dẫn Khởi chạy & Sử dụng Nhanh](#5-hướng-dẫn-khởi-chạy--sử-dụng-nhanh)

---

## 1. Sơ đồ Cấu trúc Tệp tin Dự án

```
visionfit-ai-pro/
├── web/                           # 🌐 GIAO DIỆN WEB STUDIO (CLIENT-SIDE)
│   ├── index.html                 # Giao diện chính (HTML5 Dashboard, Canvas, HUD)
│   ├── app.js                     # Toàn bộ logic ứng dụng web, Vision, TF.js, Biomechanics
│   └── style.css                  # Thiết kế Cyberpunk Dark Mode, Glassmorphism, Animation
│
├── web_server.py                  # 🔌 Server trung gian (Local Web Server & Bridge API)
├── run_web.bat                    # ⚡ Kịch bản khởi động Web Studio 1-click trên Windows
│
├── visionfit.py                   # 🧠 Mô hình ActionNet PyTorch, Chuẩn hóa, Train & Rep Counter
├── pose_model.py                  # 👁️ Mô hình PoseNet cục bộ (Heatmap Regression 33 điểm)
├── pose_annotator.py              # 🏷️ GUI OpenCV gán nhãn thủ công 33 điểm khớp trên ảnh
├── live.py                        # 🎥 Ứng dụng chạy trực tiếp Webcam/Video bằng OpenCV & PyTorch
├── local_video.py                 # 🎞️ Bộ công cụ đọc video và vẽ khung xương skeleton
├── repetition.py                  # 🔢 Bộ đếm rep mở rộng (PushUpCounter, LungeCounter)
├── metrics.py                     # 📊 Bộ tính toán độ chính xác (Confusion Matrix, F1-Score)
├── build_dataset.py               # 📦 Đóng gói các clip cử động thành file dataset .npz
├── extract_clips.py               # ✂️ Cắt nhỏ video dài thành các clip cử động
├── extract_frames.py              # 🖼️ Trích xuất từng khung hình ảnh từ video
│
├── labels.json                    # 🏷️ Danh sách 4 lớp động tác: squat, push_up, lunge, standing
├── data/                          # 📁 Dữ liệu huấn luyện
│   ├── demo.npz                   # 60 chuỗi cử động mẫu (X: 32 frames x 33 joints x 4, y, subjects)
│   ├── pose_demo.npz              # 20 ảnh mẫu kèm nhãn tọa độ 33 điểm (images, keypoints)
│   └── web_recorded.npz           # Dữ liệu thu thập từ Web chuyển sang cho Python
├── models/                        # 💾 Trọng số mô hình đã huấn luyện
│   └── action_demo.pt             # Mô hình ActionNet PyTorch đã train (F1 = 1.000)
│
├── test_pro.py                    # 🧪 Bộ kiểm thử tự động (Unit Tests) cho hệ thống
└── test_visionfit.py              # 🧪 Bộ kiểm thử độ tương thích và tính toán số học
```

---

## 2. Cơ sở Khoa học: Vì sao máy tính nhận dạng được các khớp cơ thể người?

### 2.1. Bản chất ảnh kỹ thuật số
Máy tính không nhìn thấy "con người" hay "cánh tay", mà chỉ nhìn thấy một **ma trận số 3 chiều** kích thước `(Chiều cao H × Chiều rộng W × 3 kênh màu RGB)`. Mỗi pixel là một số nguyên từ `0` đến `255`.

### 2.2. Kiến trúc Mạng Nơ-ron Tích chập (Convolutional Neural Network - CNN)
Để tìm ra các khớp từ ma trận pixel:
1. **Lớp tích chập cấp thấp (Low-level Convolution)**: Các bộ lọc kích thước nhỏ (3×3) quét qua ảnh để phát hiện các cạnh viền, độ dốc ánh sáng, ranh giới giữa da và quần áo/nền.
2. **Lớp tích chập cấp trung (Mid-level Features)**: Kết hợp các cạnh viền để nhận diện kết cấu hình trụ (bắp tay, đùi, cẳng chân) và hình tròn/oval (đầu, khớp vai, đầu gối).
3. **Lớp tích chập cấp cao (High-level Semantic Features)**: Mô hình hóa cấu trúc giải phẫu người (vai liên kết với khuỷu tay, khuỷu tay liên kết với cổ tay).

### 2.3. Cơ chế Hồi quy Tọa độ (Heatmap Regression & Direct Landmark Prediction)
Trong kiến trúc của **MediaPipe BlazePose** và **PoseNet**:
* Thay vì chỉ đoán trực tiếp số tọa độ `(x, y)`, mạng nơ-ron xuất ra **33 Bản đồ nhiệt (Heatmaps)** có độ phân giải nhỏ hơn ảnh gốc (ví dụ 32×32 pixel).
* Mỗi bản đồ nhiệt đại diện cho xác suất vị trí của **1 khớp duy nhất**:
  * Đỉnh của phân phối xác suất hình chuông (Gaussian Peak) chính là vị trí khớp cần tìm.
  * Tọa độ `(x, y)` được tính bằng toán tử **Soft-Argmax** (lấy kỳ vọng trọng số không gian), cho độ chính xác đến từng phần phân số của pixel (Sub-pixel accuracy).
* Đi kèm tọa độ là chỉ số **Visibility / Confidence (độ tin cậy)** từ `0.0` đến `1.0`: Máy tính tự biết được khớp đó có đang bị che khuất (occluded) hay không.
* **Tọa độ trục Z (Depth)**: Được mô hình ước lượng tương đối so với tâm hông dựa trên tỷ lệ giải phẫu (ví dụ: khi đưa tay ra trước ống kính, kích thước tương đối của bàn tay tăng lên so với vai, từ đó suy ra Z âm).

### 2.4. Cấu trúc 33 Khớp Chuẩn MediaPipe Pose & 21 Khớp MediaPipe Hands
* **MediaPipe Pose (33 điểm - Toàn thân)**:
  * Đầu & Mặt: Mũi (0), Mắt (1-6), Tai (7-8), Miệng (9-10).
  * Chi trên: Vai (11, 12), Khuỷu tay (13, 14), Cổ tay (15, 16), Ngón tay ước lượng (17-22).
  * Thân dưới & Chi dưới: Hông (23, 24), Đầu gối (25, 26), Mắt cá chân (27, 28), Gót chân (29, 30), Mũi bàn chân (31, 32).
* **MediaPipe Hands (21 điểm - Mỗi bàn tay)**:
  * Cổ tay: Điểm 0.
  * Ngón cái: 1 (CMC), 2 (MCP), 3 (IP), 4 (Đầu ngón Tip).
  * Ngón trỏ: 5 (MCP), 6 (PIP), 7 (DIP), 8 (Đầu ngón Tip).
  * Ngón giữa: 9, 10, 11, 12.
  * Ngón áp út: 13, 14, 15, 16.
  * Ngón út: 17, 18, 19, 20.
  * Cầu nối khớp đốt bàn tay: `[5, 9], [9, 13], [13, 17]`.

### 2.5. Cơ chế Bất biến Tọa độ (Hip-Centering & Torso-Scaling Invariance)
Nếu người dùng đứng xa camera thì người nhỏ lại, đứng gần camera thì người to ra; đứng lệch sang trái thì tọa độ `x` giảm đi. Để AI phân loại cử động chuẩn xác mà không bị phụ thuộc vào vị trí đứng:
* **Dịch gốc tọa độ (Translation Invariance)**: Lấy tâm giữa 2 khớp hông `(Hip_Left + Hip_Right) / 2` làm gốc tọa độ `(0, 0, 0)`. Mọi khớp khác đều được trừ đi tọa độ tâm hông.
* **Co giãn tỷ lệ (Scale Invariance)**: Chia khoảng cách cho khoảng cách giữa 2 vai (Shoulder Width) hoặc chiều dài thân mình (Torso Length).
* **Kết quả**: Dù người dùng cao hay thấp, đứng gần 1m hay 3m, dạng chuẩn hóa của động tác squat, push-up hay lunge đưa vào mô hình là **hoàn toàn đồng nhất**.

---

## 3. Luồng Chạy Chi Tiết Của Toàn Bộ Hệ Thống (Pipeline Flows)

```
[Camera Web / File Video] 
       │
       ▼ (Khung hình RGB 640x480 @ 60 FPS)
┌─────────────────────────────────────────────────────────┐
│               WEB BROWSER PIPELINE (web/app.js)         │
│                                                         │
│  1. Vision Engines (MediaPipe Pose & Hands)             │
│     ├── Pose BlazePose Lite (33 landmarks: x, y, z, v)  │
│     └── Hands BlazeHands    (21 landmarks x 2 hands)    │
│                                                         │
│  2. Biomechanical Heuristics & Temporal Latch           │
│     ├── Tính góc gối, khuỷu, góc nghiêng thân mình      │
│     └── Action Hysteresis (giữ nhịp rep, chống flicker) │
│                                                         │
│  3. Neural Network Classifier (TensorFlow.js)           │
│     ├── Vector 142 đặc trưng (132 coords + 10 kinematic)│
│     └── Softmax Probabilities: Squat, Push-up, Lunge... │
│                                                         │
│  4. Rep Counter & Audio Feedback                        │
│     ├── Đếm Rep chu kỳ UP -> DOWN -> UP                 │
│     └── Phát tiếng bip (Phase Down) & chime (Rep Done)  │
│                                                         │
│  5. Canvas Renderer (drawCyberSkeleton)                 │
│     └── Khung xương neon Cyberpunk, ngón tay lục bảo    │
└──────────────────────────┬──────────────────────────────┘
                           │ 
                           ▼ (Nhấn "Chuyển sang Python .NPZ")
┌─────────────────────────────────────────────────────────┐
│              BACKEND BRIDGE (web_server.py)             │
│  Nhận HTTP POST /api/save_npz -> Ghi data/web_recorded.npz
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼ (Lệnh train Python)
┌─────────────────────────────────────────────────────────┐
│             PYTORCH PIPELINE (visionfit.py)             │
│  1. Đọc tệp .npz (X: N, 32, 33, 4; y: N)                │
│  2. normalize() & resample(32)                          │
│  3. ActionNet (1D Temporal CNN: Conv1D -> BatchNorm)    │
│  4. Đánh giá kiểm thử (metrics.py -> F1-score)          │
│  5. Lưu mô hình (models/action_demo.pt)                 │
│  6. Chạy offline trên video/webcam (live.py)            │
└─────────────────────────────────────────────────────────┘
```

### 3.1. Luồng 1: Nhận diện trực tiếp & Đếm Rep thời gian thực (Live Workout Flow)
1. Người dùng mở `http://localhost:8000/index.html` hoặc chạy `run_web.bat`.
2. Trình duyệt tải `index.html`, nạp CSS `style.css` và khởi chạy script `app.js`.
3. Hàm `initVisionEngines()` trong `app.js` khởi tạo song song:
   * **MediaPipe Pose** với `modelComplexity: 0` (chế độ BlazePose Lite siêu tốc 60 FPS).
   * **MediaPipe Hands** theo dõi chi tiết 21 điểm từng khớp ngón tay.
4. Người dùng bấm **"Bật Webcam"**:
   * Hàm `toggleWebcam()` yêu cầu camera với cấu hình mở khóa 60 FPS: `{ width: 640, height: 480, frameRate: { ideal: 60 } }`.
   * Vòng lặp camera `Camera.onFrame()` gửi frame hình ảnh tuần tự đến `pose.send()` và `hands.send()`.
5. Khi có kết quả landmarks:
   * Hàm `onPoseResults()` cập nhật bộ đệm thời gian `state.recentFrames` (32 frames trượt).
   * Hàm `analyzePoseHeuristic(landmarks)` tính toán góc khuỷu tay, góc gối trái/phải, độ nghiêng thân mình relative to vertical.
   * Nếu kích hoạt mô hình AI: Hàm `predictWithTfjs()` trích xuất vector 142 đặc trưng (132 tọa độ chuẩn hóa + 10 đặc trưng động học cử động), mạng nơ-ron TensorFlow.js xuất ra xác suất % của 4 lớp.
   * Hàm `updateRepCounter()` kiểm tra trạng thái chuyển động: Khi người tập đạt góc chuẩn pha `down`, âm thanh trầm phát ra. Khi đẩy người lên hoàn thành rep ở pha `up`, biến đếm rep tăng 1 và âm thanh chuông chúc mừng vang lên.
   * Hàm `drawCyberSkeleton()` vẽ khung xương cơ thể màu cyan, các khớp chính màu xanh ngọc, cổ tay màu hổ phách, và 5 ngón tay với đầu ngón xanh lục bảo. Nếu bàn tay ở xa camera, hệ thống tự động vẽ hand fallback để đảm bảo tay **không bao giờ bị mất**.

### 3.2. Luồng 2: Thu thập mẫu & Đồng bộ hóa Dữ liệu (Data Lab Flow)
1. Người dùng chuyển sang tab **"Thu thập & Gán nhãn"** (`collectTab`).
2. Chọn nhãn động tác (ví dụ: `squat`), mã người tập (`person01`), chế độ `clip32` (32 khung hình).
3. Bấm **"Bắt đầu ghi mẫu"**:
   * Hệ thống đếm ngược 3 giây để người tập vào vị trí.
   * `handleLandmarksBuffer()` bắt đầu đẩy liên tục các khung hình vào `state.recording.buffer`.
   * Thanh tiến trình chạy từ 0% đến 100% (đủ 32 frames).
4. Hàm `finishRecordingClip()` đóng gói mẫu dữ liệu thành một object JSON có ID duy nhất, thời gian, nhãn, và 32 frames tọa độ `(x, y, z, visibility)`.
5. Người dùng có 2 lựa chọn:
   * **Lưu file client**: Bấm "Xuất Dataset (.JSON)" để tải file về máy.
   * **Đồng bộ hóa với Python**: Bấm "Chuyển sang Python .NPZ" -> Trình duyệt gửi HTTP POST tới `web_server.py` qua API `/api/save_npz`. `web_server.py` đọc dữ liệu, chuyển thành mảng n-chiều NumPy và lưu nén thành `data/web_recorded.npz`.

### 3.3. Luồng 3: Huấn luyện Mô hình Trong Trình Duyệt (TensorFlow.js Flow)
1. Trong tab **"Huấn luyện AI"** (`trainTab`), người dùng cấu hình số Epochs (mặc định 30), Batch Size (16), Learning Rate (0.005).
2. Khi bấm **"Bắt đầu Huấn luyện AI"** (hoặc khi trang web tự động chạy `autoTrainOnStartup()`):
   * Hàm `prepareTrainingTensors()` duyệt qua danh sách mẫu trong `state.dataset`.
   * Trích xuất vector 142 đặc trưng cho từng mẫu -> Tạo Tensor đầu vào `xs` kích thước `[N, 142]` và Tensor nhãn one-hot `ys` kích thước `[N, 4]`.
   * Khởi tạo kiến trúc mạng nơ-ron: `Dense(128, ReLU)` -> `Dropout(0.2)` -> `Dense(64, ReLU)` -> `Dense(4, Softmax)`.
   * Tiến hành huấn luyện `model.fit()`: Callback `onEpochEnd` cập nhật độ mất mát (Loss) và độ chính xác (Accuracy) theo thời gian thực lên 2 biểu đồ Chart.js.
3. Khi huấn luyện xong: Mô hình được gán vào `state.tfModel`, sẵn sàng phục vụ kiểm thử nhận diện ngay lập tức mà không cần cài đặt bất kỳ thư viện Python nào.

### 3.4. Luồng 4: Huấn luyện & Chạy Mô hình PyTorch Cục Bộ (Python Backend Flow)
1. Từ Terminal, chạy lệnh:
   ```bash
   python visionfit.py train --data data/demo.npz --labels labels.json --out models/action_demo.pt --epochs 20
   ```
2. Tệp `visionfit.py`:
   * Đọc tệp nén `data/demo.npz` qua hàm `load_data()`.
   * Chuẩn hóa tọa độ bất biến `normalize()` và nội suy độ dài thời gian chuẩn 32 frames `resample()`.
   * Chia tập dữ liệu thành Train / Validation / Test theo nguyên tắc tách biệt người tập (`split_train_val_test`).
   * Khởi tạo mạng **ActionNet** (Mạng nơ-ron tích chập 1 chiều theo thời gian - 1D Temporal CNN):
     * Input `(Batch, 132 features, 32 time steps)`.
     * Conv1D(132 -> 128, kernel=5) + BatchNorm1D + ReLU.
     * Conv1D(128 -> 128, kernel=5) + BatchNorm1D + ReLU + MaxPool1D.
     * Conv1D(128 -> 256, kernel=3) + BatchNorm1D + ReLU + AdaptiveAvgPool1D.
     * Dropout(0.25) + Linear(256 -> 4 classes).
   * Tối ưu hóa bằng AdamW, lưu trọng số tốt nhất vào `models/action_demo.pt`.
   * Tệp `metrics.py` xuất báo cáo đánh giá ma trận nhầm lẫn (Confusion Matrix) và chỉ số Macro F1-score vào `reports/evaluation.json`.
3. Khi chạy trực tiếp qua `live.py`:
   * OpenCV chụp khung hình từ webcam.
   * `local_video.py` đưa frame qua `pose_model.py` để lấy 33 điểm khớp.
   * `ActionSmoother` trong `live.py` lọc nhiễu xác suất qua cửa sổ trượt.
   * Bộ đếm rep trong `repetition.py` đếm số lần hoàn thành và OpenCV vẽ khung xương trực tiếp lên cửa sổ màn hình.

---

## 4. Bảng Đặc Tả Chi Tiết: Vai Trò, Đầu Vào (Input) & Đầu Ra (Output) Của Từng Tệp Tin

| Tên Tệp Tin | Vai Trò & Trách Nhiệm | Đầu Vào (Input) | Đầu Ra (Output) |
| :--- | :--- | :--- | :--- |
| **`web/index.html`** | Giao diện đồ họa người dùng (GUI Dashboard), chứa khung nhìn camera, HUD hiển thị số rep, tab thu thập dữ liệu, biểu đồ huấn luyện và bảng quản lý dataset. | Thao tác click chuột, chọn dropdown, luồng video camera từ trình duyệt. | Cây phân cấp DOM, các thẻ `<canvas>` và `<video>` để JavaScript tương tác. |
| **`web/app.js`** | Não bộ điều khiển toàn bộ ứng dụng web: Tích hợp MediaPipe Pose & Hands, tính toán góc sinh cơ học, trích xuất 142 đặc trưng, huấn luyện & suy luận TensorFlow.js, đếm rep, phát âm thanh Web Audio API. | Luồng video từ thẻ `<video>`, sự kiện người dùng, dữ liệu landmarks từ MediaPipe. | Khung xương skeleton render trên `<canvas>`, số Rep trên HUD, âm thanh phản hồi, mô hình TensorFlow.js đã train. |
| **`web/style.css`** | Định nghĩa toàn bộ hệ thống phong cách giao diện: Màu sắc Cyberpunk neon, hiệu ứng kính mờ Glassmorphism, lưới Grid responsive, thiết kế thanh tiến trình và bảng dữ liệu. | Các class và ID trong HTML DOM. | Giao diện hiển thị trực quan đẹp mắt, tối ưu hiệu năng render GPU. |
| **`web_server.py`** | Máy chủ HTTP nội bộ không phụ thuộc framework ngoài: Phục vụ các file tĩnh trong `web/`, cung cấp API `/api/status` và `/api/save_npz` để chuyển dữ liệu từ trình duyệt thành tệp `.npz`. Tự động chuyển cổng sang 8080 nếu 8000 bị bận. | Yêu cầu HTTP GET/POST từ trình duyệt, payload JSON chứa danh sách mẫu cử động. | Phản hồi JSON trạng thái hệ thống, tạo ra tệp nén dữ liệu `data/web_recorded.npz`. |
| **`run_web.bat`** | Kịch bản thực thi tự động (Batch script) trên Windows: Tự động phát hiện phiên bản Python phù hợp (kiểm tra thư viện NumPy) và khởi động `web_server.py`. | Lệnh kích hoạt nhấp đúp chuột của người dùng. | Cửa sổ dòng lệnh chạy server và tự động mở trình duyệt web đến URL của ứng dụng. |
| **`visionfit.py`** | Bộ mã nguồn phân loại hành động cốt lõi PyTorch: Định nghĩa mạng 1D Temporal CNN `ActionNet`, các hàm chuẩn hóa tọa độ `normalize()`, nội suy thời gian `resample()`, logic huấn luyện `train`, đánh giá `eval` và lớp `SquatCounter`. | File dataset `.npz` chứa chuỗi cử động, file cấu hình nhãn `labels.json`. | Tệp trọng số mô hình đã huấn luyện `.pt`, file báo cáo độ chính xác `reports/evaluation.json`. |
| **`pose_model.py`** | Mô hình trích xuất tư thế người cục bộ bằng PyTorch: Định nghĩa mạng `PoseNet` sử dụng phương pháp hồi quy bản đồ nhiệt (Heatmap Regression) cho 33 điểm khớp, hàm giải mã `decode_pose()` và hàm suy luận ảnh RGB `infer_rgb()`. | Mảng tensor hình ảnh kích thước `(Batch, 3, 128, 128)` pixel. | Tensor bản đồ nhiệt `(Batch, 33, 32, 32)` và tọa độ ước lượng `(Batch, 33, 4)`: x, y, z, visibility. |
| **`pose_annotator.py`** | Ứng dụng OpenCV gán nhãn thủ công ảnh ngoại tuyến: Mở từng ảnh trong thư mục, cho phép người dùng click lần lượt 33 điểm khớp theo thứ tự chuẩn MediaPipe Pose. | Thư mục chứa các tệp ảnh `.jpg` / `.png`. | Tệp dữ liệu nén `.npz` chứa mảng ảnh và mảng tọa độ điểm gán nhãn tương ứng. |
| **`live.py`** | Ứng dụng nhận diện trực tiếp bằng Python thuần qua OpenCV: Đọc webcam hoặc tệp video cục bộ, kết nối mô hình PoseNet với ActionNet, làm mượt xác suất qua `ActionSmoother` và đếm rep. | Camera ID (ví dụ: `0`) hoặc đường dẫn tệp video `.mp4`, mô hình `.pt`. | Cửa sổ hiển thị video kèm khung xương vẽ trực tiếp, nhãn động tác, độ tin cậy và số rep. |
| **`local_video.py`** | Thư viện hỗ trợ dùng chung cho OpenCV: Mở video/webcam cục bộ, trích xuất landmarks từ ảnh BGR, kiểm tra người có trong khung hình `body_visible()`, vẽ đường nối skeleton `draw_skeleton()`. | Frame hình ảnh OpenCV (BGR numpy array), mô hình trích xuất pose. | Mảng tọa độ landmarks `(33, 4)` và khung hình ảnh đã vẽ khung xương trực quan. |
| **`repetition.py`** | Định nghĩa các bộ đếm lặp lại sinh cơ học: Lớp `PushUpCounter` (theo dõi độ gập khuỷu tay và tư thế nằm ngang) và `LungeCounter` (theo dõi độ bất đối xứng và góc gập gối trước). | Chuỗi landmarks `(33, 4)` của từng khung hình thời gian thực. | Trạng thái chuyển động (`up`/`down`), góc khớp hiện tại và tổng số rep hoàn thành. |
| **`metrics.py`** | Mô-đun tính toán số liệu thống kê máy học: Tính ma trận nhầm lẫn (Confusion Matrix), độ chính xác tổng thể (Accuracy), Precision, Recall và điểm F1 vĩ mô (Macro F1-Score). | Mảng nhãn thực tế `y_true`, mảng nhãn dự đoán `y_pred` và danh sách tên các lớp. | Đối tượng từ điển (Dictionary) chứa chi tiết các chỉ số đánh giá mô hình. |
| **`build_dataset.py`** | Công cụ tiền xử lý dữ liệu: Quét các thư mục chứa clip cử động định dạng `.npy`, kiểm tra độ tin cậy của khớp, nội suy về 32 khung hình và đóng gói thành tệp nén `.npz`. | Thư mục chứa các clip cử động dạng `.npy` được phân loại theo nhãn lớp. | Tệp dữ liệu hoàn chỉnh `.npz` gồm 3 trường: `X` (đặc trưng), `y` (nhãn số), `subjects` (người tập). |
| **`extract_clips.py`** | Công cụ tiện ích xử lý video: Cắt nhỏ các đoạn video tập luyện dài thành các video clip ngắn theo từng rep hoặc từng hiệp dựa trên mốc thời gian. | Tệp video gốc dài và danh sách khoảng thời gian (Start time, End time). | Danh sách các file video clip ngắn đã được phân tách. |
| **`extract_frames.py`** | Công cụ trích xuất ảnh tĩnh: Trích xuất các khung hình riêng rẽ từ video với bước nhảy khung hình xác định để phục vụ việc gán nhãn tọa độ khớp. | Tệp video đầu vào và tần suất lấy mẫu khung hình (Step interval). | Thư mục chứa các tệp hình ảnh định dạng `.jpg`. |
| **`test_pro.py`** | Bộ kiểm thử tích hợp chuyên sâu (Pytest): Kiểm tra suy luận PoseNet, phân tách tập dữ liệu không rò rỉ người tập, tính toán metrics, làm mượt cử động và đếm rep. | Dữ liệu giả lập tổng hợp (Synthetic test fixtures). | Kết quả kiểm thử thành công (PASS/FAIL) trên terminal. |
| **`test_visionfit.py`** | Bộ kiểm thử chức năng cơ bản: Kiểm tra tính toán chuẩn hóa tọa độ `normalize()`, nội suy thời gian `resample()` và khởi tạo mô hình ActionNet. | Dữ liệu ma trận số học giả lập. | Xác nhận các phép toán đại số tuyến tính hoạt động chính xác. |
| **`labels.json`** | Tệp định nghĩa danh mục lớp: Lưu mảng JSON thứ tự các nhãn cử động được hỗ trợ trong toàn bộ dự án: `["squat", "push_up", "lunge", "standing"]`. | Được đọc bởi server web, mô hình PyTorch và script huấn luyện. | Ánh xạ đồng nhất giữa chỉ số số nguyên `(0, 1, 2, 3)` và tên động tác dạng chữ. |

---

## 5. Hướng dẫn Khởi chạy & Sử dụng Nhanh

### 5.1. Khởi chạy Giao diện Web Studio (Khuyên dùng)
* **Cách 1**: Nhấp đúp vào tệp [`run_web.bat`](file:///e:/Downloads/VisionFit_AI_Pro_ATTACH/visionfit-ai-pro/run_web.bat) trên Windows.
* **Cách 2**: Chạy từ terminal:
  ```bash
  python web_server.py --port 8000
  ```
* Mở trình duyệt tại địa chỉ: `http://localhost:8000/index.html`.
* Ngay khi mở lên, hệ thống sẽ:
  1. Tự động tải 128 mẫu dữ liệu giả lập chuẩn của cả 4 động tác vào tab Data Lab.
  2. Tự động huấn luyện mô hình TensorFlow.js Neural Network ngay trong trình duyệt.
  3. Kích hoạt chế độ **⚡ 60 FPS Turbo** siêu mượt.
  4. Nhấn **"Bật Webcam"** để bắt đầu tập luyện và đếm rep tự động!

### 5.2. Chạy Huấn luyện PyTorch Offline
* Huấn luyện mô hình mạng nơ-ron ActionNet với tệp dữ liệu có sẵn:
  ```bash
  python visionfit.py train --data data/demo.npz --labels labels.json --out models/action_demo.pt --epochs 20
  ```
* Chạy kiểm thử tự động toàn bộ dự án:
  ```bash
  python -m pytest test_pro.py test_visionfit.py
  ```
