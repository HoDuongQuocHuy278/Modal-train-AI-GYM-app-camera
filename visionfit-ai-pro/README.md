# VisionFit AI Pro — tự huấn luyện từ ảnh đến động tác gym (offline)

**Phiên bản:** bộ code nghiên cứu/MVP, chạy trên máy của bạn, không gọi HTTP/REST API, không dùng cloud inference và không tự tải pretrained weights. Hai mạng PyTorch đều được định nghĩa trong source và có thể **huấn luyện từ đầu**. `numpy`, `torch` và `opencv-python` là thư viện được cài vào máy (không phải API gọi server); nếu cần cài đặt *ngay cả khi không có Internet*, tải trước các Python wheels phù hợp và dùng `pip install --no-index --find-links ./wheels -r requirements-camera.txt`.

**Phân biệt hai mạng:**

```text
Ảnh/video RGB (người thật)
       │
       ▼
PoseNet của VisionFit (pose_model.py)
  ảnh 128×128 → 33 điểm (x,y,z=0,visibility)
       │      KHÔNG CÓ pretrained weights cho người thật trong ZIP
       ▼
32 frame liên tiếp, resample + hip-centered normalization
       │
       ▼
ActionNet của VisionFit (visionfit.py) / Temporal CNN
       │
       ├── squat / push_up / lunge / standing / unknown (ngưỡng)
       └── bộ đếm rep cơ bản + lọc nhãn dao động
```

**Điều kiện tiên quyết:** hệ thống webcam **chưa thể nhận diện người thật ngay sau giải nén**. Cần gán nhãn ảnh thật và train `models/pose.pt`, sau đó dùng đúng PoseNet ấy để lấy 33 điểm từ video thật, gán nhãn động tác và train `models/action.pt`. `models/action_demo.pt`, `data/demo.npz`, `data/example.npy` là **tài nguyên tổng hợp để kiểm thử chương trình**, không phải mô hình/dữ liệu gym đáng tin cậy. Không có file `models/pose.pt` dựng sẵn. Không có API, nhưng cũng không có phép màu suy khớp từ video nếu chưa có mô hình học được.

## 1. Cài trên Windows / macOS / Linux

Yêu cầu Python 3.10+, tương thích PyTorch của hệ điều hành. Trong thư mục đã giải nén:

```bash
python -m venv .venv
# Windows CMD: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-camera.txt
# Nếu chạy kiểm thử: python -m pip install -r requirements-dev.txt
```

Windows có thể nhấp đôi `setup_windows.bat` để tạo môi trường; hoặc dùng lệnh bên trên. `opencv-python` chỉ cần cho video, webcam và ứng dụng gán nhãn. Phần train action/pose từ `.npz` chỉ cần `requirements.txt`. PyTorch CPU chạy được, CUDA tùy cấu hình phần cứng.

## 2. Xem thử mô hình action cũ ngay (KHÔNG PHẢI nhận diện người thật)

```bash
python visionfit.py predict --model models/action_demo.pt --input data/example.npy --threshold 0
python visionfit.py demo --out data/demo.npz
python visionfit.py train --data data/demo.npz --labels labels.json --out models/action_demo_new.pt --epochs 3 --report-dir reports/demo --augment --class-weights
```

Mặc định predict yêu cầu độ tin cậy softmax 0.65, nếu thấp hơn sẽ trả `unknown`; `--threshold 0` chỉ dùng để kiểm thử đường đi của code. Đọc `reports/demo/evaluation.json` (test/validation, confusion matrix và điểm theo lớp), `reports/demo/history.csv`. Các con số trên **dữ liệu giả không thể dùng làm số liệu đánh giá sản phẩm**.

## 3. Làm dữ liệu ảnh thật cho pose (chính mô hình 33 điểm của bạn)

Quay video **có sự đồng ý** của người tham gia và lưu trên máy. Tách ảnh để gán nhãn:

```bash
python extract_frames.py --video videos/person01_squat.mp4 --out data/frames/person01 --every 12 --limit 150
```

Gán nhãn ảnh bằng GUI OpenCV, **33 điểm theo thứ tự MediaPipe Pose landmark indices** (chỉ mượn QUY ƯỚC CHỈ SỐ, không dùng API/model MediaPipe):

```bash
python pose_annotator.py --images data/frames/person01 --subject person01 --out data/pose_real.npz
python pose_annotator.py --images data/frames/person02 --subject person02 --out data/pose_real.npz --append
python pose_annotator.py --images data/frames/person03 --subject person03 --out data/pose_real.npz --append
```

Trong cửa sổ gán nhãn: nhấp chuột để chọn điểm hiện tại; `X` = điểm không nhìn thấy; `U` = hoàn tác; gán đủ 33 điểm rồi `S` để lưu ảnh; `Q` để dừng và lưu những ảnh đã hoàn thành. Dòng trên đầu hiện tên điểm và chỉ số. Ảnh resize **128×128**, tọa độ x/y theo đúng ảnh đã resize. `--append` để không xóa các ảnh đã gán nhãn từ người khác; mặc định từ chối ghi đè file tồn tại.

Định dạng `.npz`:

- `images`: `uint8 (N,128,128,3)` theo RGB.
- `keypoints`: `float32 (N,33,3)`: x, y chuẩn hóa 0..1, visible 0 hoặc 1.
- `subjects`: `(N,)` ID ẩn danh để chia train/validation khác người; thu ít nhất 3 người, tốt nhất nhiều hơn.

Huấn luyện **pose từ đầu**:

```bash
python pose_model.py train --data data/pose_real.npz --out models/pose.pt --epochs 50 --batch 8 --device auto
```

Mô hình này đơn giản, 2D, ảnh nhỏ, **không học chiều sâu 3D**. Nó xuất `z=0` cho cả 33 điểm. Không trộn dữ liệu `(x,y,z)` từ pose extractor khác với `(x,y,0)` của PoseNet này khi train action. Cần huấn luyện bằng nhiều ảnh thực, phong phú tư thế, khung hình và đánh giá chất lượng khớp bằng nhãn của người chưa gặp; loss thấp không tự chứng minh accuracy cao. Nếu dùng camera 16:9, quy trình hiện tại resize vuông nên có méo tỷ lệ/góc; cải tiến bằng letterbox nhất quán trước khi đánh giá form.

**Chạy thử luồng train pose (KHÔNG dùng được cho người):**

```bash
python pose_model.py demo --out data/pose_demo.npz
python pose_model.py train --data data/pose_demo.npz --out models/pose_toy.pt --epochs 1
```

Bộ `pose_demo.npz` chỉ gồm chấm màu ở tọa độ ngẫu nhiên, không chứa hình người.

## 4. Tạo dữ liệu động tác từ video thật

Sau khi đã train PoseNet từ ảnh thật, quay đủ động tác cho **ít nhất 5 người ẩn danh khác nhau**, đa dạng góc máy/ánh sáng/tốc độ. Đặt file video trong máy. Ví dụ:

```bash
python extract_clips.py --source videos/person01_squat.mp4 --pose-model models/pose.pt --label squat --subject person01
python extract_clips.py --source videos/person01_pushup.mp4 --pose-model models/pose.pt --label push_up --subject person01
python extract_clips.py --source videos/person01_lunge.mp4 --pose-model models/pose.pt --label lunge --subject person01
python extract_clips.py --source videos/person01_rest.mp4 --pose-model models/pose.pt --label standing --subject person01
```

Lặp lại với `person02`, `person03`, `person04`, `person05` (và tốt hơn: nhiều người hơn). Công cụ xử lý **video local hoặc camera index `0`**, cấm URL/network source. Kết quả:

```text
data/action_clips/
  squat/person01/person01_squat_00000032.npy
  squat/person02/...npy
  push_up/person01/...npy
  lunge/person01/...npy
  standing/person01/...npy
```

Tên video có khoảng trắng/dấu sẽ được chuẩn hóa thành tên file clip an toàn; nhãn và subject ID chỉ dùng chữ/số/gạch dưới/gạch ngang. Tool dùng cửa sổ 32 frame, trượt 16 frame, bỏ/reset cửa sổ khi không thấy thân người. Tham số `--stride 2` lấy một frame mỗi hai frame; `--frames 32 --hop 16 --min-body-visibility 0.4` tùy chỉnh. Các clip chồng lấn của cùng một người sẽ ở cùng partition (không rò rỉ train/test qua clip).

Nếu đã có bộ 33 landmark được trích offline từ hệ thống riêng, không cần PoseNet: chỉ việc lưu file `(T,33,4)` `.npy` theo đúng cấu trúc thư mục trên và **giữ nhất quán hệ tọa độ**.

Đóng gói dataset:

```bash
python build_dataset.py --root data/action_clips --labels labels.json --out data/action_real.npz
```

`data/action_real.npz` có `X: (N,32,33,4)`, `y: (N,)`, `subjects: (N,)`. Label mặc định `labels.json`:

```json
["squat", "push_up", "lunge", "standing"]
```

Clip cần chứa nhãn đúng; không lấy động tác chuyển pha giữa squat và đứng rồi gán cả clip là squat nếu nhãn không phản ánh đúng. Có thể thêm lớp động tác vào `labels.json` nếu thu đủ dữ liệu cho tất cả các lớp. `standing` không đủ bao phủ toàn bộ hoạt động khác: trong thực tế nên thu thêm lớp `other`/`transition` và dữ liệu phủ định nếu cần.

## 5. Train action thật và xem báo cáo

```bash
python visionfit.py train --data data/action_real.npz --labels labels.json --out models/action.pt --epochs 80 --batch 32 --patience 10 --augment --class-weights --report-dir reports/real
python visionfit.py predict --model models/action.pt --input data/action_clips/squat/person01/person01_squat_00000032.npy
```

Train/validation/test mặc định chia theo **người** (60/20/20 xấp xỉ), khi có `subjects`; train thiếu class báo lỗi. Điều chỉnh `--val-fraction`, `--test-fraction`, `--seed`, `--device`. Chỉ dùng validation chọn checkpoint; file JSON báo test sau khi đã chọn mô hình, có confusion matrix, precision/recall/F1 theo lớp. Không nên tinh chỉnh hyperparameter liên tục trên test set; cần test set mới nếu đã làm vậy. Trong dự án thực tế, đánh giá thêm theo camera/góc quay và người hoàn toàn mới. `confidence` là softmax chưa được hiệu chuẩn.

## 6. Webcam hoàn toàn offline (chỉ sau khi đã train **hai** model thực)

```bash
python live.py --source 0 --pose-model models/pose.pt --action-model models/action.pt
# Hoặc dùng file video trong máy
python live.py --source videos/test_person_new.mp4 --pose-model models/pose.pt --action-model models/action.pt
# Dành cho server không GUI: JSON stdout
python live.py --source videos/test_person_new.mp4 --pose-model models/pose.pt --action-model models/action.pt --headless
```

Phím `Q` thoát GUI. Streaming dùng 32-frame buffer, suy luận mỗi 4 frame; exponential smoothing + số lần nhãn ổn định; class không đủ ngưỡng trả `unknown`. Bộ đếm squat (góc gối), push-up (góc khuỷu) và lunge (góc gối thấp nhất) có hai ngưỡng và reset pha khi đổi động tác/mất pose. Hiển thị khung xương, nhãn và số rep trong cửa sổ; JSON trên stdout khi phân loại. `--threshold 0.65 --alpha 0.6 --step 4 --min-votes 2` cho phép điều chỉnh.

**Giới hạn chuyên môn:** đếm rep và đo góc là quy tắc hình học 2D, nhạy với góc đặt camera, khuất khớp, pose sai, động tác lệch mặt phẳng. Đừng xem là đánh giá form an toàn hay lời khuyên y tế. Cần làm bộ nhãn form đúng/sai, hiệu chuẩn threshold trên thực tế, kiểm thử từng bài tập với huấn luyện viên trước khi phát cảnh báo.

## 7. Test và cấu trúc source

```bash
python -m pytest -q
```

| File | Chức năng |
| --- | --- |
| `visionfit.py` | ActionNet Temporal CNN; normalize/resample; train/predict; early stopping; test split; SquatCounter |
| `pose_model.py` | PoseNet 33 điểm, hàm loss heatmap + visibility, train và suy luận từ ảnh RGB |
| `pose_annotator.py` | Gán nhãn 33 điểm thủ công trên ảnh offline |
| `extract_frames.py` | Video local → ảnh cho annotate |
| `extract_clips.py` | Video/camera → PoseNet của bạn → clip landmarks có nhãn |
| `build_dataset.py` | Nhiều clip khác độ dài → một action dataset `.npz` |
| `local_video.py` | Mở camera/video local, kiểm tra chất lượng pose, vẽ skeleton |
| `live.py` | Offline webcam/video + smoothing + counter |
| `repetition.py` | PushUpCounter, LungeCounter |
| `metrics.py` | Confusion matrix, precision/recall/F1, accuracy |
| `test_visionfit.py`, `test_pro.py` | Kiểm thử đơn vị bằng dữ liệu tổng hợp |
| `requirements-camera.txt` | Thư viện camera/GUI cài vào máy, không phải API |

Không có dịch vụ bên ngoài, khóa API, model hub hay lệnh tự download. Không lưu mặc định video webcam từ `live.py`. Khuyến nghị xóa video của tình nguyện viên khi không còn cần, cấp quyền truy cập thích hợp, tránh dùng tên thật làm subject ID. Python scripts nhận checkpoint bằng `torch.load(..., weights_only=True)`; vẫn chỉ mở file mô hình bạn tin cậy.
