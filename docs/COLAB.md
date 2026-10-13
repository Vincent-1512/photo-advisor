# Chạy Photo Advisor trên Google Colab

Mục tiêu đầu tiên: hai ảnh thật, mỗi ảnh một lượt inference và một bộ artifacts riêng. Chưa batch dataset lớn hoặc training. Kết quả mock dùng để kiểm tra phần mềm, không đo chất lượng model.

## 1. Đưa source vào runtime

Upload riêng notebook là chưa đủ: runtime cần `src/`, `prompts/`, `tests/` và requirements.

Nếu repository có thể truy cập từ runtime, thêm ô đầu notebook:

```python
%cd /content
!git clone https://github.com/Vincent-1512/photo-advisor.git
%cd /content/photo-advisor
```

Nếu repository private và Colab chưa được xác thực, tải ZIP source qua GitHub rồi upload lên Colab hoặc Drive. Giải nén và chuyển working directory đến thư mục có cả `src/` và `prompts/`; không nhập token vào notebook/chia sẻ notebook có token.

Ví dụ source được giải nén trên Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/photo_advisor
```

Cell 3 dùng working directory để tìm source. Không thay tên package `photo_advisor` trong `os.path.join()` bằng một absolute path; thay working directory hoặc đường dẫn root trước khi chạy cell.

## 2. Bật GPU và chạy prerequisites

1. Chọn **Runtime → Change runtime type → GPU**.
2. Chạy Cell 1 để xem CUDA/VRAM; real mode cần CUDA.
3. Chạy Cell 2 để cài thư viện rồi Cell 3 để nạp source. Nếu Colab yêu cầu restart sau cài đặt, restart runtime và chạy lại các cell cần thiết, gồm ô chuyển working directory.
4. Kiểm tra đường dẫn source được in ra thuộc repository bạn vừa nạp.

Notebook chứa dải phiên bản remote riêng; full tests local không xác nhận install/model fitting trên Colab. Ghi lại versions và lỗi thực tế nếu có.

## 3. Chạy mock để thử luồng

Cell 4 giữ `EXECUTION_MODE = "mock"`. Chạy Cell 4–8, rồi Cell 10 nếu muốn tải zip. Mock không tải weights kể cả runtime có GPU. Thiếu ảnh mẫu trong mock mode thì notebook sinh synthetic.

Kết quả có `provider_type="mock"`, `is_real_model=false`, tokens simulated. Những câu mô tả người/cây/ánh sáng trong mock là nội dung mẫu.

## 4. Chạy real với ảnh thứ nhất

Chuẩn bị ảnh JPEG/PNG người thật được phép sử dụng; ghi quyền sử dụng trong manifest dữ liệu riêng. Dòng mô tả consent cố định trong notebook không tự xác minh quyền dùng ảnh.

1. Upload ảnh, ví dụ `/content/photo-advisor/work/phase02b/portrait_1.jpg`.
2. Cell 4: đổi `EXECUTION_MODE = "real"`, chạy Cell 4–5.
3. Cell 6: đặt `input_image_path` đúng với file đã upload và `GOAL = "natural_portrait"`.
4. Chạy **Cell 6 → 7 → 8**. Model lazy-load trong pipeline; lần đầu cần tải checkpoint trên Colab.
5. Kiểm tra status, `provider_type`, `is_real_model`, model revision/config và preview. Lỗi/OOM/schema cần được bàn giao như lỗi, không báo đạt.
6. Chạy Cell 10 để tải ZIP, đổi tên ZIP trên máy thành `portrait-01-artifacts.zip`.

Notebook chọn model 3B hoặc 7B dựa theo VRAM. [3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) có nhãn license `qwen-research`, [7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) có `Apache-2.0` theo metadata đã review; đọc LICENSE chính thức theo revision sử dụng. Không suy giấy phép model từ giấy phép của thư viện.

## 5. Chạy ảnh thứ hai

Đổi đường dẫn/goal trong Cell 6, rồi **chạy lại Cell 6 trước Cell 8**. Nếu chỉ thay biến đường dẫn và chạy Cell 8, `raw_bytes` có thể vẫn là ảnh trước.

Chạy Cell 7–8–10, tải ZIP và đổi tên thành `portrait-02-artifacts.zip`. Mỗi lượt sinh output directory mới; hai ZIP tải xuống có thể cùng tên mặc định nhưng phải giữ riêng.

Hướng dẫn batch nằm trong notebook; chỉ dùng sau khi đã xác nhận hai lượt đơn lẻ. Không đổi prompt/processor/model giữa hai lượt nếu đang so sánh cùng một baseline.

## 6. Kiểm tra và bàn giao

| File/thông tin | Kiểm tra |
|---|---|
| `reference_input.png` | Đúng ảnh, đúng chiều, màu hợp lý |
| `vlm_raw_response.json` | Provider thật, status, raw text, config/model revision, timings/tokens |
| `manifest.json` | Source/hash đúng ảnh, requested/effective recipe, fallback lý do |
| Preview + `index.html` | Đối chiếu giữ gốc, rule và AI; không gọi fallback là ảnh đã cải thiện |
| Feedback | Người dùng chọn phương án nào, lý do, có thích hơn gốc không |

Hash input hai lượt cần tương ứng hai ảnh khác nhau. Không lấy `is_real_model=true` riêng lẻ làm kết luận chất lượng; kiểm tra cả status/raw response và nội dung ảnh.

Cell 9 chạy tests offline, ép CUDA ẩn trong tiến trình test; không thay cho real smoke test. Time budget generation là cooperative, không ngắt cứng kernel CUDA bị treo.

Nếu lỗi đỏ ở bước prerequisites, dừng và lưu nội dung lỗi/versions. Nếu pipeline xuất artifacts với status lỗi, giữ bundle để review; không tải lại model hoặc retry hàng loạt để che lỗi.
