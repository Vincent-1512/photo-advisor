# Photo Advisor

Trợ lý cải thiện ảnh chụp điện thoại: xem ảnh và mục tiêu của người dùng, giải thích điểm có thể cải thiện, đề xuất tối đa hai công thức chỉnh sửa và tạo preview để so sánh với ảnh giữ gốc.

MVP ưu tiên **ảnh có người trong bối cảnh thực tế**: cận mặt, nửa người, toàn thân và người trong cảnh rộng. Người dùng có thể giữ ảnh gốc; dự án không coi một phong cách là chuẩn đẹp cho tất cả mọi người.

## Trạng thái

| Hạng mục | Tình trạng |
|---|---|
| Pipeline Python, rule baseline, mock/replay và guardrails | Đã triển khai và kiểm thử offline |
| Notebook Colab tích hợp Qwen2.5-VL | Có sẵn, cần xác minh trên phiên GPU thực tế |
| Kiểm thử trước khi dọn repository | 104 tests đạt trong lần review offline |
| Kết quả Colab đã review ngày 10/10/2026 | Mock trên ảnh tổng hợp; chưa chứng minh chất lượng VLM thật |
| App điện thoại, API, cá nhân hóa, training | Chưa triển khai |

`mock` kiểm tra luồng chạy bằng phản hồi mẫu. Chỉ khi provider thật thực hiện inference mới có thể đánh giá AI nhìn ảnh đúng và đề xuất hữu ích hay không. Không dùng số token/latency mô phỏng làm benchmark.

## Đọc tài liệu theo thứ tự

1. [Hướng dẫn nhóm](docs/TEAM_GUIDE.md): mục tiêu, cách đóng góp và công việc tiếp theo.
2. [Kiến trúc và workflow](docs/ARCHITECTURE.md): trách nhiệm từng module và dữ liệu đi qua hệ thống.
3. [Chạy trên Colab](docs/COLAB.md): source, GPU, hai ảnh thật và cách lấy kết quả.
4. [Dữ liệu và đánh giá](docs/EVALUATION.md): pilot, tiêu chí chấm và dataset tham khảo.

## Chạy thử offline

Chạy từ thư mục gốc repository. Python 3.10 đã được kiểm chứng trong quá trình phát triển; Python khác cần chạy lại checks.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m photo_advisor.cli \
  --synthetic --goal natural_portrait --provider mock \
  --output-dir work/demo-mock-01
```

Mở `work/demo-mock-01/index.html` để so sánh giữ gốc, rule baseline và các phương án mock. Khi chạy lại, chọn thư mục output mới: pipeline chủ động từ chối ghi đè artifacts.

Chạy với ảnh JPEG/PNG của bạn nhưng vẫn dùng mock:

```bash
PYTHONPATH=src python -m photo_advisor.cli \
  --image data/portrait.jpg --goal natural_portrait --provider mock \
  --output-dir work/demo-photo-01
```

Để chạy VLM thật, dùng [notebook Colab](notebooks/phase02b_colab.ipynb) theo [hướng dẫn](docs/COLAB.md). Không cần tải model về máy cá nhân để chạy demo offline.

## Kiểm thử

Full suite cần thêm thư viện cho notebook và stub model; cài `requirements-test.txt` chỉ cài thư viện, không tải checkpoint.

```bash
python -m pip install -r requirements-test.txt
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

Các tests dùng ảnh tổng hợp/mock/stub. Chúng kiểm tra tính đúng của phần mềm, không chứng minh ảnh đẹp hơn hay đo khả năng model thật. Môi trường remote của notebook cài dependencies riêng; việc chạy full suite offline không xác nhận remote setup.

## Cấu trúc repository

```text
photo-advisor/
├── README.md
├── requirements.txt            # Demo/processor offline
├── requirements-test.txt       # Full test suite
├── src/photo_advisor/          # Pipeline, VLM, validator, processor, report
├── prompts/                    # System prompt và output schema có phiên bản
├── notebooks/                  # Notebook Colab GPU
├── tests/                      # Unit/integration/regression tests
├── examples/recipes/           # Recipe fixtures, không phải inference thật
└── docs/                       # Tài liệu cho nhóm và nghiên cứu
```

`work/` chứa output, `data/` chứa ảnh riêng của nhóm và `archive/` chứa lịch sử local. Các thư mục này được bỏ qua khi commit. Ảnh trong `tests/fixtures/` là ảnh tổng hợp.

## Phạm vi và giới hạn

- Engine hiện có bốn thao tác: độ sáng, tương phản, bão hòa, crop. Chỉnh toàn ảnh; chưa có mask để chỉnh riêng mặt/người/nền.
- Lời khuyên chụp lại là văn bản; không dựng preview giả cho góc chụp chưa tồn tại.
- Luôn có phương án giữ gốc. Preview No-op là ảnh đã chuẩn hóa; file gốc không bị ghi đè.
- Guardrails kiểm tra công thức và mức clipping; chúng không thay thế đánh giá thẩm mỹ của người dùng.
- Chưa training từ đầu, chưa bắt buộc Kev, chưa học sở thích cá nhân.

## Quyền sử dụng

Giấy phép mã nguồn dự án chưa được nhóm chọn; repository chưa cấp một giấy phép open-source riêng. Giấy phép model và điều khoản dataset là những phần độc lập. Đọc LICENSE/model card chính thức trước khi sử dụng hoặc phân phối, đặc biệt khi chuyển sang mục đích thương mại. Không đưa ảnh người thật, token hoặc checkpoint vào Git.
