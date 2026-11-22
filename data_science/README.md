# Phân tích phản hồi

Sau khi gửi phản hồi từ ứng dụng, chạy từ thư mục gốc dự án.

```sh
python data_science/feedback_report.py backend/data/photo_advisor.sqlite3
```

Báo cáo mở SQLite ở chế độ chỉ đọc. Đầu ra JSON gồm số lượt phản hồi, số job, số phiên, phân bố lựa chọn theo goal/nguồn và số lần hiển thị, số lần chọn theo vị trí. `preference_share` lấy số câu trả lời “thích hơn gốc” chia cho số câu trả lời có/không. Khi mẫu số bằng 0, giá trị là null. Một phiên có thể gửi nhiều lượt. Phiên là lần mở app, một người có thể có nhiều phiên. Trường vị trí chỉ bao gồm lượt có gửi shown_order.

## Thu thập pilot

Trước khi dùng ảnh người thật, ghi nhận sự đồng ý cho từng mục đích (xử lý ảnh, phân tích nghiên cứu, trình chiếu). Lưu thông tin liên hệ và bản đồng ý ở khu vực riêng của nhóm. Gắn mã ảnh, mã nhóm cảnh/người, thiết bị, điều kiện sáng, quyền sử dụng và phiên bản vào manifest riêng. Checkbox xử lý trong app chỉ bao phủ việc gửi ảnh để tạo kết quả. Dùng DB riêng cho demo tổng hợp và pilot. Khi người tham gia rút lại đồng ý, xóa job, ảnh và bản sao trong tập nghiên cứu theo mã ảnh.

Chốt nhóm người/cảnh trước khi chia train, validation và test. Tất cả crop và biến thể từ một ảnh thuộc cùng split. Khóa test trước khi chọn ngưỡng trên validation. Báo cáo theo goal, ánh sáng, nguồn và vị trí hiển thị. Khi có đủ người tham gia, ước lượng khoảng tin cậy bằng tái lấy mẫu theo người. Dữ liệu hiện tại chỉ chứng minh luồng phần mềm. [Giao thức và rubric](../docs/EVALUATION.md)

## Thí nghiệm khôi phục phơi sáng

`PYTHONPATH=src python data_science/exposure_experiment.py --seed 42` tạo 48 cảnh màu tổng hợp. Mỗi cảnh có bốn biến thể phơi sáng. Chia theo cảnh thành 30 train, 9 validation và 9 test. Hồi quy ridge dự đoán mức bù EV từ ba phân vị độ sáng. Chọn hệ số phạt theo MAE trên validation. Processor v1 áp dụng dự đoán cùng guardrail. Kết quả seed 42 trên 36 biến thể test có MAE RGB chuẩn hóa 0,0690 khi giữ nguyên và 0,0186 sau chỉnh, 0 lần fallback. Đây là sai số phục hồi theo ảnh tham chiếu đã biết. Chất lượng thẩm mỹ và sở thích người dùng cần pilot riêng.
