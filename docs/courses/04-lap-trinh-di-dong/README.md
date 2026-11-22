# Lập trình di động

[Ứng dụng Flutter và hướng dẫn chạy](../../../mobile/README.md). Màn hình quản lý ảnh đang chọn, consent, trạng thái chờ, kết quả và lịch sử. Tầng PhotoApi phụ trách multipart upload, polling, feedback, export và xóa. Màn hình giữ thứ tự hiển thị khi người dùng chọn hoặc nhập nhận xét.

Khi nghiệm thu trên thiết bị, thử từ chối quyền camera, chọn ảnh quá 15 MiB, mất mạng lúc tải lên, xuất một phương án và xóa job rồi mở lại lịch sử. Kiểm tra trên cả Android và iOS. Kết quả widget test chỉ bao phủ điều kiện bật nút gửi.
