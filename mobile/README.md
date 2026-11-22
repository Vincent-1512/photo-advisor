# Chạy ứng dụng Photo Advisor

Ứng dụng Flutter trong `app/` gọi FastAPI. Từ gốc repository, khởi động backend trong môi trường Python 3.10.

```sh
python -m pip install -r requirements.txt -r backend/requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Trong terminal thứ hai, chạy các lệnh sau với Flutter stable và Android Emulator đang mở.

```sh
cd mobile/app
flutter pub get
flutter run --dart-define=API_URL=http://10.0.2.2:8000
```

Trên iOS Simulator, dùng `flutter run --dart-define=API_URL=http://127.0.0.1:8000`. Trên điện thoại thật, cấu hình URL HTTPS của backend mà điện thoại truy cập được.

Chọn JPEG/PNG hoặc chụp ảnh, chọn mục tiêu rồi xác nhận đồng ý. App hiển thị từng ảnh ở cùng kích thước và xáo thứ tự mỗi lần mở kết quả. Chạm vào ảnh để lưu JPEG vào thư mục tài liệu của ứng dụng hoặc gửi lựa chọn kèm lý do. Lịch sử lưu ID trên thiết bị theo địa chỉ máy chủ. Nút xóa yêu cầu xác nhận và xóa ảnh, phương án cùng phản hồi trên máy chủ. JPEG đã xuất về máy thuộc người dùng.

```sh
flutter analyze
flutter test
```
