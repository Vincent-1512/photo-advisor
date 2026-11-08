# Đóng góp

## Nhánh

- Môn xử lý ảnh: `feat/img-...`
- Môn phân tích và thiết kế: `feat/sad-...`
- Môn khoa học dữ liệu: `feat/ds-...`
- Môn lập trình di động: `feat/mobile-...`
- Công việc repo dùng tiền tố phù hợp như `chore/...`; sửa lỗi dùng `fix/...`.
- Không push thẳng lên `main`; mở pull request để review.

## Commit

Dùng tiền tố `feat:`, `fix:`, `docs:`, `test:`, hoặc `chore:` trước mô tả ngắn.

## Kiểm thử

Chạy từ thư mục gốc repository:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```
