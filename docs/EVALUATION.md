# Dữ liệu và đánh giá

## Phạm vi của pilot

Ưu tiên ảnh smartphone có người, bao phủ cận mặt, nửa người, toàn thân và người trong cảnh rộng. Ánh sáng có thuận/ngược sáng, silhouette có chủ đích, thiếu sáng và chênh sáng; nền gồm cây, đường, kiến trúc và trong nhà. Có cả ảnh đã tốt để kiểm tra khả năng khuyên giữ gốc.

| Bước | Quy mô dự kiến | Mục đích |
|---|---|---|
| Smoke test | 2 ảnh thật | Xác minh runtime/model/schema/preview trước |
| Chẩn đoán ban đầu | 8–12 ảnh | Tìm lỗi theo ngữ cảnh và cỡ cảnh |
| Pilot phát triển | 40–60 ảnh | Kiểm tra tính hữu ích và lựa chọn người dùng |
| Đánh giá cuối | Tập riêng được chốt sau | Không dùng các ảnh đã điều chỉnh prompt để báo generalization |

Đây là kế hoạch, không phải số ảnh đã thu hoặc đã chạy. Hàng trăm ảnh không tự chứng minh đủ để train VLM từ đầu; quyết định training cần learning curve và dữ liệu preference phù hợp.

## Manifest và quyền dùng ảnh

[Manifest template](research/manifest-template.json) là mẫu chưa có ảnh, checksum hoặc consent được xác minh. Dùng để tạo manifest private trong `data/`/`work/`; không sửa trạng thái thành complete khi chưa có ảnh.

Ghi ít nhất image ID, đường dẫn riêng, checksum, dimensions, cỡ cảnh, ánh sáng, nền, mục tiêu và trạng thái quyền sử dụng. Với dữ liệu nhiều buổi chụp, thêm person/session/group ID để tách split. Consent dùng cho nghiên cứu nội bộ không tự cho phép đăng ảnh công khai trên GitHub.

## So sánh công bằng

Mỗi ảnh so sánh No-op, rule baseline và tối đa hai phương án VLM qua cùng processor. Ghi đúng requested/effective recipe và phương án bị fallback. Lời khuyên chụp lại đánh giá riêng, không dùng ảnh giả làm đối chứng.

Tách nhóm/person/session giữa train/dev/test nếu sau này training; không để ảnh gần trùng, crop hoặc augmentation của cùng ảnh xuất hiện ở cả hai phía. Pilot dùng chỉnh prompt là dev, không phải test khóa.

## Rubric review

Mỗi tiêu chí ghi **đạt / không đạt / không chắc / N/A**, kèm lý do hoặc vùng ảnh liên quan:

1. **Đúng nội dung:** chủ thể, cỡ cảnh, nền và ánh sáng có khớp ảnh?
2. **Đúng mục tiêu:** có tôn trọng giữ bối cảnh, màu tự nhiên hoặc silhouette?
3. **Thực hiện được:** thao tác có trong engine và preview có khớp recipe?
4. **Bảo toàn người/bối cảnh:** có cắt cụt vô lý hoặc làm mất chi tiết quan trọng?
5. **Màu da:** có tự nhiên, mất chất da hoặc ám màu? N/A khi không thấy da.
6. **Clipping/artifacts:** có làm nặng thêm lỗi so với input, không chỉ nhìn lỗi có sẵn?
7. **Lời khuyên chụp lại:** có cụ thể, có căn cứ và tách khỏi preview?
8. **Giữ gốc:** có nhận ra ảnh đã tốt hoặc can thiệp không có lợi?

Ghi preference riêng: phương án chọn, thích hơn gốc hay không, lý do, mục tiêu và option ID. Không ép chọn ảnh chỉnh; giữ gốc và không thích phương án nào là feedback hợp lệ.

## Dataset tham khảo

[Catalog](research/datasets.csv) và [source notes](research/sources.md) lưu nghiên cứu từ vòng thiết kế, không phải inventory dữ liệu đã tải. Source notes có snapshot cũ; đọc nguồn chính thức và điều khoản hiện tại trước khi tải.

- **Ảnh tự thu:** ưu tiên smoke/pilot vì sát camera, người dùng và bối cảnh thực tế.
- **PPR10K:** tham khảo portrait retouching; tách giấy phép code khỏi data terms và kiểm tra dung lượng trước khi tải.
- **GAICD:** tham khảo crop; chọn đúng subset và điều khoản, không coi crop score là đánh giá mọi thao tác sáng/màu.
- Dataset khác chỉ bổ sung khi trả lời một vấn đề cụ thể; không bulk download để chứng minh tiến độ.

Repo không chứa ảnh dataset bên ngoài, model weights hoặc kết quả benchmark thật. Model/dataset license phải kiểm tra riêng theo nguồn/revision.

## Mốc chuyển giai đoạn

Sau smoke/pilot, nhóm tổng hợp lỗi theo cỡ cảnh, ánh sáng và nền; xem AI tốt hơn rule/giữ gốc ở đâu. Cải thiện prompt/processor trước khi thêm training. Ranker nhẹ có thể học preference khi dữ liệu đủ tin cậy; fine-tune VLM/Kev chỉ cân nhắc nếu bằng chứng và nguồn lực phù hợp.
