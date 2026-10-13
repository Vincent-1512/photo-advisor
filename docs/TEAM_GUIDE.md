# Hướng dẫn làm việc nhóm

## Ý tưởng thống nhất

Người dùng chụp hoặc tải ảnh, chọn mong muốn; VLM phân tích người, cỡ cảnh, ánh sáng và bối cảnh; hệ thống tạo tối đa hai preview chỉnh sửa cùng lựa chọn giữ gốc. Người dùng quyết định ảnh nào phù hợp với mình.

MVP ưu tiên ảnh có người. Phong cảnh/đồ vật thuần, chỉnh vùng mặt, xóa vật thể, cá nhân hóa và training là hướng mở rộng cần bằng chứng trước khi làm.

## Người mới bắt đầu từ đâu?

1. Đọc [README](../README.md) và [kiến trúc](ARCHITECTURE.md).
2. Chạy demo mock để hiểu output và guardrails.
3. Chạy full suite nếu sửa source/notebook/prompt.
4. Làm smoke test hai ảnh thật theo [Colab](COLAB.md).
5. Chấm tư vấn và preview theo [tiêu chí đánh giá](EVALUATION.md).

## Phân chia trách nhiệm

| Mảng | Sản phẩm bàn giao |
|---|---|
| Dữ liệu và đánh giá | Ảnh được phép sử dụng, manifest, rubric và feedback người chọn ảnh |
| AI/prompt | Raw responses, config/model revision, lỗi schema, phân tích đúng/sai |
| Xử lý ảnh | Recipe chạy đúng, preview nhất quán, tests và guardrails |
| Tích hợp/giao diện | Luồng input → lựa chọn → trước/sau → lưu bản mới; nối API sau pilot |
| Điều phối/review | Kiểm tra bằng chứng, phạm vi và trạng thái; cập nhật tài liệu |

Nhóm tự gán người phụ trách; repository không giả định tên thành viên hoặc vai trò đã được phân công.

## Quy trình một thay đổi

1. Tạo branch theo việc: `docs/...`, `fix/...`, `feat/...`.
2. Mô tả vấn đề và tiêu chí hoàn thành trước khi sửa.
3. Giữ thay đổi nhỏ; không sửa cả model, engine và giao diện trong một việc nếu chưa cần.
4. Chạy checks phù hợp; với thay đổi pipeline, chạy full suite và mock CLI.
5. Review source và preview, rồi mở pull request với vấn đề, thay đổi, kiểm chứng, giới hạn.
6. Người review chốt ACCEPT hoặc nêu lỗi cụ thể cần sửa; không dùng câu “tests pass” làm kết luận model hiểu ảnh đúng.

Commit dùng tiền tố `feat:`, `fix:`, `docs:`, `test:`, `chore:` và mô tả ngắn. Không force-push main hay ghi đè output thử nghiệm.

## Quy tắc bằng chứng

- Ghi rõ `mock`, `replay`, `real`; dùng `is_real_model`, status và source ảnh để kiểm tra.
- `status=success` chỉ chứng minh lượt xử lý thành công, không tự chứng minh ảnh đẹp.
- Preview phải khớp effective recipe. Khi guardrail fallback, không giới thiệu preview như đã chỉnh thành công.
- Không báo VRAM, latency, token, chất lượng hoặc preference khi chưa đo đúng môi trường.
- Giữ gốc là một lựa chọn hợp lệ; lời khuyên chụp lại không được giả thành preview.
- Không hardcode token, không upload ảnh riêng/consent form hoặc checkpoint lên Git.

## Nơi đặt file

- Source: `src/photo_advisor/`; tests: `tests/`; prompt/schema: `prompts/`.
- Hướng dẫn hiện hành: `docs/`; recipe mẫu: `examples/recipes/`.
- Input private: `data/` hoặc thư mục `work/` phù hợp notebook, đều Git-ignored.
- Run thực nghiệm: `work/<run-name>/`, không commit. Chia sẻ riêng các artifacts cần review.
- Lịch sử prompt, Word, review và output cũ của máy gốc: `archive/`, Git-ignored. Không coi hướng dẫn cũ trong archive là quy trình hiện hành.

## Việc tiếp theo

1. Hai ảnh thật với VLM thật, giữ hai bundle riêng; lần Colab đã review là mock/synthetic.
2. Mở rộng 8–12 ảnh để tìm lỗi theo cỡ cảnh/ánh sáng/bối cảnh.
3. Pilot 40–60 ảnh với lựa chọn của người dùng và tách nhóm/session.
4. Dựa trên kết quả để cải thiện prompt/processor; cân nhắc ranker khi có preference đủ tin cậy.
5. Nối API/mobile sau khi luồng và chất lượng baseline được chứng minh.
