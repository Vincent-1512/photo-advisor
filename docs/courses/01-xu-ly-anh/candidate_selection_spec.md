# Đặc tả bộ phong cách để người dùng lựa chọn

**Trạng thái:** thiết kế hợp đồng, chưa triển khai. Phần `learned` là kế hoạch, chưa triển khai hoặc huấn luyện. Tài liệu này định nghĩa cách gom và chọn phong cách từ nhiều nguồn; hệ thống không chấm điểm chất lượng để chọn hộ người dùng.

## 1. Mục tiêu và phạm vi

Mỗi kết quả có ảnh gốc (`noop`) và 0–5 phong cách chỉnh sửa. Mặc định selector yêu cầu `K=4`; cấu hình chỉ nhận `K=4` hoặc `K=5`. Guardrail, tính hợp lệ và độ khác biệt có thể làm số phong cách thấp hơn K. Không nhân bản candidate để đạt K. Ảnh gốc nằm ngoài K, được hiển thị ngang hàng với phong cách; thứ tự UI được app xáo ngẫu nhiên.

Không có nhãn “tốt nhất”, “đề xuất”, điểm chất lượng hay vị trí ưu tiên. `keep_original.recommended` trong payload VLM cũ không được dùng để lọc, xếp hạng, ẩn hay trình bày nổi bật bất kỳ lựa chọn nào; đây là trường tương thích/provenance cho tới khi nhóm bỏ nó khỏi hợp đồng.

## 2. Nguồn ứng viên

| Source | Vai trò | Giới hạn/trạng thái |
|---|---|---|
| `noop` | Ảnh chuẩn hóa không áp dụng chỉnh sửa; luôn trả về, nằm ngoài K. | Đúng một candidate. Không tính là phong cách. |
| `rule` | Recipe từ rule baseline hiện có. | Có thể sinh nhiều recipe; tất cả vẫn qua guardrail và selector. |
| `vlm` | Recipe do VLM sinh. | Tối đa hai recommendation theo prompt/schema VLM hiện tại. Giới hạn này chỉ áp dụng output VLM. |
| `learned` | Bộ dự đoán được huấn luyện từ dữ liệu phản hồi; mỗi “đầu chuyên gia” sinh recipe cho một kiểu phong cách có tên ổn định. | **Kế hoạch, chưa triển khai**: chưa có dữ liệu pilot/feedback, model, huấn luyện hoặc benchmark. Không gọi là nguồn sẵn sàng hoặc kết quả thật. |

Candidate phong cách phải có `status=success`. Candidate `fallback_noop` và `rejected` bị loại khỏi pool phong cách, không biến thành một ô No-op thứ hai. Candidate `noop` duy nhất được tạo riêng và luôn còn trong kết quả.

## 3. Thuật toán selector

### 3.1 Gom, kiểm tra và loại

1. Tạo candidate No-op một lần.
2. Gom recipe ứng viên từ `rule`, tối đa 2 từ `vlm`, và các đầu chuyên gia `learned` nếu sau này được triển khai. Lỗi VLM không làm hỏng candidate hợp lệ từ nguồn khác.
3. Chuẩn hóa recipe thành `requested_recipe`; validate/clamp theo hợp đồng recipe, chạy processor và guardrail cho từng candidate. Ghi `effective_recipe` như hiện hành.
4. Loại recipe invalid, `rejected`, hoặc `fallback_noop`; candidate fallback chỉ là bằng chứng truy vết nội bộ, không phải phong cách.
5. Khử trùng lặp recipe có cùng canonical `effective_recipe` (serialize JSON với key sort, số đã chuẩn hóa theo effective recipe). Nếu trùng, giữ candidate có tie-break key ổn định nhỏ nhất; không giữ bản sao theo source.

### 3.2 Đo khác biệt preview

Đề xuất khoảng cách là trung bình ΔE2000 trên thumbnail preview:

1. Render preview của mỗi candidate từ cùng ảnh đầu vào, chuẩn hóa thành ô 256×256 bằng `contain` có giữ tỉ lệ và nền xám RGB 128. Dùng một resize filter cố định, ghi trong version selector khi triển khai.
2. Chuyển RGB sRGB sang Lab D65 theo cùng quy ước ở mọi candidate; tính ΔE2000 từng pixel giữa hai thumbnail tương ứng, rồi lấy trung bình số học trên 256×256 pixel. Đây là chỉ số khác biệt màu/độ sáng ở thumbnail, không phải điểm đẹp, độ hữu ích hay preference.
3. Quy tắc ban đầu đề xuất coi hai candidate quá giống nếu `mean(ΔE00) < 3.0`. Đây là ngưỡng heuristic chưa được kiểm chứng với người dùng; nhóm cần hiệu chỉnh bằng cặp preview tổng hợp trước, rồi dữ liệu pilot có consent. Không gọi ngưỡng này là chuẩn thị giác.

### 3.3 Max-min tham lam

Với pool M candidate phong cách còn lại, tạo ma trận khoảng cách cặp đôi. Nếu M=0, trả riêng No-op. Nếu M=1, chọn candidate đó. Nếu M≥2:

1. Tìm cặp có ΔE00 lớn nhất; nếu hòa, chọn cặp theo thứ tự `style_id`, rồi `candidate_id` tăng dần. Đưa cả hai vào tập được chọn nếu khoảng cách đạt ngưỡng tối thiểu.
2. Nếu cặp xa nhất vẫn dưới ngưỡng, chỉ chọn một đại diện: candidate có trung bình khoảng cách tới các candidate còn lại lớn nhất; nếu hòa, dùng tie-break ổn định như trên. Đây là cách lấy mẫu khác biệt, không phải xếp hạng chất lượng.
3. Lặp đến khi đủ K hoặc không còn candidate nào có khoảng cách tối thiểu tới mọi candidate đã chọn. Ở mỗi vòng chọn candidate có `min(distance(candidate, selected))` lớn nhất; nếu hòa, dùng tie-break ổn định. Candidate dưới ngưỡng không được thêm.
4. Trả ít hơn K khi pool hợp lệ/khác biệt không đủ. Không tạo placeholder, không lặp recipe, không hạ ngưỡng tự động.

Độ phức tạp tính toán là O(M²P + KM), với P=256×256 điểm thumbnail; bộ nhớ O(M²) cho ma trận khoảng cách. M phụ thuộc số rule/head đã cấu hình và chưa có số đo. Cần đo thời gian sinh preview, tính Lab/ΔE2000, peak RAM theo M và cấu hình máy; không giả định ngưỡng latency.

## 4. Mã phong cách và nhãn tiếng Việt

`style_id` là ID ổn định của một kiểu điều chỉnh, không phải `candidate_id` và không mã hóa độ tốt. ID đã phát hành không được đổi nghĩa hoặc dùng lại; đổi taxonomy thì version hóa taxonomy. `style_label` mô tả dấu và độ lớn recipe. Ví dụ:

| Dấu/độ lớn recipe hiệu lực | Nhãn mô tả gợi ý |
|---|---|
| `temperature >= 0.25` / `<= -0.25` | “Ấm hơn” / “Mát hơn” |
| `tint >= 0.25` / `<= -0.25` | “Ngả hồng” / “Ngả xanh lục” |
| `exposure >= 0.25` / `<= -0.25` | “Sáng hơn” / “Tối hơn” |
| `contrast >= 1.08` / `<= 0.92` | “Tương phản cao hơn” / “Tương phản thấp hơn” |
| `saturation >= 1.08` / `<= 0.92` | “Bão hòa cao hơn” / “Bão hòa thấp hơn” |
| `shadows >= 0.10` / `<= -0.10` | “Nâng vùng tối” / “Hạ vùng tối” |
| `highlights >= 0.10` / `<= -0.10` | “Nâng vùng sáng” / “Hạ vùng sáng” |
| `sharpen >= 0.20` | “Tăng độ nét” |
| `straighten_deg > 0` / `< 0` | “Cân thẳng phải” / “Cân thẳng trái” |

Các mốc trên là quy tắc đặt nhãn ban đầu, không phải guardrail hoặc đánh giá chất lượng; cần version hóa và nhóm duyệt trước khi triển khai. Có thể ghép tối đa hai mô tả có độ lớn chuẩn hóa cao nhất; hòa thì dùng thứ tự field cố định trong schema. Không đặt nhãn cho candidate trung tính; recipe trung tính được loại trùng với No-op. Cấm ngôn từ như “đẹp nhất”, “tốt hơn”, “khuyên dùng”, “ưu tiên”. `name` và `rationale` cũng chỉ nêu thay đổi tham số/hiệu ứng dự kiến, không khẳng định cải thiện.

## 5. Thứ tự API và thứ tự hiển thị

Server chọn và trả candidates theo thứ tự ổn định, ví dụ `noop` trước, sau đó candidate phong cách theo `style_id` rồi `candidate_id`. Đây là quy tắc serialization để cùng kết quả có payload tái lập, không phải thứ tự xếp hạng.

App tạo một hoán vị ngẫu nhiên của toàn bộ candidate được hiển thị, gồm `noop`, khi bắt đầu xem kết quả; giữ hoán vị đó khi render lại cùng phiên kết quả. Khi gửi feedback, app gửi `shown_order` là mảng `candidate_id` đúng thứ tự nhìn thấy. Backend xác thực ID thuộc job, không lặp, giới hạn 1–6 phần tử và lưu cả thứ tự lẫn candidate. Không có trường này vẫn chấp nhận feedback từ client cũ. Ngẫu nhiên vị trí giúp giảm thiên lệch vị trí đầu; cần dùng thống kê `shown_order` khi phân tích lựa chọn.

## 6. Trường hợp biên và quy tắc theo goal

- **0 phong cách hợp lệ:** trả đúng No-op; chỉ hiện ảnh gốc và `retake_advice` dạng văn bản nếu có. Không dựng preview giả hoặc gọi No-op là candidate chỉnh sửa.
- **VLM lỗi/timeout/schema sai:** loại output VLM lỗi; tiếp tục chọn từ `rule` và nguồn khác đã triển khai. Nếu không còn style hợp lệ, trả No-op. `provider_type` ghi đúng provider VLM đã hoàn tất inference, null nếu không có inference hoàn tất; không đánh đồng với `Candidate.source`.
- **`silhouette` và `memorial`:** đề xuất cap K theo goal là 2 phong cách (có thể ít hơn), dù cấu hình toàn cục là 4/5; vẫn hiển thị No-op. Candidate phải qua guardrail theo goal. Với `silhouette`, không chọn style làm mất ý đồ tương phản chủ thể/nền; với `memorial`, chỉ dùng mô tả trung tính, không áp ngôn từ cảm xúc hoặc gây áp lực. Chính sách lọc cụ thể cần nhóm chốt và test trước khi triển khai.
- **Ảnh nhỏ, preview hỏng hoặc metric không hữu hạn:** loại candidate phong cách tương ứng; không tự cho điểm khác biệt bằng 0 hoặc chèn lại No-op. Nếu không thể tính khoảng cách cho pool, trả No-op cùng các candidate đã được xác minh theo một đường fallback được định nghĩa; trước mắt đề xuất dừng chọn style và ghi lỗi nội bộ.
- **Ngẫu nhiên hiển thị:** xáo một lần cho mỗi lượt xem kết quả, không xáo lại khi widget rebuild; nếu feedback được gửi lại do lỗi mạng, dùng nguyên `shown_order` đã gửi lần đầu.

## 7. Hợp đồng API/SQLite liên quan

- `Result.candidates`: 1–6 item: đúng một `noop` và 0–5 phong cách `status=success`. Trường hợp `fallback_noop`/`rejected` có thể tồn tại trong payload cũ nhưng selector mới không trả chúng như phong cách.
- `Candidate.source`: `noop | rule | vlm | learned`. `learned` là kế hoạch, chưa triển khai.
- `Candidate.style_id` và `Candidate.style_label`: tùy chọn để client cũ tiếp tục đọc payload cũ. Candidate phong cách mới nên cung cấp cả hai; No-op không cần.
- `FeedbackInput.shown_order`: tùy chọn `candidate_id[]`, unique, tối đa 6. Mảng không có trong client cũ thì server lưu feedback mà không có thứ tự.
- SQLite lưu `style_id`, `style_label` dạng nullable trên `candidates`. `feedback_shown_order` chuẩn hóa mỗi phần tử mảng thành `(feedback_id, job_id, position, candidate_id)` với FK để kiểm tra thứ tự/ID hợp lệ. Thiếu mảng tương đương không có row.

**Khả năng tương thích:** `style_id`, `style_label`, `shown_order` là optional. Tuy vậy, thêm giá trị `learned` vào enum `source` có thể làm client cũ dùng enum đóng từ chối response; client nên có nhánh unknown-source. Candidate `noop` vẫn bắt buộc như hợp đồng trước.

## Điều nhóm cần quyết

1. Chốt ngưỡng khác biệt ban đầu ΔE00=3.0, kích thước/resize thumbnail và phương thức hiệu chỉnh trước pilot.
2. Chốt taxonomy `style_id`, các ngưỡng sinh nhãn và cách ghép tối đa hai mô tả.
3. Chấp thuận cap K=2 cho `silhouette`/`memorial` và quy tắc goal cụ thể; hiện đây là đề xuất.
4. Chốt vòng đời trường `keep_original.recommended`: giữ compatibility-only đến khi có version mới hay loại khỏi API.
5. Chốt ứng xử khi tính khoảng cách lỗi: dừng toàn bộ phong cách hay giữ các candidate đã xử lý được theo một thứ tự fallback.
