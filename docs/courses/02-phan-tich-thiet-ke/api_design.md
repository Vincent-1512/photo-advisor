# Thiết kế API và CSDL Photo Advisor (G1)

## Phạm vi và đầu vào

Đây là hợp đồng đề xuất cho cổng G1, chưa phải API/backend đã triển khai. Tài liệu SAD-2 gồm ERD và `models.md` chưa được cung cấp trong workspace; mô hình bảng dưới đây được suy ra từ BLOCK-0 và manifest pipeline hiện có. Cần đối chiếu lại với SAD-2 khi tài liệu đó sẵn sàng.

- OpenAPI: [`docs/contracts/openapi.yaml`](../../contracts/openapi.yaml), OpenAPI 3.0.3.
- SQLite: [`backend/db/schema.sql`](../../../backend/db/schema.sql).
- Prefix API: `/v1`; trường JSON và tên tham số dùng `snake_case`.
- Thời gian lưu/trao đổi phải là ISO-8601 UTC, ví dụ `2026-11-08T10:00:00Z`.
- ID là chuỗi có tiền tố: `job_`, `candidate_`, `feedback_`, `session_`. Phần sau tiền tố do server sinh, không nhận ID tùy ý từ upload.
- Ảnh JPEG/PNG tối đa 15 MiB (15,728,640 byte). `consent=true` bắt buộc.
- Poll `GET /v1/jobs/{job_id}` theo `Retry-After`; khuyến nghị 2 giây khi queued/running. Header là gợi ý, không phải cam kết thời gian hoàn tất.
- Idempotency: không hỗ trợ trong phiên bản này. Mỗi lần POST hợp lệ tạo job mới; client không nên tự retry POST sau timeout nếu không kiểm tra kết quả nghiệp vụ.

## Quyết định schema API

### Job/result

POST trả `202` cùng `job_id` và trạng thái `queued`. GET trả `200` cho cả queued/running/succeeded/failed, trong đó result/error có thể null. Trạng thái cuối `succeeded` mang `result`; `failed` mang `error`. HTTP `404` dành cho job hoặc candidate không tồn tại; lỗi nghiệp vụ job ở trạng thái failed vẫn được trả trong payload GET `200`. `provider_type` là provenance của provider VLM hoàn tất inference và null nếu không có inference hoàn tất; nguồn từng style là `Candidate.source`.

Result luôn có `analysis`, `keep_original`, `candidates`, `retake_advice`. `candidates` chứa đúng một lựa chọn `noop` giữ gốc và từ 0 đến 5 phong cách hợp lệ; `noop` nằm ngoài số K (mặc định K=4, cấu hình K=4 hoặc 5). Nếu guardrail và lọc khác biệt không còn đủ phong cách, trả ít hơn K, không nhân bản hoặc độn candidate. Kết quả không có nhãn tốt nhất/đề xuất và không xếp hạng. Giới hạn tối đa hai option chỉ áp dụng cho JSON do VLM sinh; rule có thể tạo thêm ứng viên, còn `learned` là nguồn dự kiến, chưa triển khai. Candidate bị `fallback_noop` hoặc `rejected` không được tính vào phong cách trả về. Recipe No-op phải thể hiện trung tính; fallback No-op không được mô tả là ảnh đã cải thiện. Retake advice chỉ là văn bản.

`style_id` và `style_label` là trường tùy chọn để giữ tương thích payload cũ. Khi có, ID là tên ổn định của kiểu chỉnh; nhãn tiếng Việt chỉ mô tả thông số như “Ấm, tương phản cao”. Không dùng `keep_original.recommended` để chọn, xếp hạng, ẩn candidate hoặc gắn nhãn UI; trường này được giữ làm dữ liệu tương thích với nội dung VLM hiện tại và không phải kết luận chất lượng.

Server trả candidates theo thứ tự ổn định để cùng job có payload tái lập. App xáo ngẫu nhiên thứ tự ảnh gốc và các phong cách một lần cho lượt hiển thị, giữ thứ tự đó trong UI state, rồi gửi `shown_order` tùy chọn trong feedback. `shown_order` gồm cả candidate `noop` nếu app đã hiển thị ảnh gốc. Trường tùy chọn cho phép client cũ tiếp tục gửi feedback; server kiểm tra ID thuộc kết quả job và không lặp.

### Recipe: `oneOf` theo pipeline version

Chọn `oneOf` gồm `RecipeV0` và `RecipeV1`, phân biệt bằng `pipeline_version`. V0 giữ nguyên tập trường bắt buộc/giới hạn đã có; V1 cho phép thêm field và thiếu field mới được hiểu là trung tính. Cách này bảo toàn ý nghĩa và tính hợp lệ của recipe cũ khi mở rộng.

BLOCK-0 chưa nêu cận dưới/cận trên cho từng phần tử `crop` ngoài dạng tọa độ chuẩn hóa, hoặc định nghĩa rõ giá trị của `local.type`. Spec giới hạn mỗi tọa độ trong [0,1], còn local để `type` dạng chuỗi và ghi rõ semantics cần chốt trước khi có processor. Đây là các quyết định đề xuất của G1, cần thống nhất cùng validator trước khi backend thực thi pipeline_v1. Crop phải giữ tối thiểu 30% diện tích theo quy tắc pipeline; tỷ lệ này là kiểm tra nghiệp vụ, không thể diễn tả đầy đủ trong OpenAPI schema.

### HTTP errors và binary

- `400`: request sai/thiếu consent.
- `404`: job hoặc candidate không tồn tại.
- `409`: candidate không đủ điều kiện export.
- `413`: upload vượt 15 MiB.
- `415`: media type không hỗ trợ.
- `422`: ảnh parse được nhưng nội dung/field không đáp ứng quy tắc.
- `204`: DELETE job và POST feedback thành công, không có response body.
- `200`: GET job/preview/health và export thành công; preview/export là `image/jpeg`.

Các endpoint JSON lỗi tuân theo `{ "error": { "code", "message" } }`. `204` và ảnh binary không có JSON example vì response body tương ứng không tồn tại hoặc không phải JSON.

## SQLite

`jobs` là aggregate gốc. `candidates` và `feedback` tham thuộc `jobs` bằng `ON DELETE CASCADE`; feedback còn tham chiếu candidate đã chọn theo khóa ghép `(job_id, candidate_id)` khi ID không null. `feedback_shown_order` lưu từng phần tử của mảng API thành một dòng `(feedback_id, position, candidate_id)` để giữ thứ tự và ràng buộc candidate thuộc đúng job, không cần cột JSON hay SQLite JSON extension. Không có `shown_order` thì không có dòng tương ứng. Khi xóa job, backend cần xóa thêm thư mục artifact trên filesystem vì FK chỉ xóa được dòng SQLite.

Không lưu bytes ảnh trong DB. `image_relative_path`, `preview_relative_path` và SHA-256 là metadata để backend quản lý storage ngoài DB. Backend phải giới hạn đường dẫn dưới root storage; không ghép path trực tiếp từ tên file do client gửi. Consent chỉ lưu cờ boolean `1`, không lưu nội dung ảnh/chi tiết consent trong bảng. `PRAGMA foreign_keys=ON` phải được bật ở mỗi connection SQLite.

`provider_runs` chưa cần ở G1: mỗi job có một provider_type và hợp đồng không yêu cầu lưu lịch sử retry/model run. Nếu cần audit retry hoặc metadata inference chi tiết, thêm bảng này sau; không đưa latency/token vào API hiện tại vì chưa có dữ liệu VLM thật.

## Đối chiếu manifest pipeline

Đã đọc `work/test-run-img0251/manifest.json`, `work/readme-smoke-20261014/manifest.json` và manifest mock trong `archive/work/phase02b/`. Tất cả các lần chạy được nêu ở đây là dữ liệu local/mock; ví dụ manifest ảnh IMG_0251 ghi rõ `provider_type: mock`, không phải kết quả VLM thật. Ánh xạ dưới đây mô tả nguồn cấu trúc, không diễn giải chất lượng.

| Trường API | Nguồn manifest hiện có | Tình trạng/adapter |
|---|---|---|
| `analysis.*` | `metadata.vlm_summary.analysis.*` | Có trong run dùng VLM; null/thiếu nếu không gọi provider. GET job phải chọn trạng thái lỗi hoặc biểu diễn analysis theo hợp đồng, không bịa kết quả. |
| `keep_original.recommended`, `reason` | `metadata.vlm_summary.keep_original.*` | Có trong `vlm_summary`; manifest candidate không lưu riêng cờ này. |
| `retake_advice` | `metadata.vlm_summary.retake_advice` | Có trong run gọi provider; nội dung chỉ là text. |
| `candidate_id` | `candidates[].candidate_id` | Có; ID pipeline hiện dạng `cand_0_noop`, API đề xuất ID `candidate_*`, cần map hoặc đổi generator. |
| `source` | `candidates[].source_type` | Có nhưng cần chuẩn hóa `noop` → `noop`, `rule`/`rule_v0` → `rule`, `vlm_mock`/`vlm_replay`/`vlm_local_hf` → `vlm`; `learned` hiện chưa có trong pipeline và là kế hoạch chưa triển khai. |
| `style_id`, `style_label` | Không có trường taxonomy phong cách trong manifest hiện tại | Trường API mới, tùy chọn. Adapter có thể tạo nhãn mô tả từ recipe; ID cần taxonomy ổn định do nhóm chốt. |
| `option_id` | Không thấy trên record candidate; có thể nằm trong raw response | API field chưa có sẵn trong manifest candidate; cần giữ option_id khi tạo candidate. |
| `name` | `candidates[].source_name` | Có; API rút gọn nhãn nguồn thành tên hiển thị. |
| `rationale` | `candidates[].notes` | Có dưới dạng note tổng hợp; mapping không bảo đảm là rationale nguyên bản của model. Nên giữ riêng rationale trong model dữ liệu khi triển khai. |
| `requested_recipe` | `candidates[].requested_recipe` | Có. |
| `effective_recipe` | `candidates[].effective_recipe` | Có. |
| `status` | `candidates[].status` | Có; các giá trị đã gồm `success`, `fallback_noop`, `rejected`. |
| `fallback_reason` | `candidates[].fallback_reason` | Có, thường null nếu không fallback. |
| `export_eligible` | `candidates[].export_eligible` | Có. |
| `preview_url` | `candidates[].preview_filename` | Chưa có URL; server phải dựng route từ `job_id/candidate_id`, không trả đường dẫn filesystem. |
| `provider_type` | `metadata.vlm_summary.provider_type` | Có: mock/replay/local_hf. Adapter map `local_hf` thành API enum `real`; vẫn giữ provenance đầy đủ nội bộ. |
| `job_id` | `metadata.run_id` | Có run_id nhưng khác định danh API job; cần tạo ID API riêng và lưu liên kết. |
| `status`, `error` của Job | Không có trạng thái job HTTP/queue chuẩn trong manifest | Cần backend job state machine. `metadata.vlm_summary.status` chỉ mô tả run provider, không thay status job. |
| `preview_relative_path`, hash preview | `candidates[].preview_filename`, `output_sha256` | Tên file và hash output có; API cần path tương đối an toàn do backend cấp. |
| `image_relative_path` | Không lưu đầy đủ trong manifest record cho mọi loại input | Backend cần quản lý storage path riêng; không đưa bytes vào DB. |
| feedback | Không có | Chưa có thu phản hồi trong pipeline; bảng/API là phần mới. |
| `shown_order` | Không có | Trường API tùy chọn mới. Backend lưu từng candidate theo thứ tự trong bảng `feedback_shown_order`; app gửi danh sách ID thực sự hiển thị, gồm cả No-op khi có. |

Các trường manifest nội bộ như model config, token, latency, timings, prompt hash, raw response và logs không được đưa vào Result công khai theo BLOCK-0. Không được dùng các con số mock để suy ra hiệu năng VLM thật. Hợp đồng selector/feedback là đề xuất, chưa có backend hoặc app triển khai; chi tiết thuật toán nằm trong [candidate_selection_spec.md](../01-xu-ly-anh/candidate_selection_spec.md). Enum `source` được mở rộng thêm `learned`; client exhaustive enum cũ cần xử lý giá trị chưa biết dù `style_id`, `style_label` và `shown_order` đều là optional.

## Kiểm tra spec và sinh Dart client

Lệnh lint đề xuất (cần mạng khi npx phải tải package):

```bash
npx @redocly/cli lint docs/contracts/openapi.yaml
```

Hoặc cài `openapi-spec-validator` trong môi trường dev phù hợp rồi chạy:

```bash
openapi-spec-validator docs/contracts/openapi.yaml
```

Sinh Dart client với OpenAPI Generator:

```bash
openapi-generator-cli generate \
  -i docs/contracts/openapi.yaml \
  -g dart-dio \
  -o mobile/generated/photo_advisor_api
```

Cần thử với phiên bản OpenAPI Generator được nhóm khóa; tên generator/kiểu nullable có thể thay đổi theo phiên bản. Chưa chạy sinh client vì repo chưa có app Dart.
