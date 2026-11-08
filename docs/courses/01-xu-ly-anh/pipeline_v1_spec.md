# Đặc tả kỹ thuật pipeline_v1

**Trạng thái:** đề xuất cho cổng G1; chưa triển khai và chưa đo hiệu năng. **Nguồn:** thiết kế, chưa có ảnh hoặc kết quả VLM thật để đánh giá. Mục tiêu là bổ sung công thức xác định, có thể tái lập, giữ hợp đồng pipeline_v0 và không tự động khẳng định ảnh đã đẹp hơn.

## 1. Phạm vi và quy ước

Ảnh đầu vào vẫn qua kiểm tra JPEG/PNG, giải mã, EXIF transpose, ICC sang sRGB và RGB 8-bit như v0. Mọi phép mới nhận mảng RGB `float32` tuyến tính trong `[0,1]` sau khi giải mã sRGB; sau mỗi phép vẫn giữ float32, rồi cuối chuỗi kẹp `[0,1]`, lượng tử hóa về uint8 bằng round-to-nearest ties-to-even. Dùng IEEE-754 float32; mask và luminance tính theo cùng dtype. Không kẹp trung gian giữa các phép tuyến tính màu, ngoại trừ lượng tử hóa v0 ở compatibility path.

Hàm giải mã sRGB từng kênh: `L = C/12.92` nếu `C <= 0.04045`, ngược lại `L=((C+0.055)/1.055)^2.4`, với `C=uint8/255`. Mã hóa ngược: `C=12.92L` nếu `L<=0.0031308`, ngược lại `C=1.055*L^(1/2.4)-0.055`. Kẹp chỉ tại ranh giới đầu ra để bảo toàn vùng vượt dải trong các phép phối hợp. Mọi op có tham số trung tính phải bypass và trả đúng mảng input, không round-trip màu gây sai khác.

### Hợp đồng tham số và phép toán

Ký hiệu `R,G,B` là các kênh tuyến tính, `Y=0.2126R+0.7152G+0.0722B`. Clamp recipe là trách nhiệm validator, processor nhận effective recipe. Mỗi phép phải hữu hạn, xác định, không dùng dữ liệu ngẫu nhiên.

| Thao tác | Mục đích, công thức và không gian | Khoảng / trung tính | Biên và lý do |
|---|---|---|---|
| Temperature + tint (một op WB) | Sửa ám màu toàn ảnh bằng cân bằng kênh tuyến tính. `a=2^(0.25*T+0.10*t)`, `b=2^(-0.05*T+0.25*t)`, `c=2^(-0.25*T-0.10*t)`; chuẩn hóa `g=(abc)^(1/3)`; output `(R*a/g,G*b/g,B*c/g)`. `T` dương ấm (tăng R/giảm B), `t` dương magenta (giảm G/tăng hai kênh ngoài). Đây là mô hình gain sRGB gần đúng, không phải chuyển đổi nhiệt độ Kelvin hay chromatic adaptation theo illuminant. Linear-light giúp gain tỷ lệ đúng với năng lượng kênh, tránh làm màu tối/sáng lệch phi tuyến như nhân mã hóa sRGB. | Mỗi field `[-1,1]`; trung tính `0,0`, bypass bit-for-bit. | Ảnh đen giữ đen; ảnh trắng/gray có thể đổi màu theo gain nếu yêu cầu WB khác 0; ảnh đơn sắc cũng có thể mất trung tính. Với `T=t=0` bắt buộc không đổi. Kẹp cuối chuỗi có thể clip highlight. Khoảng ±1 giới hạn gain theo công thức xấp xỉ, cần xem bằng mẫu da và chart trung tính trước khi coi là an toàn thị giác. |
| Shadows | Nâng/hạ vùng tối trơn, giữ vùng sáng ít ảnh hưởng. `w_s=4*(1-Y)^2*Y`; `Y'=max(0,Y + 0.5*s*w_s)`; nếu `Y>0`, scale RGB cùng hệ số `Y'/Y`, nếu `Y=0` giữ (0,0,0). Dùng luminance và scale đồng nhất để giữ chroma tương đối; linear-light vì mức sáng được cộng theo độ sáng tuyến tính. | `s ∈ [-0.5,0.5]`, trung tính `0` bypass. | Đen tuyệt đối không thể khôi phục chi tiết. Ảnh trắng không đổi do mask bằng 0. Nâng shadow khuếch đại lượng tử hóa/noise vốn có; ảnh nhỏ không được sharpen hoặc nội suy mask theo kích thước khác. Không thêm dither trong MVP để giữ kết quả xác định. |
| Highlights | Hạ/nâng vùng sáng trơn, ít tác động vùng tối. `w_h=4*Y^2*(1-Y)`; `Y'=max(0,Y + 0.5*h*w_h)`; scale RGB như op shadows. Đây là nắn tone toàn cục, không phục hồi detail đã clip. Linear-light vì tác động trực tiếp lên luminance. | `h ∈ [-0.5,0.5]`, trung tính `0` bypass. | Trắng tuyệt đối không đổi; đen không đổi. Pixel đã clip ở đầu vào không lấy lại texture. Nâng highlights có thể clip thêm khi `h>0`, hạ không tái tạo chi tiết mất. |
| Straighten | Xoay ảnh ngược chiều nghiêng quanh tâm bằng phép affine inverse mapping, lấy mẫu bicubic trên RGB tuyến tính; sau xoay lấy hình chữ nhật lớn nhất nằm trọn trong canvas xoay, tâm đồng tâm ảnh. Canvas trung gian mở rộng để không cắt góc trước khi lấy hình chữ nhật. Tính vùng inscribed rectangle từ góc `abs(theta)` và kích thước canvas đã xoay; làm tròn biên vào trong (ceil cạnh trái/trên, floor phải/dưới). Không thêm pixel nền giả. | `straighten_deg ∈ [-5,5]`; `0` bypass nguyên pixel. | Ảnh kích thước 1 hoặc 2 px một chiều / hình học không tạo được hình chữ nhật dương: reject candidate, không tự thay bằng ảnh “đã chỉnh”. Xoay khác 0 làm mất vùng rìa và giảm pixel; export eligibility kiểm tra kích thước sau xoay/crop. Bicubic có thể tạo overshoot, kẹp cuối chuỗi. |
| Sharpen | Unsharp mask trên luminance tuyến tính: `B=Gaussian(Y, sigma=1.0 px, radius=3*sigma, separable kernel truncate 3σ, reflect padding)`, `D=Y-B`, `Y'=Y+k*D`; scale RGB theo `Y'/Y` với guard `Y=0` như tone op. Đây là sharpening toàn cục, không phải deconvolution. | `k ∈ [0,1]`; `0` bypass. Sigma cố định 1 px cho v1 để tham số duy nhất dễ giải thích. | Không áp dụng thêm upscale/resample. Ảnh nhỏ có thể preview nhưng sharpening làm halo rõ; độ phân giải dưới 400 px bất kỳ chiều vẫn không đủ export theo quy tắc hiện tại. Phẳng/đen/trắng gần như không đổi, biên đen-trắng có nguy cơ overshoot và clip. |

**Mâu thuẫn thuật ngữ cần lưu ý:** `temperature/tint` ở đây là hai control gain có biên độ chuẩn hóa, không phải nhiệt độ màu Kelvin / tint camera. Không chuyển nhãn/đơn vị sang Kelvin nếu chưa có thiết kế chromatic adaptation và kiểm chứng riêng.

## 2. Thứ tự thực thi cố định

1. **Decode + allowlist + EXIF transpose.** Mọi tọa độ về sau dùng ảnh upright; giữ chính sách decode/failure hiện tại.
2. **ICC → sRGB, alpha composite và RGB8.** Không gian làm việc đã xác định; alpha hiện composite trên nền trắng như processor hiện hành. Không đổi chính sách âm thầm trong thay đổi v1.
3. **Straighten.** Làm trước crop recipe để người dùng/recipe crop nhìn hệ tọa độ đã cân thẳng; tính vùng inscribed để tránh viền rỗng.
4. **Crop.** Tọa độ normalized tính trên ảnh upright trước straighten theo hợp đồng v0; sau straighten map tọa độ crop qua biến đổi xoay rồi lấy crop. (Nếu implementation không hỗ trợ map hình chữ nhật qua xoay, crop recipe phải được định nghĩa sau straighten ở G1; không được chọn tùy tiện.) Crop trước cân màu để pixel ngoài crop không ảnh hưởng kết quả hay clipping reference. Crop giữ tối thiểu 30% diện tích và theo các kiểm tra kích thước v0.
5. **WB.** Sửa cast trên ảnh tuyến tính trước các chỉnh độ sáng.
6. **Exposure.** `2**exposure` trên RGB tuyến tính v1 để đúng nghĩa stop tuyến tính. V0 tiếp tục dùng `ImageEnhance.Brightness` trên RGB mã hóa sRGB để không đổi kết quả cũ.
7. **Shadows rồi highlights.** Hai đường cong luminance riêng, có mask trơn; thực hiện trước contrast để contrast không khuếch đại mask/clip đã có.
8. **Contrast.** V1 định nghĩa contrast luma tuyến tính quanh pivot `0.18`: `Y'=0.18+c*(Y-0.18)`, `c` trong `[0.7,1.3]`; scale RGB để giữ chroma (Y=0 thì giữ đen). `c=1` bypass. V0 giữ nguyên Pillow Contrast.
9. **Saturation.** V1 nội suy tuyến tính RGB quanh luma: `RGB'=Y+(RGB-Y)*s`, `s` trong `[0.8,1.3]`; trung tính 1 bypass. V0 giữ Pillow Color.
10. **Sharpen.** Cuối chuỗi để làm nét trên kích thước ảnh cuối, không làm các bước scale/crop sau đó phá đáp ứng nét.
11. **Clamp, encode sRGB, quantize RGB8, đo guardrail và encode preview/export.** Guardrail so ảnh xử lý với cùng vùng crop từ ảnh normalized; tăng clipping mỗi loại >0.01 thì fallback No-op như v0. Nếu fallback, candidate trạng thái `fallback_noop`, không mô tả là cải thiện.

**G1 phải quyết định hệ tọa độ crop khi kết hợp straighten.** Hợp đồng v0 hiện áp crop trực tiếp trên upright image. Đề xuất tương thích tốt nhất là crop vẫn tính trên upright ảnh nguồn, rồi ánh xạ bốn góc crop qua phép xoay và lấy bounding box giao với vùng inscribed; ghi rõ phần crop bị mất do straighten. Nếu G1 không chấp nhận thay đổi hành vi crop v0, chỉ cho phép kết hợp crop/straighten khi có test xác định ánh xạ.

### Tương thích v0

Recipe `pipeline_v0` phải tiếp tục chạy đúng chuỗi hiện hữu EXIF → ICC → RGB8 → crop → Pillow Brightness → Pillow Contrast → Pillow Color. Không route v0 qua linear ops vì sẽ thay pixel output và kết quả test. Recipe `pipeline_v1` thiếu field mới được validator điền trung tính; `exposure/contrast/saturation/crop` vẫn dùng các cùng bounds/ý nghĩa như v0 hay được định nghĩa lại tuyến tính cho v1? Đề xuất để `exposure` là stop tuyến tính, contrast/saturation theo công thức trên chỉ trong v1; đây là thay đổi semantics bắt buộc gắn phiên bản. Chưa mở rộng schema prompt/model cho đến khi adapter chấp nhận cả hai phiên bản.

## 3. Recipe schema, defaults và validator

File kèm theo `recipe_v1.schema.json` dùng draft-07. `pipeline_v0` cần đủ 5 field cũ và cấm field v1; `pipeline_v1` nhận trường v0 bắt buộc cùng field mới tùy chọn, không field lạ. Trường thiếu trong v1 lấy trung tính runtime: `temperature=0`, `tint=0`, `shadows=0`, `highlights=0`, `sharpen=0`, `straighten_deg=0`, `local=null`. JSON Schema `default` chỉ là annotation theo draft-07, **không tự điền giá trị**; validator phải materialize defaults trước khi tạo effective recipe. `local` hiện chỉ chấp nhận null; object sẽ reject đến khi nhóm chốt thuật toán, type, value và acceptance criteria. `default` không đồng nghĩa schema cho phép null ở số.

Validator: reject key lạ, version không hỗ trợ, sai kiểu (bool không được coi là number), NaN/Infinity, magnitude số vượt 1e12, crop sai hình học/kích thước/diện tích; clamp số hữu hạn nằm ngoài bounds cho cả field v0/v1, ghi warning từng trường với requested và effective. Lưu nguyên requested recipe đã sanitize; effective recipe sau clamp/default. Crop không clamp tọa độ: reject như v0. Nếu clip guardrail kích hoạt, effective recipe kết quả phải biểu thị No-op và candidate status fallback; requested giữ để truy vết. Defaults và clamp phải chạy server-side, không dựa vào JSON Schema validator tự sửa.

## 4. Giao diện package dự kiến

Tạo `src/photo_advisor/ops/` với một module mỗi thao tác: `white_balance.py`, `exposure.py`, `shadows.py`, `highlights.py`, `contrast.py`, `saturation.py`, `straighten.py`, `sharpen.py`; `color.py` chứa encode/decode sRGB (hàm tiện ích, không phải thao tác recipe); `__init__.py` export API. Crop hình học ở module riêng hoặc hàm processor vì crop đã tồn tại từ v0. Mỗi thao tác riêng có chữ ký thuần, không sửa input:

```python
def apply_white_balance(rgb_linear: NDArray[np.float32], temperature: float, tint: float) -> NDArray[np.float32]: ...
def apply_exposure(rgb_linear: NDArray[np.float32], exposure: float) -> NDArray[np.float32]: ...
def apply_shadows(rgb_linear: NDArray[np.float32], amount: float) -> NDArray[np.float32]: ...
def apply_highlights(rgb_linear: NDArray[np.float32], amount: float) -> NDArray[np.float32]: ...
def apply_contrast(rgb_linear: NDArray[np.float32], factor: float) -> NDArray[np.float32]: ...
def apply_saturation(rgb_linear: NDArray[np.float32], factor: float) -> NDArray[np.float32]: ...
def apply_straighten(rgb_linear: NDArray[np.float32], degrees: float) -> NDArray[np.float32]: ...
def apply_sharpen(rgb_linear: NDArray[np.float32], amount: float) -> NDArray[np.float32]: ...
```

`NDArray` có thể là alias typing của NumPy; shape `(height,width,3)`, dtype float32, RGB, linear sRGB. Processor xác thực hợp đồng một lần; op fail-fast trên shape/dtype sai. Không dùng PIL `ImageEnhance` bên trong v1 ops. `processor.py` phân nhánh recipe theo version: compatibility path Pillow v0 không đổi; v1 normalize → decode linear → gọi ops theo thứ tự mục 2 → encode → guardrail/fallback/output `ProcessingResult`. PIL vẫn xử lý đọc/ghi và ICC. Bump `PROCESSOR_VERSION` thành `processor_numpy_linear_v1`; giữ `PIPELINE_VERSION`/recipe version là `pipeline_v1` khi run v1. Ghi cả hai version vào manifest; không ghi đè giá trị v0 cho artifacts cũ.

## 5. Hiệu năng và preview

Chưa có số đo. Đề xuất preview cạnh dài tối đa 1280 px, giữ tỉ lệ; xử lý preview riêng để UI nhanh, còn export xử lý từ ảnh nguồn full-resolution, không upscale preview. Đo trên CPU sẵn có, không GPU thuê: cố định ảnh JPEG/PNG tổng hợp 12MP, 24MP và 48MP nếu máy đủ bộ nhớ; mỗi kích thước 10 lần warm-up + 30 lần đo, tách decode/ICC, ops, encode, peak RSS; báo median, p95, máy/CPU/RAM, phiên bản Python/Pillow/NumPy, dimensions, recipe và nguồn ảnh `synthetic|mock|replay|real`. Không gọi số đó là latency người dùng nếu chưa đo qua API (API chưa có). Ghi nhận linear float32 RGB tiêu tốn khoảng 12 byte/pixel cho một buffer; tránh giữ nhiều bản sao toàn ảnh, buffer peak cần đo. Preview metrics riêng export metrics.

## 6. Tương thích và kiểm chứng cần có

Không chạy hoặc sửa test ở bước đặc tả. Khi triển khai, giữ nguyên test hiện có cho v0: `tests/test_processor.py`, `test_validator.py`, `test_clipping_guardrail.py`, `test_regression_a1_a5.py`, `test_vlm_phase02b.py`, `test_vlm_phase02b_revise.py` và CLI/report/replay tests. Rủi ro cao nhất là test dựa vào bytes/hash PNG deterministic, validator yêu cầu chính xác key/version, và các fake provider fixture chỉ tạo `pipeline_v0`. Giữ fixture/expected output v0 bất biến; thêm test v1 tách riêng cho neutral identity (array equality), monotonic WB/tone, scalar clamp+warning/requested/effective, malformed/nonfinite/bool, clipping fallback, EXIF/ICC, rotation geometry, crop transform, edge dimensions, output deterministic hash và v0 regression. Thêm property checks cho output hữu hạn, shape đúng, kênh trong `[0,1]` sau encode. Test ảnh đen/trắng/gray/gradient/edge tổng hợp, không commit ảnh người thật.

## 7. Ma trận rủi ro

| Rủi ro | Nguyên nhân | Phát hiện/giảm thiểu | Còn lại |
|---|---|---|---|
| Halo khi sharpen | Unsharp mask tăng overshoot ở biên tương phản | k tối đa 1, sigma cố định; test chart cạnh đen-trắng, quan sát 100%/200%; clipping guardrail | Guardrail đo pixel bị clip, không phát hiện halo không clip |
| Banding khi nâng shadow | Nguồn 8-bit và gamma/noise/quantization | Phép nội bộ linear float32, lượng tử hóa một lần cuối; test gradient; không tuyên bố phục hồi dữ liệu đã mất | 8-bit đầu vào vốn đã lượng tử hóa; không có dither trong MVP |
| Ám màu khi WB | Gain kênh thay đổi trung tính, gamut bị clip | Có bypass chính xác ở 0, test neutral patch và chart da; ghi công thức/recipe; xem preview trước chọn | Không đo illuminant thực; gain gần đúng có thể sai |
| Lệch tông da | WB, saturation và tone toàn ảnh không có mask mặt | Giới hạn tham số bảo thủ, compare với gốc, giữ No-op; test nhiều màu da có consent/nguồn rõ khi có tập phù hợp | MVP chưa có dữ liệu đánh giá người thật hoặc preference; không tuyên bố bảo toàn tông da |
| Crop và straighten làm cụt chủ thể | Hai phép đổi hình học theo thứ tự/hệ tọa độ khác nhau | G1 chốt mapping, reject hình học suy biến; giữ guardrail full_body/keep_context khi không có bbox tin cậy | Không có detector bbox trong MVP |
| Highlight/shadow bị tưởng là phục hồi | Điểm đã clip không chứa chi tiết | rationale/UI gọi là điều chỉnh tone; không dùng từ phục hồi; so gốc | Không thể khôi phục detail đã mất |

## 8. Goal thống nhất

Danh sách chuẩn giữ nguyên 6 goal đang khai báo ở `types.py`: `natural_portrait`, `vibrant`, `silhouette`, `full_body`, `keep_context`, `memorial`. Prompt `vlm_system_prompt_v1.txt` lại nêu ví dụ `person_first`, `street_portrait`, đồng thời thiếu `vibrant`, `full_body`, `memorial`; đây là mâu thuẫn contract. Đề xuất bỏ hai tên không chuẩn khỏi prompt, thay bằng sáu goal chính thức trong schema/API/client; trước đó validator nên từ chối goal lạ hoặc map alias có bảng tường minh (không tự suy diễn). `person_first` có thể là ý định gần `natural_portrait` hoặc `full_body` tùy bố cục, `street_portrait` có thể gần `natural_portrait`/`keep_context`; không có mapping 1:1 đủ an toàn. `landscape` xuất hiện trong `GOALS_PRESERVE_ALL` nhưng không nằm trong sáu goal và cũng không nằm trong prompt; gỡ khỏi guard set hoặc bổ sung goal chỉ qua quyết định versioned.

## Câu hỏi mở cần nhóm quyết

1. `local` có cần trong v1? Nếu có, nhóm định nghĩa loại, vùng/mask, đơn vị value, thuật toán và test chấp nhận nào; cho đến khi chốt, đề xuất chỉ cho `null`.
2. Khi recipe có cả crop và straighten, có chấp nhận tọa độ crop tính trên ảnh upright rồi biến đổi qua phép xoay không? Cần chốt trước khi đóng băng G1.
3. Có chấp nhận semantics tuyến tính mới cho exposure/contrast/saturation trong `pipeline_v1`, trong khi v0 giữ nguyên Pillow, hay muốn chỉ thêm ops mới và giữ ba ops cũ cùng semantics sRGB?
4. Danh sách goal chuẩn có đúng 6 nhãn hiện có không; `person_first`, `street_portrait`, `landscape` sẽ bị loại hay được bổ sung bằng versioned contract?
5. Duy trì alpha PNG composite nền trắng như hiện tại, hay v1 cần chính sách khác? Thay đổi sẽ ảnh hưởng tính tương thích pixel.
