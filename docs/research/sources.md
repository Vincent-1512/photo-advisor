> Snapshot nghiên cứu ngày 07/10/2026, không phải trạng thái hiện tại hoặc bằng chứng đã tải dataset. Hợp đồng processor hiện hành nằm trong [kiến trúc](../ARCHITECTURE.md).

# Bảng tra cứu nguồn kiểm chứng (SOURCES.md - Bản cập nhật F1 & F2)

Tài liệu này cung cấp các source locators (đường dẫn, vị trí trong văn bản, commit/release) chứng minh cho từng khẳng định về quy mô, định dạng, license, data terms và cấu trúc split của các nguồn dữ liệu và mô hình trong Vòng 01.

---

## 1. Nguồn dữ liệu (Datasets)

### 1.1 Mobile-SelfCollected (Ảnh điện thoại tự thu của nhóm)
- **Vị trí kiểm tra:** Toàn bộ workspace dự án `workspace phát triển local` (kiểm tra ngày 07/10/2026).
- **Hiện trạng:** Chưa có bất kỳ tệp ảnh nào trong workspace.
- **Data Terms / Pháp lý:** Bắt buộc phải có phiếu đồng thuận (consent form) của người được chụp trước khi đưa vào tập pilot. Nhóm sở hữu nội bộ dữ liệu tự thu.

---

### 1.2 PPR10K (Portrait Photo Retouching)
- **Kho lưu trữ chính thức:** https://github.com/csjliang/PPR10K (nhánh `master`, kiểm tra ngày 07/10/2026).
- **Bài báo gốc:** Jie Liang et al., *"PPR10K: A Large-Scale Portrait Photo Retouching Dataset with Human-Region Mask and Group-Level Consistency"*, CVPR 2021 (https://arxiv.org/abs/2105.09180).
- **Quy mô và Dung lượng công bố (README bảng 'Overview'):**
  - RAW: 11.161 ảnh (.CR2, .NEF, .ARW), dung lượng 313 GB.
  - Gói `train_val_images_tif_360p`: 91 GB, 97.894 tệp TIF. Gồm ảnh nguồn 16-bit TIF (kèm 5 phiên bản tăng cường sáng/màu) và ảnh mục tiêu 8-bit TIF retouched bởi 3 chuyên gia (a, b, c) trong không gian màu sRGB (độ phân giải 540x360 hoặc 360x540).
  - Metadata: `xmp_source`, `xmp_target_a/b/c` (130 MB mỗi loại, định dạng XMP CameraRaw).
  - Masks: `masks_full` 697 MB PNG, `masks_360p` 56 MB PNG (mặt nạ vùng người nhị phân).
  - Tổng thư mục Google Drive / Baidu Yun: 406 GB (176.072 tệp).
- **Bản quyền mã nguồn vs Điều khoản dữ liệu:**
  - *Mã nguồn:* Apache-2.0 (mục 'License' trong README).
  - *Dữ liệu:* Chỉ nghiên cứu phi thương mại (mục 'Agreement' trong README: *"All files in the PPR10K dataset are available for non-commercial research purposes only. You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data"*).
- **Đơn vị tải nhỏ nhất & Điểm chưa xác minh:**
  - Gói tải công bố nhỏ nhất cho huấn luyện/đánh giá là `train_val_images_tif_360p` (91 GB). Không có gói nén mini-subset chính thức.
  - *Chưa xác minh:* Khả năng tải script tự động từng file riêng lẻ từ Google Drive.
- **Split & Group Mapping:**
  - README ghi dữ liệu thuộc 1.681 nhóm (groups); phân chia 8.875 ảnh đầu làm train và 2.286 ảnh sau làm val.
  - *Chưa xác minh:* Bảng mapping công khai chi tiết giữa từng tên file và Group ID tại thư mục gốc của repository; cần kiểm tra metadata nội bộ khi tải để tránh rò rỉ session chụp.

---

### 1.3 GAICD (Grid Anchor-based Image Cropping Database) & Human-Centric Subset
- **Kho lưu trữ mã nguồn GAICD gốc:** https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch (nhánh `master`, tác giả Hui Zeng et al.).
  - File `LICENSE` chính thức trong repository: **MIT License**.
  - README mục 'Usage' cung cấp 2 liên kết Google Drive độc lập:
    1. *Conference version (CVPR 2019):* Google Drive ID `1KhmyjoimsQVXqPnLjKZiU4iXNKNyyxqW`. Quy mô: 1.236 ảnh gốc, 106.853 crop annotations; split 1.036 train / 200 test.
    2. *Journal version (TPAMI 2020):* Google Drive ID `1tDdQqDe8dMoMIVi9Z0WWI5vtRViy01nR`. Quy mô: 3.336 ảnh gốc, 288.069 crop annotations; split 2.636 train / 200 val / 500 test.
- **Kho lưu trữ Human-Centric Cropping:** https://github.com/bcmi/Human-Centric-Image-Cropping (ECCV 2022, tác giả Bo Zhang et al.).
- **Bằng chứng đối soát phiên bản (Metadata bcmi - Reviewer-verified):**
  - `human_bboxes/GAICD/original_data_split.json`: 1.036 train, 200 test (tổng 1.236 ảnh). Khớp chính xác với phiên bản **GAICD Conference**.
  - `human_bboxes/GAICD/image_crop.json`: 1.236 ảnh với 106.853 crop evaluations.
  - `human_bboxes/GAICD/human_bboxes.json`: 339 ảnh có người (chứa bounding boxes người).
  - `human_bboxes/GAICD/human_data_split.json`: 289 train, 50 test (tổng 339 ảnh). Union của train và test bằng chính xác 339 IDs. Toàn bộ 339 IDs đều có mặt trong `image_crop.json` và thuộc split gốc 1.036/200.
- **Kết luận nối phiên bản và Trạng thái dữ liệu:**
  - Human subset 339 ảnh của bcmi được xây dựng trên **GAICD Conference (1.236 ảnh)**, KHÔNG PHẢI trên GAICD Journal (3.336 ảnh).
  - *Trạng thái kiểm chứng:* Metadata JSON đã được reviewer kiểm chứng (Reviewer-verified). Đối soát tệp ảnh vật lý trong archive vẫn ở trạng thái **pending (chưa tải ảnh)**.
- **Bản quyền & Điều khoản:**
  - *Mã nguồn:* Repo PyTorch gốc có giấy phép MIT License. Mã nguồn bcmi: Unlicensed (mặc định GitHub).
  - *Data terms:* **Chưa xác minh data terms** (không có văn bản cấp phép riêng cho tập ảnh trong repository; ảnh thu thập từ Flickr, hoãn sử dụng cho thương mại).
  - *Dung lượng & Color:* **Chưa xác minh dung lượng byte và không gian màu chính thức**; hoãn tải archive lớn.

---

### 1.4 CPC (Comparative Photo Composition)
- **Trang chủ chính thức:** https://www3.cs.stonybrook.edu/~cvl/projects/wei2018goods/VPN_CVPR2018s.html (Zijun Wei et al., CVPR 2018).
- **Quy mô công bố:** 10.797 ảnh gốc; hơn 1.000.000 comparative view pairs.
- **Subset người bcmi:** `human_bboxes/CPC/human_bboxes.json` (1.154 ảnh); `image_crop.json` (~12.5 MB).
- **Liên kết tải:** Google Drive ID `1TMvuCSONEN1_9y7KnzKgy_7_fSFTHzyO` (CPCDataset).
- **Điều khoản & Đánh giá:**
  - *Data terms:* **Chưa xác minh data terms** (project page không công bố điều khoản bản quyền cụ thể; hoãn sử dụng nguồn này).
  - *Dung lượng & Color:* **Chưa xác minh dung lượng byte và profile màu chính thức**.
  - Không chọn cho pilot vòng 1 vì dung lượng lớn, điều khoản chưa rõ và cấu trúc pairwise phức tạp.

---

### 1.5 SPAQ (Smartphone Photography Attribute and Quality)
- **Kho lưu trữ chính thức:** https://github.com/h4nwei/SPAQ (Yuming Fang et al., CVPR 2020).
- **Quy mô công bố:** 11.125 ảnh gốc chụp từ 66 smartphone thực tế; nhãn MOS chất lượng, 5 thuộc tính kỹ thuật, nhãn cảnh, thông số EXIF.
- **Điều khoản & Đánh giá:**
  - *Data terms:* **Chưa xác minh data terms** (README repository chỉ nêu quy mô và link tải, không chứa văn bản cấp phép bản quyền dữ liệu; hoãn sử dụng cho thương mại).
  - *Dung lượng:* **Chưa xác minh dung lượng byte chính thức**; hoãn tải archive lớn.

---

### 1.6 Q-Instruct
- **Kho lưu trữ chính thức:** https://github.com/Q-Future/Q-Instruct (Haoning Wu et al., CVPR 2024).
- **Quy mô:** 200.000 instruction-response pairs trên 18.973 ảnh Q-Pathway.
- **Bản quyền & Điều khoản:**
  - *Mã nguồn:* **S-Lab License 1.0** (quy định tại file `LICENSE` chính thức https://raw.githubusercontent.com/Q-Future/Q-Instruct/main/LICENSE, có giới hạn phi thương mại).
  - *Model Checkpoints / Weights:* **Chưa xác minh riêng cho từng checkpoint** (tuân theo base model card như LLaVA-1.5 / mPLUG-Owl2; hoãn sử dụng checkpoint khi chưa kiểm tra điều khoản base model).

---

## 2. Mô hình Vision-Language (VLM Candidates)

### 2.1 Phương án 1: Qwen2-VL-7B-Instruct (Local Open-weights)
- **Model Card chính thức:** https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct (kiểm tra ngày 07/10/2026).
- **Bản quyền:** Apache-2.0 (thương mại và nghiên cứu).
- **Phân biệt hai cơ chế Structured Output:**
  1. *Constrained Decoding (Ép token ở bước lấy mẫu):* Sử dụng inference engine (như vLLM tích hợp Outlines / Guidance / lm-format-enforcer) để ép không gian sinh token thỏa mãn ngữ pháp JSON schema, loại bỏ lỗi cú pháp từ gốc.
  2. *Validator hậu kiểm (Post-generation Validation):* Tầng kiểm tra sau khi sinh bằng Pydantic / jsonschema. Kiểm tra kiểu dữ liệu, trường bắt buộc, số hữu hạn (finite float); nếu vi phạm thì reject và fallback về Giữ gốc.
- **Tài nguyên VRAM (Ước tính có điều kiện - Chưa đo kiểm):**
  - Ước tính runtime FP16/BF16: ~18–24 GB VRAM (gồm trọng số ~14 GB + visual tokens 1024x1024 + KV cache).
  - Ước tính runtime INT4 (AWQ/GPTQ): ~8–12 GB VRAM.
  - *Trạng thái:* Ước tính có điều kiện; chưa đo kiểm trên cấu hình máy trạm thực tế của nhóm.

---

### 2.2 Phương án 2: Cloud API Thương mại (OpenAI `gpt-4o-mini`)
- **Tài liệu chính thức & Vòng đời:**
  - Model Endpoint: `gpt-4o-mini` (OpenAI Docs: https://platform.openai.com/docs/models/gpt-4o-mini, kiểm tra ngày 07/10/2026). Model đang hoạt động chính thức.
  - (Đã loại bỏ `gemini-1.5-flash` do shutdown 29/09/2025; Google Gemini 2.5 Flash bị hạn chế người dùng cũ theo https://ai.google.dev/gemini-api/docs/deprecations).
- **Cơ chế Structured Output:** Hỗ trợ native Structured Outputs của OpenAI qua cấu hình `response_format: { type: "json_schema", json_schema: {...} }`.
- **Bản quyền & Điều khoản:** Tuân theo Commercial Terms of Service của OpenAI.
- **Trạng thái truy cập:** Chờ cấp API key và hạn mức sử dụng từ người dùng / nhóm.
