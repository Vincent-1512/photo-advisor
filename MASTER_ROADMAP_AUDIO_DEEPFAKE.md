# HỆ THỐNG PHÁT HIỆN VÀ ĐỊNH DANH NGUỒN GỐC GIẢ MẠO GIỌNG NÓI BẰNG THỊ GIÁC MÁY TÍNH VÀ HỌC SÂU

**Audio Deepfake Detection & Model Attribution via Computer Vision and Deep Learning**

---

> **Tác giả:** [Họ và tên sinh viên]  
> **GVHD:** [Họ và tên giảng viên]  
> **Chuyên ngành:** Trí tuệ Nhân tạo / Khoa học Máy tính  
> **Năm học:** 2026 – 2027  

---

## MỤC LỤC

- [Chương 1 – Đặt Vấn đề & Ý nghĩa Điều tra](#chương-1--đặt-vấn-đề--ý-nghĩa-điều-tra-forensics)
- [Chương 2 – Kỹ thuật Xử lý Ảnh Quang phổ & Dấu vết Trình sinh Giọng](#chương-2--kỹ-thuật-xử-lý-ảnh-quang-phổ--dấu-vết-trình-sinh-giọng)
- [Chương 3 – Ma trận Nhãn 3 Cấp độ (Hierarchical Label Matrix)](#chương-3--ma-trận-nhãn-3-cấp-độ-hierarchical-label-matrix)
- [Chương 4 – Kiến trúc Mô hình AI Đa nhiệm (Multi-Task Deep Learning Pipeline)](#chương-4--kiến-trúc-mô-hình-ai-đa-nhiệm)
- [Chương 5 – Tập dữ liệu & Metric Đánh giá](#chương-5--tập-dữ-liệu--metric-đánh-giá)
- [Chương 6 – Lộ trình Thực hiện Chi tiết (Master Roadmap)](#chương-6--lộ-trình-thực-hiện-chi-tiết-master-roadmap)
- [Tài liệu Tham khảo](#tài-liệu-tham-khảo)

---

## DANH MỤC VIẾT TẮT

| Viết tắt | Đầy đủ |
|----------|--------|
| TTS | Text-to-Speech – Chuyển văn bản thành giọng nói |
| VC | Voice Conversion – Chuyển đổi giọng nói |
| GAN | Generative Adversarial Network – Mạng sinh đối kháng |
| LFCC | Linear Frequency Cepstral Coefficients |
| CQCC | Constant Q Cepstral Coefficients |
| STFT | Short-Time Fourier Transform |
| CQT | Constant Q Transform |
| LCNN | Light Convolutional Neural Network |
| MFM | Max-Feature-Map |
| EER | Equal Error Rate – Tỉ lệ lỗi cân bằng |
| t-DCF | tandem Detection Cost Function |
| BCE | Binary Cross-Entropy |
| CE | Cross-Entropy |
| ASV | Automatic Speaker Verification |
| SPL | Sound Pressure Level |
| F0 | Fundamental Frequency – Tần số cơ bản |
| SNR | Signal-to-Noise Ratio |

---

# CHƯƠNG 1 – ĐẶT VẤN ĐỀ & Ý NGHĨA ĐIỀU TRA (FORENSICS)

## 1.1. Bối cảnh: Mối đe dọa từ Audio Deepfake

Sự phát triển bùng nổ của các mô hình sinh giọng nói (Speech Synthesis) dựa trên Học sâu (Deep Learning) đã tạo ra một mối đe dọa an ninh chưa từng có. Năm 2019, một CEO tại Anh đã bị lừa chuyển **243.000 USD** qua điện thoại bởi giọng nói deepfake mô phỏng chính xác giọng sếp của mình [1]. Năm 2024, một nhân viên tài chính tại Hong Kong đã chuyển **25.6 triệu USD** sau cuộc gọi video deepfake giả mạo CFO công ty [2].

Các hệ thống TTS (Text-to-Speech) và VC (Voice Conversion) hiện đại — bao gồm VITS, VALL-E, Bark, ElevenLabs, và các dịch vụ thương mại — có khả năng tạo ra giọng nói **gần như không thể phân biệt bằng tai người** chỉ từ vài giây mẫu giọng nói gốc (zero-shot / few-shot voice cloning) [3]. Thời gian tạo deepfake đã giảm từ **hàng giờ** (2018) xuống **vài giây** (2024), trong khi chất lượng tăng theo cấp số nhân.

### 1.1.1. Các kịch bản tấn công điển hình

| Kịch bản | Kỹ thuật sử dụng | Mức độ nguy hiểm |
|-----------|------------------|-------------------|
| **Lừa đảo tài chính** (Vishing) | TTS clone giọng CEO/CFO gọi điện chỉ đạo chuyển tiền | 🔴 Cực cao |
| **Phá hoại chính trị** | TTS/VC tạo bản ghi giả mạo phát biểu của chính trị gia | 🔴 Cực cao |
| **Bypass xác thực giọng nói** | VC chuyển đổi giọng kẻ tấn công thành giọng nạn nhân | 🔴 Cực cao |
| **Tống tiền / Bôi nhọ** | TTS tạo bản ghi âm giả mạo nạn nhân nói lời xúc phạm | 🟠 Cao |
| **Replay Attack** | Thu lại giọng thật và phát lại qua loa | 🟡 Trung bình |

## 1.2. Ba Cấp độ Phát hiện — Từ "Có/Không" đến "Ai đã làm?"

Hầu hết các hệ thống phát hiện audio deepfake hiện tại chỉ dừng ở **Cấp độ 1** — trả lời câu hỏi nhị phân "Đoạn âm thanh này thật hay giả?". Điều này **hoàn toàn không đủ** cho mục đích điều tra pháp y số (Digital Forensics), nơi điều tra viên cần biết **kẻ tấn công đã dùng công cụ gì** để truy vết nguồn gốc.

Đồ án này đề xuất hệ thống phát hiện **3 cấp độ** (Three-Level Hierarchical Detection):

```
┌─────────────────────────────────────────────────────────────────────┐
│              HỆ THỐNG PHÁT HIỆN 3 CẤP ĐỘ                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Level 1: BINARY DETECTION                                          │
│  "Đoạn này thật hay giả?"                                          │
│  ┌──────────┐    ┌──────────┐                                       │
│  │ Bonafide │    │  Spoof   │                                       │
│  │  (Thật)  │    │  (Giả)  │                                       │
│  └──────────┘    └────┬─────┘                                       │
│                       │                                              │
│  Level 2: ATTACK TYPE CLASSIFICATION                                 │
│  "Giả bằng phương thức nào?"                                       │
│                       │                                              │
│         ┌─────────────┼─────────────┐                               │
│         ▼             ▼             ▼                               │
│    ┌─────────┐  ┌──────────┐  ┌──────────┐                         │
│    │   TTS   │  │    VC    │  │  Replay  │                         │
│    │ (Tổng  │  │ (Chuyển │  │  (Phát  │                         │
│    │  hợp)  │  │  đổi)   │  │   lại)  │                         │
│    └────┬────┘  └────┬─────┘  └──────────┘                         │
│         │            │                                               │
│  Level 3: MODEL ATTRIBUTION                                         │
│  "Chính xác AI nào đã tạo ra?"                                     │
│         │            │                                               │
│    ┌────┴────────────┴────────────────────┐                         │
│    ▼         ▼       ▼       ▼       ▼    ▼                         │
│  HiFi-  Mel-   Wave-  Wave-  Para-  VITS                           │
│  GAN    GAN    Net    Glow   llel   /Taco                          │
│                              WaveGAN tron                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2.1. Chi tiết từng cấp độ

| Cấp độ | Câu hỏi trả lời | Giá trị pháp y | Độ khó |
|--------|-----------------|----------------|--------|
| **Level 1: Binary** | "Thật hay Giả?" | Sàng lọc ban đầu, loại bỏ bằng chứng giả | ⭐ Dễ |
| **Level 2: Attack Type** | "TTS, VC, hay Replay?" | Thu hẹp phạm vi điều tra, xác định kỹ thuật kẻ tấn công sử dụng | ⭐⭐ Trung bình |
| **Level 3: Model Attribution** | "Đích danh AI/Tool nào?" | **Truy vết nguồn gốc**: xác định công cụ cụ thể → liên kết đến tài khoản mua/tải → xác định nghi phạm | ⭐⭐⭐ Rất khó |

### 1.2.2. Ý nghĩa điều tra của Level 3 — Model Attribution

Level 3 mang giá trị **quyết định** cho công tác truy vết tội phạm không gian mạng:

1. **Mỗi vocoder/generator để lại "dấu vân tay" riêng** trên spectrogram — tương tự dấu vân tay đạn đạo (ballistic fingerprint) trong điều tra vụ nổ súng. Xác định được "khẩu súng" (= AI model) giúp thu hẹp nghi phạm.

2. **Liên kết công cụ → người dùng:** Nếu xác định đoạn deepfake được tạo bởi ElevenLabs API → điều tra viên có thể yêu cầu ElevenLabs cung cấp thông tin tài khoản đã sử dụng API tại thời điểm đó → xác định nghi phạm.

3. **Bằng chứng tòa án:** Kết quả Model Attribution với độ tin cậy cao có thể được sử dụng như **bằng chứng kỹ thuật** (expert witness testimony) tại tòa án.

## 1.3. Ý tưởng cốt lõi: Spectrogram-as-Image — Biến bài toán Âm thanh thành Thị giác Máy tính

Giống như dự án giám sát âm thanh đàn bò, ý tưởng then chốt của đồ án này là **chuyển đổi paradigm**: biến tín hiệu âm thanh 1D thành **ảnh quang phổ 2D** (spectrogram image), từ đó tái sử dụng toàn bộ kho kiến trúc CNN đã được kiểm chứng cho bài toán phân loại ảnh.

**Điểm khác biệt quan trọng:** Trong bài toán deepfake, CNN không tìm kiếm "pattern âm thanh" (tiếng kêu, tiếng ho) mà tìm kiếm **"vết sọc bất thường"** (spectrogram artifacts) — những dấu vết vi mô mà mô hình AI sinh giọng để lại trên ảnh quang phổ, vô hình với tai người nhưng **hiện rõ trên spectrogram**.

---

# CHƯƠNG 2 – KỸ THUẬT XỬ LÝ ẢNH QUANG PHỔ & DẤU VẾT TRÌNH SINH GIỌNG

## 2.1. Chuyển đổi Âm thanh 1D thành Ảnh 2D — Ba phương pháp biểu diễn phổ

### 2.1.1. Mel-Spectrogram (Log-Mel)

Mel-Spectrogram là phương pháp phổ biến nhất, chuyển đổi tín hiệu âm thanh thô thành biểu diễn thời gian-tần số 2D trên thang Mel (mô phỏng cảm nhận thính giác con người).

**Quy trình tính toán:**

```
Raw Waveform (1D)
    │
    ▼
[1] Pre-emphasis: y[n] = x[n] - 0.97·x[n-1]
    │
    ▼
[2] STFT (Short-Time Fourier Transform):
    S(m,k) = Σ x[n + m·H] · w[n] · e^(-j2πkn/N_FFT)
    │
    ▼
[3] Power Spectrum: |S(m,k)|²
    │
    ▼
[4] Mel Filterbank (80 filters, 0-8000 Hz):
    mel(f) = 2595 · log₁₀(1 + f/700)
    │
    ▼
[5] Log compression: log(Mel·|S|² + ε)
    │
    ▼
Output: Ma trận 2D (T × 80) — "Ảnh" đầu vào cho CNN
```

**Tham số tối ưu cho Audio Deepfake Detection:**

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| Sample Rate | 16,000 Hz | Chuẩn cho speech processing, giữ đủ thông tin đến 8 kHz |
| n_fft | 512 hoặc 1024 | Cân bằng giữa phân giải tần số (~31 Hz/bin) và thời gian (~32 ms) |
| hop_length | 160 (= 10 ms) | Phân giải thời gian cao để bắt micro-artifacts |
| n_mels | 80 | Tiêu chuẩn cho speech; phủ toàn dải 0–8 kHz |
| fmin / fmax | 0 / 8000 Hz | Toàn dải phổ speech |
| Window | Hann | Sidelobe suppression tốt |

**Ưu điểm:** Trực quan, nén thông tin hiệu quả, phổ biến → có nhiều pretrained model.  
**Nhược điểm:** Thang Mel nén vùng cao tần → có thể mất micro-artifact ở dải > 4 kHz.

### 2.1.2. LFCC — Linear Frequency Cepstral Coefficients

LFCC sử dụng **thang tần số tuyến tính** (linear-scale filterbank) thay vì thang Mel. Đây là đặc trưng **được khuyến nghị chính thức** bởi ASVspoof challenge [4] vì nó bảo toàn thông tin ở **toàn bộ dải tần số một cách đồng đều**, không ưu tiên vùng thấp như Mel.

**Quy trình tính toán:**

$$\text{LFCC} = \text{DCT}\left(\log\left(\mathbf{H}_{linear} \cdot |S(m,k)|^2\right)\right)$$

Trong đó:
- $\mathbf{H}_{linear}$: Filterbank với các bộ lọc tam giác phân bố **đều** trên trục tần số tuyến tính (không phải thang Mel).
- DCT: Discrete Cosine Transform — giữ lại 20–60 hệ số đầu tiên.

**So sánh LFCC vs Mel-Spectrogram vs MFCC:**

| Đặc trưng | Thang tần số | Số hệ số | Ưu điểm cho Deepfake | Nhược điểm |
|-----------|--------------|----------|----------------------|------------|
| **MFCC** | Mel (logarit) | 13–40 | Tốt cho nhận dạng giọng nói | Nén quá mạnh vùng cao → mất artifact |
| **Mel-Spectrogram** | Mel (logarit) | 80–128 | Giàu thông tin hơn MFCC, trực quan | Vẫn nén vùng cao tần |
| **LFCC** | **Linear (tuyến tính)** | 20–60 | **Bảo toàn đều toàn dải tần** → bắt artifact ở cả vùng cao | Ít phổ biến, ít pretrained |

**Tại sao LFCC quan trọng cho deepfake detection?**

Nhiều vocoder (đặc biệt GAN-based) để lại artifact chủ yếu ở **dải cao tần** (4–8 kHz) — nơi thang Mel nén lại rất mạnh. LFCC với thang tuyến tính **không nén** → bảo toàn đầy đủ các dấu vết này. Đây là lý do ASVspoof challenge chọn LFCC làm baseline feature [4].

### 2.1.3. CQCC — Constant Q Cepstral Coefficients

CQCC sử dụng **Constant Q Transform (CQT)** thay vì STFT. Đặc điểm cốt lõi: phân giải tần số **tỷ lệ nghịch** với tần số — tần số thấp có phân giải cao, tần số cao có phân giải thời gian cao [5].

$$X_{CQ}(k, n) = \sum_{m=0}^{N_k - 1} x[m+n] \cdot w_k[m] \cdot e^{-j2\pi Q m / N_k}$$

Trong đó $N_k = Q \cdot f_s / f_k$ thay đổi theo từng frequency bin $k$, và $Q$ là hằng số (thường = 1/(2^{1/B}-1)$ với $B$ bins/octave).

**CQCC sau đó được tính:**

$$\text{CQCC} = \text{DCT}\left(\log\left(|X_{CQ}|^2\right)\right)$$

**Ưu điểm:** Phân giải thích ứng → bắt tốt cả harmonic structure (tần số thấp) lẫn transient artifacts (tần số cao). Hiệu quả đặc biệt với **replay attack** [5].  
**Nhược điểm:** Tính toán phức tạp hơn STFT, ít công cụ hỗ trợ.

### 2.1.4. Tổng hợp: Chọn đặc trưng nào cho từng cấp độ?

| Cấp độ | Đặc trưng khuyên dùng | Lý do |
|--------|------------------------|-------|
| **Level 1 (Binary)** | LFCC hoặc Mel-Spec | LFCC cho EER thấp nhất trên ASVspoof; Mel-Spec dễ visualize |
| **Level 2 (Attack Type)** | LFCC + CQT (fusion) | CQT tốt cho replay detection; LFCC tốt cho TTS/VC |
| **Level 3 (Model Attribution)** | **Mel-Spectrogram (full-band)** | Cần "nhìn" toàn bộ spectrogram như ảnh để CNN bắt micro-texture của từng vocoder |

## 2.2. CNN xử lý Spectrogram như "Bức ảnh" — Cơ chế hoạt động

Khi spectrogram (2D) được đưa vào CNN, mạng xử lý nó **chính xác như một bức ảnh grayscale**:

```
┌──────────────────────────────────────────────────────────────────┐
│     CNN "NHÌN" SPECTROGRAM NHƯ ẢNH — CÁC LỚP TRÍCH XUẤT       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Spectrogram Input (1 × 80 × T)                                 │
│  ┌──────────────────────┐                                        │
│  │ ═══════ harmonics    │                                        │
│  │ ░░░ artifact vùng    │                                        │
│  │     cao tần          │                                        │
│  │ ═══════ F0           │                                        │
│  └──────────────────────┘                                        │
│            │                                                      │
│  Layer 1-2 (Conv 3×3): Bắt MICRO-TEXTURE                        │
│  → Phát hiện "sọc" bất thường, gaps, over-smoothing             │
│  → Tương tự edge detection trong ảnh thường                     │
│            │                                                      │
│  Layer 3-4 (Conv 3×3): Bắt PATTERN CỤC BỘ                      │
│  → Kết hợp micro-texture thành đặc trưng vùng                  │
│  → "Vùng này quá mượt", "Vùng này có sọc lặp"                  │
│            │                                                      │
│  Layer 5+ (Conv + Pooling): Bắt CẤU TRÚC TOÀN CỤC              │
│  → Nhận diện pattern đặc trưng của từng vocoder                 │
│  → "Bức ảnh này có dấu vân tay của HiFi-GAN"                   │
│            │                                                      │
│  FC + Sigmoid/Softmax: QUYẾT ĐỊNH                               │
│  → Level 1: Thật/Giả                                            │
│  → Level 2: TTS/VC/Replay                                       │
│  → Level 3: HiFi-GAN / MelGAN / WaveNet / ...                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Insight then chốt:** CNN trong deepfake detection không học "nội dung lời nói" (speaker identity, linguistic content) mà học **"texture vật lý"** của spectrogram — tương tự cách CNN phân biệt ảnh thật vs ảnh GAN bằng texture analysis trong Computer Vision. Đây chính xác là bài toán **Image Forensics** trên "ảnh âm thanh".

## 2.3. Dấu vết Quang phổ của AI Sinh Giọng (Spectrogram Artifacts)

### 2.3.1. Bốn loại dấu vết chính

Mỗi trình sinh giọng (vocoder/generator) do kiến trúc mạng nơ-ron và hàm mất mát khác nhau, sẽ để lại **các loại dấu vết đặc trưng** trên spectrogram:

#### a) Over-smoothing ở dải cao tần

**Bản chất:** Các mô hình sinh giọng (đặc biệt autoregressive như WaveNet, Tacotron) tối ưu bằng cách **cực tiểu hóa sai số trung bình** (MSE/MAE). Ở dải cao tần (> 4 kHz), tín hiệu speech rất phức tạp và nhiễu → mô hình "chọn an toàn" bằng cách sinh ra giá trị **trung bình** → mất chi tiết (detail loss).

```
Giọng THẬT:                    Giọng GIẢ (Over-smoothed):
┌──────────────────┐           ┌──────────────────┐
│ ░▓░▒▓░▒░▓▒░▓░▒▓ │ > 4kHz   │ ░░░░░░░░░░░░░░░░ │ > 4kHz   ← MỜ, PHẲNG
│ ═══╤═══╤═══╤═══ │           │ ═══╤═══╤═══╤═══ │
│ ═══╤═══╤═══╤═══ │           │ ═══╤═══╤═══╤═══ │
│ ════════════════ │ < 1kHz   │ ════════════════ │ < 1kHz
└──────────────────┘           └──────────────────┘
    Vùng cao tần "sống động"       Vùng cao tần "chết", quá mượt
```

**Vocoder đặc trưng:** WaveNet, Tacotron2, WaveRNN.

#### b) Spectral Gaps (Khoảng trống phổ)

**Bản chất:** Các vocoder GAN-based sử dụng **upsampling layers** (transposed convolution / sub-pixel convolution) để tăng sampling rate. Mỗi tầng upsampling có thể tạo ra **"khoảng trống"** (gaps) hoặc **"sọc lặp"** (periodic patterns) tại tần số tương ứng với tỷ lệ upsampling.

```
Giọng THẬT:                    Giọng GIẢ (Spectral Gap):
┌──────────────────┐           ┌──────────────────┐
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │           │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │           │ ░░░░░░░░░░░░░░░░ │ ← GAP (mất năng lượng)
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │           │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │           │ ░░░░░░░░░░░░░░░░ │ ← GAP
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │           │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
└──────────────────┘           └──────────────────┘
    Phổ liên tục                   Sọc lặp theo tỷ lệ upsampling
```

**Vocoder đặc trưng:** MelGAN (upsampling ratio 8×8×2×2 → gaps ở ~2kHz, ~4kHz), HiFi-GAN (ít hơn nhờ Multi-Period Discriminator).

#### c) Phase Mismatch (Lệch pha)

**Bản chất:** Hầu hết pipeline TTS/VC hoạt động trên **magnitude spectrogram** (bỏ qua phase). Khi tái tạo waveform, phase được ước lượng bằng Griffin-Lim hoặc vocoder → **không khớp chính xác** với phase tự nhiên. Trên spectrogram, lệch pha biểu hiện dưới dạng:
- **"Smearing"** (nhòe) tại các transient (onset phụ âm: /t/, /k/, /p/)
- **Phase discontinuity** giữa các frame STFT liền kề

**Vocoder đặc trưng:** Tất cả vocoder đều có ở mức độ khác nhau; WaveGlow và Griffin-Lim rõ nhất.

#### d) Harmonic Distortion (Méo hài âm)

**Bản chất:** Ở voiced speech (nguyên âm), phổ tự nhiên có **cấu trúc harmonic rõ ràng** — các đường song song ở bội số của F0 (100, 200, 300, ... Hz). Vocoder GAN-based có thể tạo ra:
- **Sub-harmonics giả** (harmonics ở tần số không phải bội F0)
- **Missing harmonics** (mất một số hài âm)
- **Harmonic jitter** (hài âm dao động bất thường)

```
Giọng THẬT:                    Giọng GIẢ (Harmonic Distortion):
┌──────────────────┐           ┌──────────────────┐
│ ═══════════════  │ 5×F0     │ ═══════════════  │ 5×F0
│                  │           │ ─ ─ ─ ─ ─ ─ ─ ─ │ ← Sub-harmonic giả!
│ ═══════════════  │ 4×F0     │ ═══════════════  │ 4×F0
│ ═══════════════  │ 3×F0     │                  │ ← Missing harmonic!
│ ═══════════════  │ 2×F0     │ ═══╤══╤═══╤══╤══ │ 2×F0 ← Jitter!
│ ═══════════════  │ F0       │ ═══════════════  │ F0
└──────────────────┘           └──────────────────┘
    Harmonics đều, ổn định         Harmonics bất thường
```

**Vocoder đặc trưng:** MelGAN (rõ nhất), Parallel WaveGAN, HiFi-GAN (ít hơn).

### 2.3.2. Dấu vết đặc trưng của 6 Trình sinh Giọng (Vocoder Fingerprints)

#### 1. WaveNet (Google DeepMind, 2016)

| Đặc điểm | Chi tiết |
|-----------|----------|
| **Kiến trúc** | Autoregressive, dilated causal convolution |
| **Nguyên lý sinh** | Sinh từng sample $x_t$ dựa trên tất cả sample trước đó $x_{<t}$ |
| **Dấu vết chính** | **Over-smoothing rõ rệt ở > 4 kHz**: do mô hình tối ưu log-likelihood, regression-to-mean ở vùng phức tạp |
| **Dấu vết phụ** | Thiếu micro-variation tự nhiên (tremor, jitter) ở F0 → F0 quá "hoàn hảo" |
| **Spectral signature** | Gradient năng lượng giảm đều ở cao tần (smooth roll-off) thay vì irregular như tự nhiên |
| **Độ khó phát hiện** | ⭐⭐ Trung bình — over-smoothing khá rõ |

#### 2. HiFi-GAN (Kong et al., 2020)

| Đặc điểm | Chi tiết |
|-----------|----------|
| **Kiến trúc** | GAN: Generator (upsampling + residual block) + Multi-Period Discriminator (MPD) + Multi-Scale Discriminator (MSD) |
| **Nguyên lý sinh** | Non-autoregressive, upsampling mel-spec → waveform |
| **Dấu vết chính** | **Periodic micro-artifacts** tương ứng với chu kỳ trong MPD (periods = [2,3,5,7,11]): sọc rất mảnh lặp lại theo chu kỳ trên spectrogram |
| **Dấu vết phụ** | Chất lượng rất cao → dấu vết vi mô, khó thấy bằng mắt; cần CNN sâu để bắt |
| **Spectral signature** | Texture "quá đều" (over-regular) so với tự nhiên — thiếu stochastic variation |
| **Độ khó phát hiện** | ⭐⭐⭐ Khó — chất lượng cao, artifact rất tinh vi |

#### 3. MelGAN (Kumar et al., 2019)

| Đặc điểm | Chi tiết |
|-----------|----------|
| **Kiến trúc** | GAN: Generator (stack of transposed conv, upsampling ratio 8×8×2×2) + Multi-Scale Discriminator |
| **Nguyên lý sinh** | Non-autoregressive, feed-forward |
| **Dấu vết chính** | **Spectral Gaps rõ ràng** ở ~2 kHz và ~4 kHz (tương ứng upsampling boundary 8× và 8×8×) + **"metallic buzz"** (tiếng kim loại) |
| **Dấu vết phụ** | Harmonic distortion rõ ràng: sub-harmonics giả, missing harmonics ở voiced segments |
| **Spectral signature** | Sọc ngang tuần hoàn (periodic horizontal bands) + vùng cao tần bị suy giảm bất thường |
| **Độ khó phát hiện** | ⭐ Dễ — artifact rõ ràng, dễ bắt |

#### 4. WaveGlow (NVIDIA, 2019)

| Đặc điểm | Chi tiết |
|-----------|----------|
| **Kiến trúc** | Flow-based generative model (Glow + WaveNet affine coupling layers) |
| **Nguyên lý sinh** | Biến đổi khả nghịch (invertible transform) từ Gaussian noise → waveform |
| **Dấu vết chính** | **"Ghost harmonics"** — harmonics mờ nhạt xuất hiện ở tần số không đúng vị trí, do flow-based transform tạo ra ánh xạ không hoàn hảo |
| **Dấu vết phụ** | Phase artifacts: onset phụ âm bị "nhòe" (smeared) do phase estimation không chính xác |
| **Spectral signature** | Tổng thể smooth nhưng có "rung nhẹ" (ripple) bất thường ở envelope phổ |
| **Độ khó phát hiện** | ⭐⭐ Trung bình — ghost harmonics là dấu hiệu đặc trưng |

#### 5. Parallel WaveGAN (Yamamoto et al., 2020)

| Đặc điểm | Chi tiết |
|-----------|----------|
| **Kiến trúc** | Non-autoregressive WaveNet + GAN training + **Multi-resolution STFT Loss** |
| **Nguyên lý sinh** | WaveNet-like nhưng sinh song song (không từng sample), train bằng adversarial + spectral loss |
| **Dấu vết chính** | **Spectral inconsistency** tại biên giữa các resolution STFT khác nhau (256, 512, 1024 FFT points): spectrogram "khớp" ở 1 resolution nhưng "lệch" ở resolution khác |
| **Dấu vết phụ** | Ít over-smoothing hơn WaveNet gốc nhờ GAN loss, nhưng có subtle periodic pattern từ discriminator |
| **Spectral signature** | Đa phân giải STFT cho thấy inconsistency → cần phân tích **multi-resolution** để bắt |
| **Độ khó phát hiện** | ⭐⭐⭐ Khó — cần multi-resolution analysis |

#### 6. Tacotron2 / VITS (End-to-End TTS)

| Đặc điểm | Chi tiết |
|-----------|----------|
| **Kiến trúc** | **Tacotron2:** Seq2seq attention (text → mel-spec) + vocoder (WaveNet/HiFi-GAN). **VITS:** End-to-end VAE + Normalizing Flow + HiFi-GAN decoder |
| **Nguyên lý sinh** | Tacotron2: 2 bước (text→mel→wave). VITS: 1 bước end-to-end (text→wave) |
| **Dấu vết chính** | **Attention artifacts:** Tacotron2 để lại dấu vết từ cơ chế attention — "lỗi alignment" (mispronunciation, word skipping, repetition) tạo ra đoạn spectrogram **bất thường về thời gian** (duration anomaly). VITS: ít hơn nhờ Monotonic Alignment Search |
| **Dấu vết phụ** | **Dấu vết kép** (Tacotron2): artifact từ acoustic model + artifact từ vocoder chồng lên nhau → đặc trưng riêng biệt |
| **Spectral signature** | Duration không tự nhiên (quá đều hoặc quá rời rạc) + prosody "phẳng" hơn giọng thật |
| **Độ khó phát hiện** | ⭐⭐ Trung bình (Tacotron2) / ⭐⭐⭐ Khó (VITS) |

### 2.3.3. Bảng tổng hợp dấu vết 6 Vocoder

| Vocoder | Over-smoothing | Spectral Gaps | Phase Mismatch | Harmonic Distortion | Dấu vết đặc trưng nhất |
|---------|:-:|:-:|:-:|:-:|---|
| **WaveNet** | 🔴 Rõ | ⚪ Không | 🟡 Nhẹ | ⚪ Không | High-freq roll-off quá smooth |
| **HiFi-GAN** | 🟡 Nhẹ | 🟡 Nhẹ | 🟡 Nhẹ | 🟡 Nhẹ | Periodic micro-artifact (MPD periods) |
| **MelGAN** | 🟡 Nhẹ | 🔴 Rõ | 🟡 Nhẹ | 🔴 Rõ | Sọc tuần hoàn + metallic buzz |
| **WaveGlow** | 🟡 Trung bình | ⚪ Không | 🔴 Rõ | 🟡 Nhẹ | Ghost harmonics + phase smearing |
| **Parallel WaveGAN** | 🟡 Nhẹ | 🟡 Nhẹ | 🟡 Nhẹ | 🟡 Nhẹ | Multi-resolution STFT inconsistency |
| **Tacotron2/VITS** | 🟡 Trung bình | 🟡 Nhẹ | 🟡 Nhẹ | 🟡 Nhẹ | Duration anomaly + attention artifact |

---

# CHƯƠNG 3 – MA TRẬN NHÃN 3 CẤP ĐỘ (HIERARCHICAL LABEL MATRIX)

## 3.1. Thiết kế Hệ thống Phân lớp Phân cấp (Hierarchical Classification)

Hệ thống sử dụng kiến trúc **Hierarchical Classification** — mỗi cấp độ kế thừa và chi tiết hóa cấp độ trước:

### 3.1.1. Cấp độ 1 — Binary Detection (Nhị phân)

| Class ID | Nhãn | Mô tả | Ứng dụng |
|----------|------|-------|----------|
| 0 | **Bonafide** (Thật) | Giọng nói tự nhiên, thu trực tiếp từ người nói | Sàng lọc ban đầu |
| 1 | **Spoof** (Giả mạo) | Giọng nói được tạo/biến đổi bởi AI hoặc phát lại | Cảnh báo giả mạo |

**Activation:** Sigmoid → $p \in [0, 1]$ → ngưỡng 0.5  
**Loss:** BCEWithLogitsLoss

### 3.1.2. Cấp độ 2 — Attack Type Classification (Phương thức tấn công)

| Class ID | Nhãn | Kỹ thuật | Đặc trưng phổ | Nguồn Dataset |
|----------|------|----------|---------------|---------------|
| 0 | **Bonafide** | Giọng thật | Phổ tự nhiên, stochastic variation | ASVspoof LA/DF |
| 1 | **TTS** (Text-to-Speech) | Tổng hợp giọng từ văn bản | Over-smoothing, duration anomaly | ASVspoof LA |
| 2 | **VC** (Voice Conversion) | Chuyển đổi giọng A → giọng B | Spectral envelope mismatch, F0 distortion | ASVspoof LA |
| 3 | **Replay** | Thu giọng thật → phát lại qua loa/mic | Channel distortion, room reverb, compression artifact | ASVspoof PA |

**Activation:** Softmax → $\sum p_i = 1$  
**Loss:** CrossEntropyLoss

### 3.1.3. Cấp độ 3 — Model Attribution (Định danh AI)

| Class ID | Nhãn | Vocoder/System | Attack Type | Nguồn Dataset |
|----------|------|---------------|-------------|---------------|
| 0 | **Bonafide** | Giọng người thật | — | ASVspoof + WaveFake |
| 1 | **HiFi-GAN** | HiFi-GAN (V1/V2/V3) | TTS/VC | WaveFake |
| 2 | **MelGAN** | MelGAN / Multi-band MelGAN | TTS/VC | WaveFake |
| 3 | **WaveNet** | WaveNet / WaveRNN | TTS | WaveFake |
| 4 | **WaveGlow** | WaveGlow (NVIDIA) | TTS | WaveFake |
| 5 | **Parallel WaveGAN** | Parallel WaveGAN | TTS | WaveFake |
| 6 | **VITS / Tacotron** | VITS, Tacotron2, Full-band MelGAN | TTS | WaveFake |

**Activation:** Softmax → $\sum p_i = 1$  
**Loss:** CrossEntropyLoss + Center Loss (tùy chọn)

### 3.1.4. Bảng Ma trận Nhãn Tổng hợp (Full Label Matrix)

| Mẫu Audio | Level 1 | Level 2 | Level 3 |
|-----------|---------|---------|---------|
| Người nói trực tiếp | Bonafide (0) | Bonafide (0) | Bonafide (0) |
| Tacotron2 + WaveNet | Spoof (1) | TTS (1) | WaveNet (3) |
| Tacotron2 + HiFi-GAN | Spoof (1) | TTS (1) | HiFi-GAN (1) |
| VITS (end-to-end) | Spoof (1) | TTS (1) | VITS/Tacotron (6) |
| StarGAN-VC + MelGAN | Spoof (1) | VC (2) | MelGAN (2) |
| AutoVC + WaveGlow | Spoof (1) | VC (2) | WaveGlow (4) |
| Phát lại qua loa | Spoof (1) | Replay (3) | N/A (không có vocoder) |

## 3.2. Mối quan hệ Phân cấp giữa 3 Cấp độ

```
                    ┌───────────────────────┐
                    │     Input Audio        │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Level 1: Binary       │
                    │  Bonafide vs Spoof     │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │      Spoof?            │
                    │   ┌───┴───┐            │
                    │  YES     NO → STOP     │
                    │   │                    │
                    └───┼────────────────────┘
                        │
                    ┌───▼───────────────────┐
                    │  Level 2: Attack Type  │
                    │  TTS / VC / Replay     │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  TTS hoặc VC?          │
                    │   ┌───┴───┐            │
                    │  YES     Replay→STOP   │
                    │   │                    │
                    └───┼────────────────────┘
                        │
                    ┌───▼───────────────────┐
                    │  Level 3: Attribution  │
                    │  HiFi-GAN / MelGAN /   │
                    │  WaveNet / WaveGlow /   │
                    │  PWGAN / VITS           │
                    └───────────────────────┘
```

**Lưu ý thiết kế:**
- Level 3 chỉ áp dụng khi Level 2 = TTS hoặc VC (Replay attack không có vocoder → không cần attribution).
- Trong kiến trúc Multi-task, cả 3 level được tính **đồng thời** (không tuần tự) → tiết kiệm thời gian inference.

---

# CHƯƠNG 4 – KIẾN TRÚC MÔ HÌNH AI ĐA NHIỆM

## 4.1. Tổng quan Pipeline Đa nhiệm (Multi-Task Architecture)

```
┌─────────────────────────────────────────────────────────────────────┐
│              MULTI-TASK DEEP LEARNING PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: 2D Spectrogram / LFCC Image                                │
│  ┌──────────────────────┐                                           │
│  │  (1, 80, T) or       │                                           │
│  │  (1, 60, T)          │                                           │
│  └──────────┬───────────┘                                           │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────────────────┐                               │
│  │    SHARED FEATURE EXTRACTOR      │                               │
│  │    (LCNN / ResNet18 / RawNet2)   │                               │
│  │                                  │                               │
│  │    Conv → MFM → Conv → MFM →    │                               │
│  │    MaxPool → ... → GAP           │                               │
│  │                                  │                               │
│  │    Output: Feature Vector (D)    │                               │
│  └──────────┬───────────────────────┘                               │
│             │                                                        │
│         ┌───┴────────────┬──────────────┐                           │
│         │                │              │                            │
│         ▼                ▼              ▼                            │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────┐                 │
│  │  HEAD 1     │  │  HEAD 2    │  │  HEAD 3      │                 │
│  │  Binary     │  │  Attack    │  │  Model       │                 │
│  │  Detection  │  │  Type      │  │  Attribution │                 │
│  │             │  │            │  │              │                 │
│  │  FC → 1     │  │  FC → 4    │  │  FC → 7      │                 │
│  │  Sigmoid    │  │  Softmax   │  │  Softmax     │                 │
│  │             │  │            │  │              │                 │
│  │ Bonafide/   │  │ Bona/TTS/  │  │ Bona/HiFi/  │                 │
│  │ Spoof       │  │ VC/Replay  │  │ Mel/WN/WG/  │                 │
│  │             │  │            │  │ PWG/VITS     │                 │
│  └─────────────┘  └────────────┘  └──────────────┘                 │
│                                                                     │
│  LOSS = α·L₁(BCE) + β·L₂(CE) + γ·L₃(CE) + δ·L_center             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 4.2. Shared Feature Extractor — LCNN (Light CNN) với Max-Feature-Map

### 4.2.1. Tại sao chọn LCNN?

**LCNN** (Light Convolutional Neural Network) được đề xuất bởi Wu et al. (2018) [6] và trở thành **baseline architecture** của ASVspoof challenge. Ưu điểm:

1. **Nhẹ** (ít tham số hơn ResNet nhiều lần) → train nhanh, ít overfit.
2. **Max-Feature-Map (MFM)** thay ReLU → hiệu quả hơn cho bài toán forensics.
3. **Đã được kiểm chứng** trên ASVspoof 2019, 2021 với EER rất thấp.

### 4.2.2. Max-Feature-Map (MFM) — Activation đặc biệt

MFM là kỹ thuật activation **cạnh tranh** (competitive activation) thay thế ReLU:

$$\text{MFM}(x) = \max(x_1, x_2)$$

Trong đó input $x$ có $2N$ channels được chia thành 2 nhóm $x_1, x_2$ (mỗi nhóm $N$ channels), và MFM lấy **max element-wise** giữa 2 nhóm.

**Ý nghĩa:** MFM buộc mạng phải "cạnh tranh" giữa 2 bộ filter → tự động chọn feature map hữu ích nhất → loại bỏ noise, giữ lại discriminative features. Đặc biệt hiệu quả cho bài toán phân biệt thật/giả nơi sự khác biệt rất tinh vi.

**So sánh MFM vs ReLU:**

| Activation | Công thức | Output channels | Ưu điểm |
|-----------|-----------|-----------------|---------|
| ReLU | max(0, x) | N | Đơn giản, phổ biến |
| **MFM** | max(x₁, x₂) | N/2 | **Feature selection tự động**, giảm 50% channels → compact |

### 4.2.3. Kiến trúc LCNN chi tiết

| Layer | Type | Kernel | Channels In → Out (sau MFM) | Output |
|-------|------|--------|-----------------------------|--------|
| Input | — | — | 1 | (1, 80, T) |
| Conv1 | Conv2D + MFM | 5×5, stride 1 | 1 → 64 → **32** | (32, 80, T) |
| MaxPool1 | MaxPool2D | 2×2 | 32 | (32, 40, T/2) |
| Conv2 | Conv2D + MFM | 1×1 | 32 → 64 → **32** | (32, 40, T/2) |
| Conv3 | Conv2D + BN + MFM | 3×3 | 32 → 96 → **48** | (48, 40, T/2) |
| MaxPool2 | MaxPool2D | 2×2 | 48 | (48, 20, T/4) |
| Conv4 | Conv2D + MFM | 1×1 | 48 → 96 → **48** | (48, 20, T/4) |
| Conv5 | Conv2D + BN + MFM | 3×3 | 48 → 128 → **64** | (64, 20, T/4) |
| MaxPool3 | MaxPool2D | 2×2 | 64 | (64, 10, T/8) |
| Conv6 | Conv2D + MFM | 1×1 | 64 → 128 → **64** | (64, 10, T/8) |
| Conv7 | Conv2D + BN + MFM | 3×3 | 64 → 64 → **32** | (32, 10, T/8) |
| MaxPool4 | MaxPool2D | 2×2 | 32 | (32, 5, T/16) |
| **GAP** | Global Average Pool | — | 32 | **(32,)** |
| **FC-shared** | Linear + MFM | — | 32 → 128 → **64** | **(64,)** |

**Feature vector kích thước 64** → chia cho 3 classification heads.

### 4.2.4. Backbone thay thế: ResNet18 và RawNet2

| Backbone | Params | Input | Ưu điểm | Nhược điểm |
|----------|--------|-------|---------|------------|
| **LCNN** | ~0.3M | Spectrogram/LFCC 2D | Nhẹ, MFM hiệu quả cho forensics, ASVspoof baseline | Capacity hạn chế cho Level 3 |
| **ResNet18** | ~11M | Spectrogram 2D | Capacity lớn, pretrained ImageNet, residual learning | Nặng hơn, có thể overfit |
| **RawNet2** | ~1M | **Raw waveform 1D** | Học trực tiếp từ waveform, không cần feature extraction | Cần GPU mạnh, khó interpret |

**Gợi ý:** Bắt đầu với **LCNN** (nhẹ, nhanh), sau đó thử **ResNet18** nếu cần tăng accuracy Level 3.

## 4.3. Multi-Head Output — 3 Đầu ra Phân loại

```python
class MultiTaskHead(nn.Module):
    def __init__(self, feature_dim=64):
        # Head 1: Binary Detection (Bonafide vs Spoof)
        self.head_binary = nn.Linear(feature_dim, 1)      # Sigmoid output
        
        # Head 2: Attack Type (Bonafide/TTS/VC/Replay)
        self.head_attack = nn.Linear(feature_dim, 4)       # Softmax output
        
        # Head 3: Model Attribution (7 classes)
        self.head_model = nn.Linear(feature_dim, 7)        # Softmax output
```

Cả 3 heads **chia sẻ chung Feature Extractor** (LCNN backbone) → shared representation learning. Gradient từ cả 3 loss cùng cập nhật backbone → backbone học được features **tổng quát** hơn so với train từng task riêng lẻ.

## 4.4. Hàm Mất mát Tổng hợp (Combined Loss Function)

### 4.4.1. Loss cho từng cấp

**Level 1 — BCEWithLogitsLoss (Binary Cross-Entropy):**

$$\mathcal{L}_1 = -\left[ y \cdot \log(\sigma(z)) + (1-y) \cdot \log(1-\sigma(z)) \right]$$

Trong đó $y \in \{0, 1\}$ (bonafide = 0, spoof = 1), $z$ là logit, $\sigma$ là Sigmoid.

**Level 2 — CrossEntropyLoss:**

$$\mathcal{L}_2 = -\sum_{c=0}^{3} y_c \cdot \log\left(\frac{e^{z_c}}{\sum_{k=0}^{3} e^{z_k}}\right)$$

**Level 3 — CrossEntropyLoss:**

$$\mathcal{L}_3 = -\sum_{c=0}^{6} y_c \cdot \log\left(\frac{e^{z_c}}{\sum_{k=0}^{6} e^{z_k}}\right)$$

### 4.4.2. Center Loss — Tăng độ tụ cụm đặc trưng

**Center Loss** (Wen et al., 2016) [7] kéo các feature vector cùng class **gần lại tâm** của class đó trong không gian embedding, giúp CNN học biểu diễn **tụ cụm chặt** (compact clusters) → cải thiện đáng kể Level 3 nơi các vocoder có đặc trưng tương tự nhau.

$$\mathcal{L}_{center} = \frac{1}{2} \sum_{i=1}^{B} \| f_i - c_{y_i} \|_2^2$$

Trong đó:
- $f_i$ là feature vector của sample $i$ (output của shared backbone)
- $c_{y_i}$ là center (tâm) của class $y_i$, được cập nhật online mỗi mini-batch
- $B$ là batch size

**Cập nhật center:**
$$\Delta c_j = \frac{\sum_{i=1}^{B} \mathbb{1}(y_i = j) \cdot (c_j - f_i)}{1 + \sum_{i=1}^{B} \mathbb{1}(y_i = j)}$$

$$c_j \leftarrow c_j - \lambda_{center} \cdot \Delta c_j$$

Với $\lambda_{center} = 0.5$ (learning rate cho center).

### 4.4.3. Tổng hợp Loss

$$\boxed{\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_1 + \beta \cdot \mathcal{L}_2 + \gamma \cdot \mathcal{L}_3 + \delta \cdot \mathcal{L}_{center}}$$

**Trọng số khuyến nghị:**

| Trọng số | Giá trị | Lý do |
|----------|---------|-------|
| $\alpha$ | 1.0 | Binary detection là task chính, ưu tiên cao nhất |
| $\beta$ | 0.5 | Attack type classification — task phụ trợ |
| $\gamma$ | 0.5 | Model attribution — task phụ trợ |
| $\delta$ | 0.01 | Center loss — regularizer, không nên quá lớn |

**Lưu ý:** Trọng số có thể được **tune** bằng grid search hoặc uncertainty-based weighting (Kendall et al., 2018).

---

# CHƯƠNG 5 – TẬP DỮ LIỆU & METRIC ĐÁNH GIÁ

## 5.1. Tập dữ liệu

### 5.1.1. ASVspoof 2019 LA (Logical Access)

| Thuộc tính | Chi tiết |
|-----------|----------|
| **Mục đích** | Phát hiện TTS và VC attacks |
| **Bonafide** | ~5,000 utterances (train) + ~5,000 (dev) + ~7,000 (eval) |
| **Spoof** | 6 TTS systems (A01–A06) + 11 VC systems (A07–A17) |
| **Tổng** | ~120,000 utterances |
| **Sử dụng cho** | **Level 1** (Binary) + **Level 2** (TTS vs VC) |
| **Format** | FLAC, 16 kHz, 16-bit |

### 5.1.2. ASVspoof 2021 DF (Deepfake)

| Thuộc tính | Chi tiết |
|-----------|----------|
| **Mục đích** | Phát hiện deepfake speech trong điều kiện "in the wild" |
| **Đặc điểm** | Audio đã qua nén (codec), truyền qua kênh (telephone), thêm noise → thực tế hơn 2019 |
| **Sử dụng cho** | **Level 1** (đánh giá robustness) |

### 5.1.3. WaveFake Dataset (Frank & Schönherr, 2021)

| Thuộc tính | Chi tiết |
|-----------|----------|
| **Mục đích** | Nghiên cứu Model Attribution — chứa audio từ **6 vocoder** riêng biệt |
| **Nguồn gốc** | Sinh từ LJSpeech (single-speaker English) |
| **6 Vocoders** | HiFi-GAN, MelGAN, Multi-band MelGAN, Parallel WaveGAN, WaveGlow, Full-band MelGAN |
| **Bonafide** | LJSpeech gốc (~13,000 utterances) |
| **Spoof** | ~13,000 utterances × 6 vocoders = ~78,000 |
| **Tổng** | **~91,000 utterances** |
| **Sử dụng cho** | **Level 3** (Model Attribution) |
| **Format** | WAV, 22,050 Hz |

### 5.1.4. Chiến lược kết hợp Dataset

| Cấp độ | Dataset chính | Dataset bổ sung | Train/Val/Test |
|--------|--------------|-----------------|----------------|
| **Level 1 (Binary)** | ASVspoof 2019 LA | ASVspoof 2021 DF (eval) | Theo protocol chuẩn ASVspoof |
| **Level 2 (Attack Type)** | ASVspoof 2019 LA (tách TTS/VC) | WaveFake (gán nhãn TTS) | 70/15/15 |
| **Level 3 (Attribution)** | WaveFake (6 vocoders) | — | 70/15/15 |

## 5.2. Metric Đánh giá

### 5.2.1. Level 1 (Binary Detection)

**EER — Equal Error Rate:**

EER là điểm trên đường cong DET (Detection Error Tradeoff) nơi **False Acceptance Rate (FAR) = False Rejection Rate (FRR)**.

$$\text{EER} = \text{FAR} = \text{FRR} \quad \text{tại ngưỡng tối ưu } \theta^*$$

- FAR = Tỉ lệ mẫu giả bị phân loại nhầm thành thật.
- FRR = Tỉ lệ mẫu thật bị phân loại nhầm thành giả.
- **EER càng thấp càng tốt.** SOTA trên ASVspoof 2019 LA: EER ~ 0.5–2%.

**min t-DCF — minimum tandem Detection Cost Function:**

t-DCF đo chi phí tổng hợp khi hệ thống countermeasure (CM) hoạt động cùng với hệ thống ASV:

$$\text{t-DCF} = C_{miss} \cdot P_{miss}^{cm} \cdot P_{target} + C_{fa} \cdot P_{fa}^{cm} \cdot (1 - P_{target})$$

Trong đó $C_{miss}, C_{fa}$ là cost factors, $P_{target}$ là prior probability. min t-DCF là giá trị nhỏ nhất trên toàn bộ ngưỡng. **min t-DCF càng thấp càng tốt.**

### 5.2.2. Level 2 & 3 (Multi-class Classification)

| Metric | Công thức | Ý nghĩa |
|--------|----------|---------|
| **Accuracy** | $\frac{\text{Correct}}{\text{Total}}$ | Tổng quan, nhưng bị ảnh hưởng bởi class imbalance |
| **Precision** (per-class) | $\frac{TP}{TP + FP}$ | "Khi model nói class X, đúng bao nhiêu %" |
| **Recall** (per-class) | $\frac{TP}{TP + FN}$ | "Trong tất cả mẫu class X, model bắt được bao nhiêu %" |
| **F1-score** (per-class) | $\frac{2 \cdot P \cdot R}{P + R}$ | Cân bằng giữa Precision và Recall |
| **Macro F1** | $\frac{1}{C}\sum_{c=1}^{C} F1_c$ | Trung bình F1 trên tất cả class (coi mọi class đều quan trọng) |
| **Confusion Matrix** | Ma trận C×C | Hiển thị chi tiết mẫu nào bị nhầm thành class nào |

**Mục tiêu:**

| Cấp độ | Metric chính | Mục tiêu |
|--------|-------------|----------|
| Level 1 | EER | ≤ 3% trên ASVspoof 2019 LA eval |
| Level 1 | min t-DCF | ≤ 0.08 |
| Level 2 | Macro F1 | ≥ 85% |
| Level 3 | Macro F1 | ≥ 80% (6 vocoders + bonafide) |
| Level 3 | Confusion Matrix | Không có cặp vocoder nào bị nhầm > 10% |

---

# CHƯƠNG 6 – LỘ TRÌNH THỰC HIỆN CHI TIẾT (MASTER ROADMAP)

## 6.1. Tổng quan 5 Giai đoạn

```
┌───────────────────────────────────────────────────────────────────┐
│                    MASTER ROADMAP — 5 GIAI ĐOẠN                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 1        Phase 2        Phase 3        Phase 4    Phase 5  │
│  DATA &         LEVEL 1 & 2    LEVEL 3        MULTI-     UI &    │
│  PREPROCESS     MODEL DEV      ATTRIBUTION    TASK       DEPLOY  │
│                                                FUSION            │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌────────┐ ┌───────┐│
│  │ ASVspoof│   │ LCNN    │   │ Vocoder │   │Combined│ │Gradio/││
│  │ WaveFake│──▶│ Binary  │──▶│ Finger- │──▶│ Loss   │▶│Stream-││
│  │ LFCC/Mel│   │ + Type  │   │ print   │   │ Train  │ │lit    ││
│  └─────────┘   └─────────┘   └─────────┘   └────────┘ └───────┘│
│                                                                   │
│  Tuần 1-3       Tuần 4-7       Tuần 8-11     Tuần 12-14 Tuần 15 │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

## 6.2. Giai đoạn 1: Chuẩn bị Data & Tiền xử lý (Tuần 1–3)

### Tuần 1: Tải và Khám phá dữ liệu

| Công việc | Chi tiết | Output |
|-----------|----------|--------|
| Tải ASVspoof 2019 LA | Từ trang chính thức ASVspoof hoặc Edinburgh DataShare | `data/asvspoof2019/` |
| Tải WaveFake | Từ GitHub/Zenodo (Frank & Schönherr) | `data/wavefake/` |
| Phân tích EDA | Thống kê số mẫu/class, duration, sample rate | Notebook `01_eda.ipynb` |
| Nghe thử + visualize | So sánh spectrogram bonafide vs từng loại spoof | Notebook `02_visualize.ipynb` |

### Tuần 2–3: Xây dựng Pipeline Tiền xử lý

```
Raw Audio (FLAC/WAV)
    │
    ▼
[1] Resample → 16 kHz (thống nhất)
    │
    ▼
[2] Trim silence / Pad to fixed length (4 giây = 64,000 samples)
    │
    ▼
[3] Trích xuất đặc trưng (chọn 1 hoặc fusion):
    ├── LFCC (60 coefficients × T frames) → (1, 60, T)
    ├── Mel-Spectrogram (80 bins × T frames) → (1, 80, T)
    └── CQCC (60 coefficients × T frames) → (1, 60, T)
    │
    ▼
[4] Normalization (per-utterance z-score)
    │
    ▼
[5] Save as .pt (PyTorch tensor) hoặc .npy (NumPy)
```

**Cấu trúc thư mục:**

```
audio deepfake/
├── data/
│   ├── asvspoof2019/           ← Dataset gốc
│   ├── wavefake/               ← Dataset gốc
│   └── processed/
│       ├── lfcc/               ← LFCC tensors
│       └── melspec/            ← Mel-Spectrogram tensors
├── src/
│   ├── config.py
│   ├── preprocessing/
│   │   ├── audio_loader.py
│   │   ├── feature_extraction.py  ← LFCC, Mel-Spec, CQCC
│   │   └── dataset.py
│   ├── models/
│   │   ├── lcnn.py
│   │   ├── resnet_adapted.py
│   │   └── multi_task_head.py
│   ├── losses/
│   │   ├── center_loss.py
│   │   └── combined_loss.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_visualize_artifacts.ipynb
│   └── 03_train_analysis.ipynb
├── configs/
│   └── train_config.yaml
└── requirements.txt
```

## 6.3. Giai đoạn 2: Phát triển Mô hình Level 1 & 2 (Tuần 4–7)

### Tuần 4–5: Level 1 — Binary Detection

| Công việc | Chi tiết |
|-----------|----------|
| Xây dựng LCNN backbone | Theo kiến trúc Bảng 4.2.3 |
| Dataset: ASVspoof 2019 LA | Train/Dev/Eval theo protocol chuẩn |
| Training | AdamW, lr=1e-4, 30 epochs, early stopping |
| Evaluation | EER + min t-DCF trên eval set |
| **Mục tiêu** | **EER ≤ 5% (baseline), ≤ 3% (optimized)** |

### Tuần 6–7: Level 2 — Attack Type Classification

| Công việc | Chi tiết |
|-----------|----------|
| Gán nhãn Level 2 | ASVspoof meta: A01–A06 → TTS, A07–A17 → VC |
| Thêm output head | FC → 4 classes (Bonafide/TTS/VC/Replay) |
| Training | CrossEntropyLoss cho Level 2, đồng thời với Level 1 |
| Evaluation | Accuracy, Macro F1, Confusion Matrix |
| **Mục tiêu** | **Macro F1 ≥ 85%** |

## 6.4. Giai đoạn 3: Nghiên cứu Level 3 — Model Attribution (Tuần 8–11)

### Tuần 8–9: Phân tích Dấu vết Vocoder

| Công việc | Chi tiết |
|-----------|----------|
| Visualize spectrogram 6 vocoders | WaveFake: HiFi-GAN, MelGAN, WaveGlow, PWGAN, MB-MelGAN, FB-MelGAN |
| Trích xuất và so sánh | Overlay spectrograms, tính spectral distance |
| Xác nhận dấu vết | Kiểm tra xem over-smoothing, spectral gaps, phase mismatch có xuất hiện đúng lý thuyết không |
| Tài liệu hóa | Ghi chú đặc trưng từng vocoder → phục vụ interpretability |

### Tuần 10–11: Huấn luyện Level 3

| Công việc | Chi tiết |
|-----------|----------|
| Dataset: WaveFake | 7 classes (bonafide + 6 vocoders), split 70/15/15 |
| Thêm output head 3 | FC → 7 classes |
| Fine-tune LCNN | Có thể freeze lower layers, chỉ train upper layers + head 3 |
| Thêm Center Loss | $\delta = 0.01$, center lr = 0.5 |
| Evaluation | Macro F1, Confusion Matrix (quan trọng: xem vocoder nào bị nhầm với nhau) |
| **Mục tiêu** | **Macro F1 ≥ 80%, không có cặp nhầm > 10%** |

## 6.5. Giai đoạn 4: Tích hợp Multi-Task & Đánh giá Toàn diện (Tuần 12–14)

### Tuần 12–13: Multi-Task Training

| Công việc | Chi tiết |
|-----------|----------|
| Kết hợp 3 dataset | ASVspoof 2019 LA (Level 1+2) + WaveFake (Level 3) |
| Combined Loss | $\mathcal{L} = \alpha \mathcal{L}_1 + \beta \mathcal{L}_2 + \gamma \mathcal{L}_3 + \delta \mathcal{L}_{center}$ |
| Tuning trọng số | Grid search $\alpha, \beta, \gamma$ trên validation set |
| Multi-task vs Single-task | So sánh: multi-task có cải thiện Level 1/2/3 so với train riêng? |

### Tuần 14: Đánh giá Toàn diện

| Đánh giá | Metric | Mục tiêu |
|----------|--------|----------|
| Level 1 trên ASVspoof 2019 eval | EER, min t-DCF | EER ≤ 3% |
| Level 2 trên ASVspoof + WaveFake | Macro F1 | ≥ 85% |
| Level 3 trên WaveFake test | Macro F1, Confusion Matrix | ≥ 80% |
| Cross-dataset (generalization) | EER trên ASVspoof 2021 DF | Theo dõi degradation |
| Ablation study | So sánh LFCC vs Mel-Spec, LCNN vs ResNet18 | Báo cáo chi tiết |

## 6.6. Giai đoạn 5: Giao diện & Triển khai (Tuần 15+)

### 5.1. Demo UI bằng Gradio / Streamlit

```
┌─────────────────────────────────────────────────────────────────┐
│         AUDIO DEEPFAKE DETECTOR — DEMO UI                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [📁 Upload Audio File]  [🎙️ Record]  [▶️ Analyze]               │
│                                                                  │
│  ┌──────────────────────────────────────┐                        │
│  │        SPECTROGRAM VISUALIZATION     │                        │
│  │  ┌──────────────────────────────┐    │                        │
│  │  │ ░░░▓▓▓░░▓▓░░░▓▓▓░░░▓▓░░░░  │    │                        │
│  │  │ ═══════════════════════════  │    │                        │
│  │  │ ═══════════════════════════  │    │                        │
│  │  └──────────────────────────────┘    │                        │
│  └──────────────────────────────────────┘                        │
│                                                                  │
│  ┌──────────────────────────────────────┐                        │
│  │         KẾT QUẢ PHÂN TÍCH            │                        │
│  │                                      │                        │
│  │  Level 1: 🔴 SPOOF (98.7%)          │                        │
│  │  Level 2: 🟠 TTS (92.3%)            │                        │
│  │  Level 3: 🔵 HiFi-GAN (87.1%)       │                        │
│  │                                      │                        │
│  │  ⚠️ Kết luận: Đoạn audio này được   │                        │
│  │  tạo bởi hệ thống Text-to-Speech    │                        │
│  │  sử dụng vocoder HiFi-GAN.          │                        │
│  └──────────────────────────────────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Chức năng chính:**
1. Upload file audio (WAV/MP3/FLAC) hoặc Record trực tiếp
2. Hiển thị **Spectrogram trực quan** (Mel-Spec + LFCC)
3. Kết quả 3 cấp độ với confidence %
4. **Highlight vùng artifact** trên spectrogram (heatmap attention)
5. Export báo cáo PDF (cho mục đích forensics)

---

# TÀI LIỆU THAM KHẢO

[1] Stupp, C. (2019). "Fraudsters Used AI to Mimic CEO's Voice in Unusual Cybercrime Case." *The Wall Street Journal*.

[2] Chen, H. (2024). "Finance worker pays out $25 million after video call with deepfake chief financial officer." *CNN*.

[3] Wang, C., et al. (2023). "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers." *arXiv:2301.02111* (VALL-E).

[4] Todisco, M., et al. (2019). "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection." *Proc. Interspeech*, 1008–1012.

[5] Todisco, M., Delgado, H., & Evans, N. (2017). "Constant Q cepstral coefficients: A spoofing countermeasure for automatic speaker verification." *Computer Speech & Language*, 45, 516–535.

[6] Wu, Z., et al. (2018). "Light Convolutional Neural Network with Feature Genuinization for Detection of Synthetic Speech Attacks." *Proc. Interspeech*.

[7] Wen, Y., Zhang, K., Li, Z., & Qiao, Y. (2016). "A discriminative feature learning approach for deep face recognition." *ECCV*, 499–515.

[8] Frank, J., & Schönherr, L. (2021). "WaveFake: A Data Set to Facilitate Audio Deepfake Detection." *Proc. NeurIPS Datasets and Benchmarks Track*.

[9] Kong, J., Kim, J., & Bae, J. (2020). "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis." *Proc. NeurIPS*.

[10] Kumar, K., et al. (2019). "MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis." *Proc. NeurIPS*.

[11] van den Oord, A., et al. (2016). "WaveNet: A Generative Model for Raw Audio." *arXiv:1609.03499*.

[12] Prenger, R., Valle, R., & Catanzaro, B. (2019). "WaveGlow: A Flow-based Generative Network for Speech Synthesis." *Proc. ICASSP*.

[13] Yamamoto, R., Song, E., & Kim, J. M. (2020). "Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram." *Proc. ICASSP*.

[14] Kim, J., Kong, J., & Son, J. (2021). "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech." *Proc. ICML* (VITS).

[15] Müller, N. M., et al. (2022). "Does Audio Deepfake Detection Generalize?" *Proc. Interspeech*.

[16] Tak, H., et al. (2021). "End-to-End Anti-Spoofing with RawNet2." *Proc. ICASSP*.

[17] Kendall, A., Gal, Y., & Cipolla, R. (2018). "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." *Proc. CVPR*.

---

> *Tài liệu này là Đề cương và Lộ trình Tổng thể (Master Roadmap). Các chương tiếp theo sẽ đi sâu vào từng giai đoạn thực hiện kèm theo code, kết quả thực nghiệm, và phân tích chi tiết.*
