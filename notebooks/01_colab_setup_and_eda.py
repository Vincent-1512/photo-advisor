"""
============================================================
AUDIO DEEPFAKE DETECTION — GOOGLE COLAB SETUP & EDA
============================================================
Notebook này chạy trên Google Colab (có GPU miễn phí).
Bao gồm:
  1. Cài đặt dependencies
  2. Mount Google Drive
  3. Tải dataset ASVspoof 2019 LA
  4. Khám phá dữ liệu (EDA)
  5. Visualize spectrogram: Bonafide vs Spoof
  6. Test pipeline tiền xử lý

HƯỚNG DẪN SỬ DỤNG:
  1. Mở Google Colab: https://colab.research.google.com
  2. Upload file này lên Colab (File → Upload notebook)
     hoặc copy paste từng cell
  3. Chuyển Runtime → Change runtime type → T4 GPU
  4. Chạy từng cell theo thứ tự
============================================================
"""

# ===========================================================
# CELL 1: Cài đặt dependencies
# ===========================================================
# !pip install -q torch torchaudio librosa soundfile numpy pandas matplotlib seaborn scikit-learn tqdm pyyaml

# ===========================================================
# CELL 2: Mount Google Drive
# ===========================================================
"""
from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_ROOT = "/content/drive/MyDrive/audio_deepfake"
os.makedirs(PROJECT_ROOT, exist_ok=True)
os.makedirs(f"{PROJECT_ROOT}/data", exist_ok=True)
os.makedirs(f"{PROJECT_ROOT}/outputs/eda", exist_ok=True)
print(f"Project root: {PROJECT_ROOT}")
"""

# ===========================================================
# CELL 3: Upload source code lên Colab
# ===========================================================
"""
# Option A: Clone từ Google Drive (nếu bạn đã copy src/ lên Drive)
import sys
sys.path.insert(0, PROJECT_ROOT)

# Option B: Upload trực tiếp (chạy cell này rồi upload file zip)
# from google.colab import files
# uploaded = files.upload()  # Upload src.zip
# !unzip -o src.zip -d {PROJECT_ROOT}
"""

# ===========================================================
# CELL 4: Tải dataset ASVspoof 2019 LA
# ===========================================================
"""
# ASVspoof 2019 LA cần tải thủ công do yêu cầu license agreement.
# Có 2 cách:

# ─── Cách 1: Tải từ Edinburgh DataShare (chính thức) ───
# 1. Truy cập: https://datashare.ed.ac.uk/handle/10283/3336
# 2. Tải file: LA.zip (~6GB)
# 3. Upload lên Google Drive vào thư mục: audio_deepfake/data/
# 4. Giải nén:

DATA_DIR = f"{PROJECT_ROOT}/data"
ASVSPOOF_ZIP = f"{DATA_DIR}/LA.zip"

import os
if os.path.exists(ASVSPOOF_ZIP):
    print("Đang giải nén ASVspoof 2019 LA...")
    !cd {DATA_DIR} && mkdir -p asvspoof2019 && unzip -q {ASVSPOOF_ZIP} -d asvspoof2019/
    print("Giải nén xong!")
else:
    print(f"Chưa tìm thấy {ASVSPOOF_ZIP}")
    print("Vui lòng tải LA.zip từ https://datashare.ed.ac.uk/handle/10283/3336")
    print(f"rồi upload lên Google Drive vào: {DATA_DIR}/")

# ─── Cách 2: Tải từ Hugging Face (nếu có mirror) ───
# !pip install -q huggingface_hub
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="asvspoof/asvspoof2019", repo_type="dataset",
#                   local_dir=f"{DATA_DIR}/asvspoof2019")
"""

# ===========================================================
# CELL 5: Kiểm tra cấu trúc thư mục dataset
# ===========================================================
"""
import os

asvspoof_la = f"{PROJECT_ROOT}/data/asvspoof2019/LA"
expected_dirs = [
    "ASVspoof2019_LA_train/flac",
    "ASVspoof2019_LA_dev/flac",
    "ASVspoof2019_LA_eval/flac",
    "ASVspoof2019_LA_cm_protocols",
]

print("Kiểm tra cấu trúc dataset ASVspoof 2019 LA:")
print("=" * 50)
all_ok = True
for d in expected_dirs:
    full_path = os.path.join(asvspoof_la, d)
    exists = os.path.exists(full_path)
    status = "✅" if exists else "❌"
    if exists and os.path.isdir(full_path):
        num_files = len(os.listdir(full_path))
        print(f"  {status} {d} ({num_files:,} files)")
    else:
        print(f"  {status} {d}")
        all_ok = False

if all_ok:
    print("\\n✅ Dataset đã sẵn sàng!")
else:
    print("\\n❌ Thiếu một số thư mục. Kiểm tra lại bước tải dataset.")
"""

# ===========================================================
# CELL 6: Parse Protocol & Thống kê Dataset
# ===========================================================
"""
import sys
sys.path.insert(0, PROJECT_ROOT)

from src.utils.protocol_parser import parse_asvspoof_protocol, get_dataset_statistics

# Parse train protocol
train_protocol = f"{PROJECT_ROOT}/data/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
train_audio_dir = f"{PROJECT_ROOT}/data/asvspoof2019/LA/ASVspoof2019_LA_train/flac"

df_train = parse_asvspoof_protocol(train_protocol, train_audio_dir)
print("=== TRAIN SET ===")
print(get_dataset_statistics(df_train))

# Parse dev protocol
dev_protocol = f"{PROJECT_ROOT}/data/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
dev_audio_dir = f"{PROJECT_ROOT}/data/asvspoof2019/LA/ASVspoof2019_LA_dev/flac"

df_dev = parse_asvspoof_protocol(dev_protocol, dev_audio_dir)
print("\\n=== DEV SET ===")
print(get_dataset_statistics(df_dev))

# Parse eval protocol
eval_protocol = f"{PROJECT_ROOT}/data/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
eval_audio_dir = f"{PROJECT_ROOT}/data/asvspoof2019/LA/ASVspoof2019_LA_eval/flac"

df_eval = parse_asvspoof_protocol(eval_protocol, eval_audio_dir)
print("\\n=== EVAL SET ===")
print(get_dataset_statistics(df_eval))
"""

# ===========================================================
# CELL 7: Visualize Phân bổ Nhãn
# ===========================================================
"""
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, df) in zip(axes, [("Train", df_train), ("Dev", df_dev), ("Eval", df_eval)]):
    counts = df["label_str"].value_counts()
    colors = ["#4CAF50", "#F44336"]  # Green=bonafide, Red=spoof
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title(f"{name} Set", fontsize=14, fontweight="bold")
    ax.set_ylabel("Số lượng mẫu")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 50, f"{v:,}", ha="center", fontsize=12)

plt.suptitle("Phân bổ Bonafide vs Spoof", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PROJECT_ROOT}/outputs/eda/label_distribution.png", dpi=150)
plt.show()
print("Saved: outputs/eda/label_distribution.png")
"""

# ===========================================================
# CELL 8: Nghe thử & So sánh Spectrogram
# ===========================================================
"""
import torch
import torchaudio
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

from src.preprocessing.audio_loader import load_and_preprocess
from src.preprocessing.feature_extraction import extract_mel_spectrogram, extract_lfcc

# Lấy 1 mẫu bonafide và 1 mẫu spoof từ train set
bonafide_sample = df_train[df_train["label_str"] == "bonafide"].iloc[0]
spoof_sample = df_train[df_train["label_str"] == "spoof"].iloc[0]

print(f"Bonafide: {bonafide_sample['filename']} (Speaker: {bonafide_sample['speaker_id']})")
print(f"Spoof:    {spoof_sample['filename']} (System: {spoof_sample['system_id']}, Speaker: {spoof_sample['speaker_id']})")

# Load audio
wav_real = load_and_preprocess(bonafide_sample["audio_path"])
wav_fake = load_and_preprocess(spoof_sample["audio_path"])

# Nghe thử
print("\\n🎧 Giọng THẬT:")
display(Audio(wav_real.numpy(), rate=16000))
print("\\n🎧 Giọng GIẢ:")
display(Audio(wav_fake.numpy(), rate=16000))
"""

# ===========================================================
# CELL 9: So sánh Visual — Mel-Spectrogram + LFCC
# ===========================================================
"""
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Mel-Spectrogram
mel_real = extract_mel_spectrogram(wav_real, normalize=False)
mel_fake = extract_mel_spectrogram(wav_fake, normalize=False)

axes[0, 0].set_title("BONAFIDE — Mel-Spectrogram", fontsize=13, fontweight="bold", color="green")
im1 = axes[0, 0].imshow(mel_real.numpy(), aspect="auto", origin="lower", cmap="magma")
axes[0, 0].set_ylabel("Mel Bin")
plt.colorbar(im1, ax=axes[0, 0], format="%+2.0f dB")

axes[0, 1].set_title("SPOOF — Mel-Spectrogram", fontsize=13, fontweight="bold", color="red")
im2 = axes[0, 1].imshow(mel_fake.numpy(), aspect="auto", origin="lower", cmap="magma")
axes[0, 1].set_ylabel("Mel Bin")
plt.colorbar(im2, ax=axes[0, 1], format="%+2.0f dB")

# LFCC
lfcc_real = extract_lfcc(wav_real, normalize=False)
lfcc_fake = extract_lfcc(wav_fake, normalize=False)

axes[1, 0].set_title("BONAFIDE — LFCC", fontsize=13, fontweight="bold", color="green")
im3 = axes[1, 0].imshow(lfcc_real.numpy(), aspect="auto", origin="lower", cmap="viridis")
axes[1, 0].set_ylabel("LFCC Coefficient")
axes[1, 0].set_xlabel("Frame")
plt.colorbar(im3, ax=axes[1, 0])

axes[1, 1].set_title("SPOOF — LFCC", fontsize=13, fontweight="bold", color="red")
im4 = axes[1, 1].imshow(lfcc_fake.numpy(), aspect="auto", origin="lower", cmap="viridis")
axes[1, 1].set_ylabel("LFCC Coefficient")
axes[1, 1].set_xlabel("Frame")
plt.colorbar(im4, ax=axes[1, 1])

plt.suptitle("So sánh Visual: Giọng Thật vs Giọng AI", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PROJECT_ROOT}/outputs/eda/spectrogram_comparison.png", dpi=150)
plt.show()
print("Saved: outputs/eda/spectrogram_comparison.png")
"""

# ===========================================================
# CELL 10: So sánh nhiều System ID (6 attack systems)
# ===========================================================
"""
fig, axes = plt.subplots(2, 4, figsize=(24, 8))

# Row 1: Bonafide + A01-A03
# Row 2: A04-A06 + Summary
system_ids = ["-", "A01", "A02", "A03", "A04", "A05", "A06"]
system_names = ["Bonafide", "A01 (TTS)", "A02 (TTS)", "A03 (TTS)",
                "A04 (TTS)", "A05 (VC)", "A06 (VC)"]

for idx, (sys_id, sys_name) in enumerate(zip(system_ids, system_names)):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]
    
    if sys_id == "-":
        sample = df_train[df_train["label_str"] == "bonafide"].iloc[0]
    else:
        samples_for_sys = df_train[df_train["system_id"] == sys_id]
        if len(samples_for_sys) == 0:
            ax.set_visible(False)
            continue
        sample = samples_for_sys.iloc[0]
    
    wav = load_and_preprocess(sample["audio_path"])
    mel = extract_mel_spectrogram(wav, normalize=False)
    
    color = "green" if sys_id == "-" else "red"
    ax.set_title(sys_name, fontsize=12, fontweight="bold", color=color)
    ax.imshow(mel.numpy(), aspect="auto", origin="lower", cmap="magma")
    ax.set_ylabel("Mel" if col == 0 else "")
    ax.set_xlabel("Frame")

# Ẩn cell cuối nếu không dùng
if len(system_ids) < 8:
    axes[1, 3].set_visible(False)

plt.suptitle("Mel-Spectrogram theo từng Attack System", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PROJECT_ROOT}/outputs/eda/systems_comparison.png", dpi=150)
plt.show()
print("Saved: outputs/eda/systems_comparison.png")
"""

# ===========================================================
# CELL 11: Test PyTorch Dataset & DataLoader
# ===========================================================
"""
from src.preprocessing.dataset import ASVspoofDataset, create_dataloaders

# Tạo dataset
train_dataset = ASVspoofDataset(
    protocol_path=train_protocol,
    audio_dir=train_audio_dir,
    feature_type="lfcc",
    random_crop=True,
)

print(f"Train dataset: {len(train_dataset):,} samples")
print(f"Label distribution: {train_dataset.get_label_distribution()}")

# Test lấy 1 sample
features, labels = train_dataset[0]
print(f"\\nFeatures shape: {features.shape}")   # Expected: (1, 60, 401)
print(f"Labels: {labels}")

# Tạo DataLoader
train_loader, dev_loader = create_dataloaders(
    train_protocol=train_protocol,
    train_audio_dir=train_audio_dir,
    dev_protocol=dev_protocol,
    dev_audio_dir=dev_audio_dir,
    feature_type="lfcc",
    batch_size=32,
    num_workers=2,  # Colab thường dùng 2
)

# Test 1 batch
batch_features, batch_labels = next(iter(train_loader))
print(f"\\nBatch features shape: {batch_features.shape}")  # (32, 1, 60, 401)
print(f"Batch binary labels: {batch_labels['binary']}")
print("\\n✅ Pipeline hoạt động!")
"""

# ===========================================================
# CELL 12: Kiểm tra GPU
# ===========================================================
"""
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    device = torch.device("cpu")
    print("⚠️ Không có GPU. Hãy chuyển Runtime → Change runtime type → T4 GPU")

print(f"\\nDevice: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"TorchAudio version: {torchaudio.__version__}")
"""

# ===========================================================
# CELL 13: Summary — Kết thúc Phase 1
# ===========================================================
"""
print("=" * 60)
print("  PHASE 1 COMPLETE — SUMMARY")
print("=" * 60)
print()
print("  ✅ Dependencies installed")
print("  ✅ Google Drive mounted")
print("  ✅ ASVspoof 2019 LA loaded")
print("  ✅ Protocol parsed (train/dev/eval)")
print("  ✅ Label distribution analyzed")
print("  ✅ Spectrogram visualized (Bonafide vs Spoof)")
print("  ✅ Feature extraction tested (Mel-Spec + LFCC)")
print("  ✅ PyTorch Dataset + DataLoader working")
print("  ✅ GPU ready")
print()
print("  → Next: Phase 2 — Train LCNN model (Level 1: Binary Detection)")
print("=" * 60)
"""
