"""
EDA Script — Khám phá dữ liệu ASVspoof 2019 LA
=================================================
Chạy: python scripts/01_eda.py

Output:
  - outputs/eda/label_distribution.png
  - outputs/eda/duration_histogram.png
  - outputs/eda/spectrogram_bonafide_vs_spoof.png
  - outputs/eda/system_spectrograms.png
  - outputs/eda/statistics.txt
"""

import sys
import os
from pathlib import Path

# Thêm project root vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.utils.protocol_parser import parse_asvspoof_protocol, get_dataset_statistics
from src.preprocessing.audio_loader import load_audio, pad_or_trim
from src.preprocessing.feature_extraction import extract_mel_spectrogram, extract_lfcc
from src.config import cfg


def setup_output_dir():
    """Tạo thư mục output."""
    output_dir = cfg.paths.eda_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def analyze_protocols():
    """Parse và in thống kê các tập train/dev/eval."""
    print("\n" + "=" * 60)
    print("  STEP 1: Parse Protocol Files")
    print("=" * 60)

    datasets = {}
    protocol_configs = [
        ("train", cfg.paths.train_protocol, cfg.paths.asvspoof_train_audio),
        ("dev", cfg.paths.dev_protocol, cfg.paths.asvspoof_dev_audio),
        ("eval", cfg.paths.eval_protocol, cfg.paths.asvspoof_eval_audio),
    ]

    for name, protocol_path, audio_dir in protocol_configs:
        if not protocol_path.exists():
            print(f"  ⚠️ {name}: Protocol file not found: {protocol_path}")
            continue

        df = parse_asvspoof_protocol(protocol_path, audio_dir)
        datasets[name] = df
        print(f"\n--- {name.upper()} SET ---")
        print(get_dataset_statistics(df))

    return datasets


def plot_label_distribution(datasets: dict, output_dir: Path):
    """Vẽ biểu đồ phân bổ nhãn."""
    print("\n" + "=" * 60)
    print("  STEP 2: Label Distribution")
    print("=" * 60)

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    colors = {"bonafide": "#4CAF50", "spoof": "#F44336"}

    for ax, (name, df) in zip(axes, datasets.items()):
        counts = df["label_str"].value_counts()
        bars = ax.bar(
            counts.index, counts.values,
            color=[colors.get(k, "#999") for k in counts.index],
            edgecolor="white", linewidth=1.5
        )
        ax.set_title(f"{name.upper()} Set", fontsize=14, fontweight="bold")
        ax.set_ylabel("Số lượng mẫu")
        for bar, v in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 50,
                f"{v:,}", ha="center", fontsize=12, fontweight="bold"
            )

    plt.suptitle("Phân bổ Bonafide vs Spoof", fontsize=16, fontweight="bold")
    plt.tight_layout()

    save_path = output_dir / "label_distribution.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {save_path}")


def analyze_durations(datasets: dict, output_dir: Path, max_files: int = 500):
    """Phân tích độ dài file audio."""
    print("\n" + "=" * 60)
    print("  STEP 3: Duration Analysis")
    print("=" * 60)

    all_durations = {"bonafide": [], "spoof": []}

    # Lấy train set để phân tích
    df = datasets.get("train")
    if df is None or "audio_path" not in df.columns:
        print("  ⚠️ Không tìm thấy audio_path trong train set. Bỏ qua.")
        return

    sample_df = df.sample(n=min(max_files, len(df)), random_state=42)

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="  Đo duration"):
        try:
            info = torchaudio.info(row["audio_path"])
            duration = info.num_frames / info.sample_rate
            all_durations[row["label_str"]].append(duration)
        except Exception:
            continue

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if all_durations["bonafide"]:
        ax.hist(all_durations["bonafide"], bins=50, alpha=0.7,
                color="#4CAF50", label=f"Bonafide (n={len(all_durations['bonafide'])})")
    if all_durations["spoof"]:
        ax.hist(all_durations["spoof"], bins=50, alpha=0.7,
                color="#F44336", label=f"Spoof (n={len(all_durations['spoof'])})")
    ax.set_xlabel("Duration (seconds)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Phân bổ Độ dài File Audio", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.axvline(x=4.0, color="blue", linestyle="--", alpha=0.5, label="Target: 4s")
    ax.legend()

    save_path = output_dir / "duration_histogram.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {save_path}")

    for label, durations in all_durations.items():
        if durations:
            print(f"  {label}: mean={np.mean(durations):.2f}s, "
                  f"min={np.min(durations):.2f}s, max={np.max(durations):.2f}s")


def visualize_spectrograms(datasets: dict, output_dir: Path):
    """So sánh spectrogram bonafide vs spoof."""
    print("\n" + "=" * 60)
    print("  STEP 4: Spectrogram Comparison")
    print("=" * 60)

    df = datasets.get("train")
    if df is None or "audio_path" not in df.columns:
        print("  ⚠️ Không tìm thấy audio_path. Bỏ qua.")
        return

    # Lấy 1 bonafide + 1 spoof
    bonafide_row = df[df["label_str"] == "bonafide"].iloc[0]
    spoof_row = df[df["label_str"] == "spoof"].iloc[0]

    wav_real, _ = load_audio(bonafide_row["audio_path"])
    wav_fake, _ = load_audio(spoof_row["audio_path"])
    wav_real = pad_or_trim(wav_real, 64000)
    wav_fake = pad_or_trim(wav_fake, 64000)

    mel_real = extract_mel_spectrogram(wav_real, normalize=False)
    mel_fake = extract_mel_spectrogram(wav_fake, normalize=False)
    lfcc_real = extract_lfcc(wav_real, normalize=False)
    lfcc_fake = extract_lfcc(wav_fake, normalize=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].set_title("BONAFIDE — Mel-Spectrogram", fontweight="bold", color="green")
    im1 = axes[0, 0].imshow(mel_real.numpy(), aspect="auto", origin="lower", cmap="magma")
    axes[0, 0].set_ylabel("Mel Bin")
    plt.colorbar(im1, ax=axes[0, 0], format="%+2.0f dB")

    axes[0, 1].set_title("SPOOF — Mel-Spectrogram", fontweight="bold", color="red")
    im2 = axes[0, 1].imshow(mel_fake.numpy(), aspect="auto", origin="lower", cmap="magma")
    axes[0, 1].set_ylabel("Mel Bin")
    plt.colorbar(im2, ax=axes[0, 1], format="%+2.0f dB")

    axes[1, 0].set_title("BONAFIDE — LFCC", fontweight="bold", color="green")
    im3 = axes[1, 0].imshow(lfcc_real.numpy(), aspect="auto", origin="lower", cmap="viridis")
    axes[1, 0].set_ylabel("LFCC Coeff")
    axes[1, 0].set_xlabel("Frame")
    plt.colorbar(im3, ax=axes[1, 0])

    axes[1, 1].set_title("SPOOF — LFCC", fontweight="bold", color="red")
    im4 = axes[1, 1].imshow(lfcc_fake.numpy(), aspect="auto", origin="lower", cmap="viridis")
    axes[1, 1].set_ylabel("LFCC Coeff")
    axes[1, 1].set_xlabel("Frame")
    plt.colorbar(im4, ax=axes[1, 1])

    plt.suptitle("So sánh: Giọng Thật vs Giọng AI", fontsize=16, fontweight="bold")
    plt.tight_layout()

    save_path = output_dir / "spectrogram_bonafide_vs_spoof.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {save_path}")

    # Hình kích thước tensor
    print(f"  Mel-Spectrogram shape: {mel_real.shape} (n_mels × T)")
    print(f"  LFCC shape:            {lfcc_real.shape} (n_lfcc × T)")


def visualize_all_systems(datasets: dict, output_dir: Path):
    """Vẽ spectrogram cho từng attack system."""
    print("\n" + "=" * 60)
    print("  STEP 5: Per-System Spectrograms")
    print("=" * 60)

    df = datasets.get("train")
    if df is None or "audio_path" not in df.columns:
        print("  ⚠️ Bỏ qua.")
        return

    system_ids = sorted(df["system_id"].unique())
    n_systems = len(system_ids)
    cols = 4
    rows = (n_systems + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()

    for idx, sys_id in enumerate(system_ids):
        ax = axes[idx]
        subset = df[df["system_id"] == sys_id]
        sample = subset.iloc[0]

        try:
            wav, _ = load_audio(sample["audio_path"])
            wav = pad_or_trim(wav, 64000)
            mel = extract_mel_spectrogram(wav, normalize=False)

            color = "green" if sys_id == "-" else "red"
            label = sample["attack_type"] if "attack_type" in sample else sys_id
            ax.set_title(f"{sys_id} ({label})", fontweight="bold", color=color)
            ax.imshow(mel.numpy(), aspect="auto", origin="lower", cmap="magma")
            ax.set_xlabel("Frame")
            if idx % cols == 0:
                ax.set_ylabel("Mel Bin")
        except Exception as e:
            ax.set_title(f"{sys_id} — Error", color="gray")
            ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes)

    # Ẩn axes thừa
    for idx in range(n_systems, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Mel-Spectrogram theo Attack System", fontsize=16, fontweight="bold")
    plt.tight_layout()

    save_path = output_dir / "system_spectrograms.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {save_path}")


def main():
    """Chạy toàn bộ EDA pipeline."""
    print("=" * 60)
    print("  AUDIO DEEPFAKE DETECTION — EDA")
    print("=" * 60)

    output_dir = setup_output_dir()
    datasets = analyze_protocols()

    if not datasets:
        print("\n❌ Không tìm thấy dataset. Hãy tải ASVspoof 2019 LA trước.")
        print(f"   Đặt vào: {cfg.paths.asvspoof_dir}")
        return

    plot_label_distribution(datasets, output_dir)
    analyze_durations(datasets, output_dir)
    visualize_spectrograms(datasets, output_dir)
    visualize_all_systems(datasets, output_dir)

    # Save text statistics
    stats_path = output_dir / "statistics.txt"
    with open(stats_path, "w") as f:
        for name, df in datasets.items():
            f.write(f"=== {name.upper()} SET ===\n")
            f.write(get_dataset_statistics(df))
            f.write("\n\n")
    print(f"\n  ✅ Statistics saved: {stats_path}")

    print("\n" + "=" * 60)
    print("  ✅ EDA COMPLETE!")
    print(f"  📁 All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
