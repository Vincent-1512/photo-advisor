"""
Batch Preprocessing Script — Trích xuất features hàng loạt
============================================================
Chạy: python scripts/02_preprocess.py --feature-type lfcc

Đọc toàn bộ file audio trong dataset, trích xuất LFCC/Mel-Spec,
lưu thành file .pt (PyTorch tensor) để training loop không cần
tính lại mỗi epoch → tăng tốc training đáng kể.

Output:
  data/processed/lfcc/train/  → {filename}.pt
  data/processed/lfcc/dev/    → {filename}.pt
  data/processed/lfcc/eval/   → {filename}.pt
"""

import sys
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from tqdm import tqdm

from src.config import cfg
from src.utils.protocol_parser import parse_asvspoof_protocol
from src.preprocessing.audio_loader import load_and_preprocess
from src.preprocessing.feature_extraction import extract_features


def preprocess_partition(
    protocol_path: Path,
    audio_dir: Path,
    output_dir: Path,
    feature_type: str = "lfcc",
    sample_rate: int = 16000,
    duration: float = 4.0,
):
    """
    Trích xuất features cho 1 partition (train/dev/eval).

    Args:
        protocol_path: Đường dẫn protocol file
        audio_dir: Thư mục chứa file .flac
        output_dir: Thư mục lưu output .pt
        feature_type: "lfcc" hoặc "melspec"
        sample_rate: Tần số lấy mẫu
        duration: Độ dài mẫu (giây)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target_length = int(sample_rate * duration)

    # Parse protocol
    df = parse_asvspoof_protocol(protocol_path, audio_dir)

    success_count = 0
    error_count = 0
    skipped_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {output_dir.name}"):
        filename = row["filename"]
        output_path = output_dir / f"{filename}.pt"

        # Skip nếu đã xử lý rồi
        if output_path.exists():
            skipped_count += 1
            continue

        try:
            audio_path = row.get("audio_path", str(audio_dir / f"{filename}.flac"))

            # Load + preprocess audio
            waveform = load_and_preprocess(
                file_path=audio_path,
                target_sr=sample_rate,
                target_length=target_length,
                pre_emphasis_coeff=0.97,
                random_crop=False,
            )

            # Trích xuất features
            features = extract_features(
                waveform=waveform,
                feature_type=feature_type,
                sample_rate=sample_rate,
            )

            # Lưu tensor + labels
            data = {
                "features": features,              # (F, T)
                "binary_label": row["label"],       # 0 or 1
                "attack_type_id": row["attack_type_id"],
                "system_id": row["system_id"],
                "speaker_id": row["speaker_id"],
                "filename": filename,
            }
            torch.save(data, output_path)
            success_count += 1

        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"    ⚠️ Error processing {filename}: {e}")

    print(f"    ✅ Done: {success_count:,} processed, "
          f"{skipped_count:,} skipped, {error_count:,} errors")
    return success_count, skipped_count, error_count


def main():
    parser = argparse.ArgumentParser(description="Batch feature extraction")
    parser.add_argument(
        "--feature-type", type=str, default="lfcc",
        choices=["lfcc", "melspec"],
        help="Loại đặc trưng: lfcc hoặc melspec (mặc định: lfcc)"
    )
    parser.add_argument(
        "--partitions", type=str, nargs="+", default=["train", "dev", "eval"],
        help="Các partition cần xử lý (mặc định: train dev eval)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"  BATCH PREPROCESSING — Feature: {args.feature_type.upper()}")
    print("=" * 60)

    start_time = time.time()

    # Cấu hình cho từng partition
    partition_configs = {
        "train": (cfg.paths.train_protocol, cfg.paths.asvspoof_train_audio),
        "dev": (cfg.paths.dev_protocol, cfg.paths.asvspoof_dev_audio),
        "eval": (cfg.paths.eval_protocol, cfg.paths.asvspoof_eval_audio),
    }

    # Thư mục output
    if args.feature_type == "lfcc":
        base_output = cfg.paths.processed_lfcc_dir
    else:
        base_output = cfg.paths.processed_melspec_dir

    total_success = 0
    total_skipped = 0
    total_errors = 0

    for partition in args.partitions:
        if partition not in partition_configs:
            print(f"  ⚠️ Unknown partition: {partition}")
            continue

        protocol_path, audio_dir = partition_configs[partition]
        output_dir = base_output / partition

        if not protocol_path.exists():
            print(f"  ⚠️ Protocol not found: {protocol_path}")
            continue

        print(f"\n  Processing: {partition}")
        s, sk, e = preprocess_partition(
            protocol_path=protocol_path,
            audio_dir=audio_dir,
            output_dir=output_dir,
            feature_type=args.feature_type,
        )
        total_success += s
        total_skipped += sk
        total_errors += e

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  PREPROCESSING COMPLETE")
    print(f"  Total: {total_success:,} processed, "
          f"{total_skipped:,} skipped, {total_errors:,} errors")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"  Output: {base_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
