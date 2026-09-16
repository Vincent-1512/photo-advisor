"""
Protocol Parser — Đọc và parse file protocol của ASVspoof
==========================================================
File protocol (.txt) là "đáp án" của dataset, chứa nhãn cho từng file audio.
Module này chuyển file text thành DataFrame dễ xử lý.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict


# ============================================================
# Mapping System ID → Attack Type (Level 2 labels)
# Dựa theo ASVspoof 2019 LA documentation
# ============================================================

# Train + Dev systems (known attacks)
SYSTEM_TO_ATTACK_TYPE: Dict[str, str] = {
    # TTS (Text-to-Speech) systems
    "A01": "TTS",   # Neural waveform (WaveNet-based)
    "A02": "TTS",   # Vocoder-based
    "A03": "TTS",   # Vocoder-based
    "A04": "TTS",   # Waveform concatenation
    "A05": "VC",    # Voice Conversion (spectral)
    "A06": "VC",    # Voice Conversion (spectral)
    
    # Eval systems (unknown attacks — ẩn khi train, chỉ dùng khi test)
    "A07": "TTS",
    "A08": "TTS",
    "A09": "TTS",
    "A10": "TTS",
    "A11": "TTS",
    "A12": "TTS",
    "A13": "TTS",   # Waveform concatenation + signal processing
    "A14": "TTS",
    "A15": "TTS",
    "A16": "TTS",
    "A17": "VC",    # Voice Conversion
    "A18": "VC",
    "A19": "VC",
}


def parse_asvspoof_protocol(
    protocol_path: str | Path,
    audio_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Parse file protocol của ASVspoof 2019 LA thành DataFrame.
    
    Cấu trúc file protocol (mỗi dòng):
        SPEAKER_ID AUDIO_FILENAME - SYSTEM_ID LABEL
        LA_0039    LA_T_1138215   - -         bonafide
        LA_0021    LA_T_1271820   - A05       spoof
    
    Args:
        protocol_path: Đường dẫn đến file protocol (.txt)
        audio_dir: Thư mục chứa file audio (optional, dùng để tạo full path)
    
    Returns:
        DataFrame với columns:
            - speaker_id (str): ID người nói
            - filename (str): Tên file audio (không có extension)
            - system_id (str): ID thuật toán tạo giọng giả (hoặc "-" nếu bonafide)
            - label_str (str): "bonafide" hoặc "spoof"
            - label (int): 0 = bonafide, 1 = spoof (Level 1)
            - attack_type (str): "bonafide", "TTS", hoặc "VC" (Level 2)
            - attack_type_id (int): 0 = bonafide, 1 = TTS, 2 = VC (Level 2)
            - audio_path (str): Đường dẫn đầy đủ đến file audio (nếu audio_dir được cung cấp)
    
    Raises:
        FileNotFoundError: Nếu file protocol không tồn tại
    """
    protocol_path = Path(protocol_path)
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    
    # Đọc file text
    records = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            
            speaker_id = parts[0]
            filename = parts[1]
            # parts[2] là "-" (placeholder, bỏ qua)
            system_id = parts[3]
            label_str = parts[4] if len(parts) > 4 else "bonafide"
            
            records.append({
                "speaker_id": speaker_id,
                "filename": filename,
                "system_id": system_id,
                "label_str": label_str,
            })
    
    df = pd.DataFrame(records)
    
    # Level 1: Binary label (bonafide=0, spoof=1)
    df["label"] = (df["label_str"] == "spoof").astype(int)
    
    # Level 2: Attack type
    df["attack_type"] = df.apply(
        lambda row: "bonafide" if row["label_str"] == "bonafide"
        else SYSTEM_TO_ATTACK_TYPE.get(row["system_id"], "unknown"),
        axis=1
    )
    
    attack_type_mapping = {"bonafide": 0, "TTS": 1, "VC": 2, "Replay": 3, "unknown": -1}
    df["attack_type_id"] = df["attack_type"].map(attack_type_mapping)
    
    # Tạo full audio path nếu audio_dir được cung cấp
    if audio_dir is not None:
        audio_dir = Path(audio_dir)
        df["audio_path"] = df["filename"].apply(
            lambda fn: str(audio_dir / f"{fn}.flac")
        )
    
    return df


def get_dataset_statistics(df: pd.DataFrame) -> str:
    """
    In thống kê tổng quan của dataset.
    
    Args:
        df: DataFrame từ parse_asvspoof_protocol
    
    Returns:
        Chuỗi text chứa thống kê
    """
    lines = [
        "=" * 50,
        "  DATASET STATISTICS",
        "=" * 50,
        "",
        f"  Total samples:    {len(df):,}",
        f"  Bonafide (Real):  {(df['label'] == 0).sum():,}",
        f"  Spoof (Fake):     {(df['label'] == 1).sum():,}",
        f"  Ratio (Real:Fake): 1:{(df['label'] == 1).sum() / max((df['label'] == 0).sum(), 1):.1f}",
        "",
        "  --- Level 2: Attack Type ---",
    ]
    
    for attack_type in df["attack_type"].unique():
        count = (df["attack_type"] == attack_type).sum()
        lines.append(f"    {attack_type:12s}: {count:,}")
    
    lines.extend([
        "",
        "  --- System ID Distribution ---",
    ])
    
    for system_id in sorted(df["system_id"].unique()):
        count = (df["system_id"] == system_id).sum()
        attack_type = SYSTEM_TO_ATTACK_TYPE.get(system_id, "bonafide")
        lines.append(f"    {system_id:6s} ({attack_type:8s}): {count:,}")
    
    lines.extend([
        "",
        f"  Speakers:         {df['speaker_id'].nunique()}",
        "=" * 50,
    ])
    
    return "\n".join(lines)
