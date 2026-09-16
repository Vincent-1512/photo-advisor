"""
Feature Extraction — Trích xuất đặc trưng từ waveform
======================================================
Module chịu trách nhiệm biến đổi sóng âm 1D thành biểu diễn 2D (ảnh quang phổ):
  1. Mel-Spectrogram: thang tần số Mel (mô phỏng tai người)
  2. LFCC: thang tần số tuyến tính (bảo toàn dải cao → bắt artifact tốt hơn)
"""

import torch
import torchaudio
import numpy as np
from typing import Optional


def extract_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = 8000.0,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Trích xuất Mel-Spectrogram từ waveform.
    
    Pipeline:
        Waveform → STFT → Power Spectrum → Mel Filterbank → Log → (Normalize)
    
    Args:
        waveform: Tensor 1D (num_samples,)
        sample_rate: Tần số lấy mẫu (Hz)
        n_fft: Kích thước cửa sổ FFT
        hop_length: Bước nhảy giữa các frame (samples)
        win_length: Kích thước cửa sổ phân tích (samples)
        n_mels: Số Mel filterbank
        fmin: Tần số thấp nhất (Hz)
        fmax: Tần số cao nhất (Hz)
        normalize: Có z-score normalize per utterance không
    
    Returns:
        Tensor 2D (n_mels, T) — "bức ảnh" quang phổ Mel
        Trong đó T = ceil(num_samples / hop_length) + 1
    """
    # Tạo Mel-Spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        power=2.0,          # Power spectrogram (|S|²)
        normalized=False,
        norm="slaney",       # Slaney-style mel normalization
        mel_scale="slaney",
    )
    
    # Tính Mel-Spectrogram
    mel_spec = mel_transform(waveform)  # (n_mels, T)
    
    # Chuyển sang thang dB (log compression)
    # Thêm epsilon nhỏ (1e-9) để tránh log(0)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB(
        stype="power",
        top_db=80,
    )(mel_spec)
    
    # Z-score normalization per utterance
    if normalize:
        mel_spec_db = normalize_features(mel_spec_db)
    
    return mel_spec_db


def extract_lfcc(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_lfcc: int = 60,
    n_filter: int = 128,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Trích xuất LFCC (Linear Frequency Cepstral Coefficients) từ waveform.
    
    LFCC khác MFCC ở chỗ dùng thang tần số TUYẾN TÍNH thay vì thang Mel.
    Điều này giúp bảo toàn thông tin ở dải cao tần — nơi chứa nhiều artifact
    của các vocoder AI. Đây là đặc trưng được ASVspoof challenge khuyên dùng.
    
    Pipeline:
        Waveform → STFT → Power Spectrum → Linear Filterbank → Log → DCT → (Normalize)
    
    Args:
        waveform: Tensor 1D (num_samples,)
        sample_rate: Tần số lấy mẫu (Hz)
        n_lfcc: Số LFCC coefficients đầu ra
        n_filter: Số linear filterbank
        n_fft: Kích thước cửa sổ FFT
        hop_length: Bước nhảy giữa các frame
        win_length: Kích thước cửa sổ phân tích
        normalize: Có z-score normalize per utterance không
    
    Returns:
        Tensor 2D (n_lfcc, T) — ma trận LFCC
    """
    # Tạo LFCC transform
    # torchaudio cung cấp LFCC transform trực tiếp
    lfcc_transform = torchaudio.transforms.LFCC(
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
        n_filter=n_filter,
        speckwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "power": 2.0,
        },
    )
    
    # Tính LFCC
    lfcc = lfcc_transform(waveform)  # (n_lfcc, T)
    
    # Thêm delta và delta-delta (tùy chọn, tăng thông tin temporal)
    # Delta = đạo hàm bậc 1 theo thời gian
    # Delta-delta = đạo hàm bậc 2
    # Bỏ qua ở đây để giữ đơn giản, sẽ thêm ở Phase 2 nếu cần
    
    # Z-score normalization
    if normalize:
        lfcc = normalize_features(lfcc)
    
    return lfcc


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """
    Z-score normalization per utterance.
    
    Mỗi file audio có điều kiện thu âm khác nhau (âm lượng, môi trường).
    Normalize giúp model không bị ảnh hưởng bởi sự khác biệt âm lượng
    mà chỉ tập trung vào "hình dạng" (pattern) trên spectrogram.
    
    Công thức: x_norm = (x - mean) / (std + epsilon)
    
    Args:
        features: Tensor 2D (F, T) — F là số frequency bins, T là số frames
    
    Returns:
        Tensor 2D đã normalize, mean ≈ 0, std ≈ 1
    """
    mean = features.mean()
    std = features.std()
    # Thêm epsilon nhỏ để tránh chia cho 0
    return (features - mean) / (std + 1e-8)


def extract_features(
    waveform: torch.Tensor,
    feature_type: str = "lfcc",
    sample_rate: int = 16000,
    **kwargs,
) -> torch.Tensor:
    """
    Hàm convenience — chọn loại đặc trưng dựa trên tham số.
    
    Args:
        waveform: Tensor 1D (num_samples,)
        feature_type: "lfcc" hoặc "melspec"
        sample_rate: Tần số lấy mẫu
        **kwargs: Tham số bổ sung cho hàm extract cụ thể
    
    Returns:
        Tensor 2D (F, T) — đặc trưng đã trích xuất
    
    Raises:
        ValueError: Nếu feature_type không hợp lệ
    """
    if feature_type == "lfcc":
        return extract_lfcc(waveform, sample_rate=sample_rate, **kwargs)
    elif feature_type == "melspec":
        return extract_mel_spectrogram(waveform, sample_rate=sample_rate, **kwargs)
    else:
        raise ValueError(
            f"Unknown feature_type: '{feature_type}'. "
            f"Expected 'lfcc' or 'melspec'."
        )
