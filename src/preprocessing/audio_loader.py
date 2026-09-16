"""
Audio Loader — Load và chuẩn hóa file âm thanh
================================================
Module chịu trách nhiệm:
  1. Load file audio (WAV/FLAC/MP3) → waveform tensor
  2. Resample về tần số chuẩn (16kHz)
  3. Pad hoặc Trim về độ dài cố định (4 giây = 64,000 samples)
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_audio(
    file_path: str | Path,
    target_sr: int = 16000,
) -> Tuple[torch.Tensor, int]:
    """
    Load file audio và resample về target sample rate.
    
    Args:
        file_path: Đường dẫn đến file audio (.wav, .flac, .mp3)
        target_sr: Tần số lấy mẫu mục tiêu (mặc định 16000 Hz)
    
    Returns:
        waveform: Tensor 1D (num_samples,)
        sample_rate: Tần số lấy mẫu sau resample
    
    Raises:
        FileNotFoundError: Nếu file không tồn tại
        RuntimeError: Nếu file không thể đọc được
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Load audio bằng torchaudio (nhanh hơn librosa cho PyTorch pipeline)
    waveform, sr = torchaudio.load(str(file_path))
    
    # Chuyển stereo → mono nếu cần (lấy trung bình 2 kênh)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Squeeze về 1D: (1, N) → (N,)
    waveform = waveform.squeeze(0)
    
    # Resample nếu sample rate khác target
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=target_sr
        )
        waveform = resampler(waveform)
    
    return waveform, target_sr


def pad_or_trim(
    waveform: torch.Tensor,
    target_length: int = 64000,
) -> torch.Tensor:
    """
    Chuẩn hóa waveform về đúng target_length samples.
    
    - Nếu ngắn hơn → Zero-pad (thêm giá trị 0 vào cuối)
    - Nếu dài hơn → Cắt bỏ phần thừa (giữ phần đầu)
    
    Args:
        waveform: Tensor 1D (num_samples,)
        target_length: Số samples mục tiêu (mặc định 64000 = 4s × 16kHz)
    
    Returns:
        Tensor 1D có đúng target_length samples
    """
    current_length = waveform.shape[0]
    
    if current_length == target_length:
        return waveform
    elif current_length > target_length:
        # Trim: cắt bỏ phần cuối
        return waveform[:target_length]
    else:
        # Pad: thêm zeros vào cuối
        padding = target_length - current_length
        return torch.nn.functional.pad(waveform, (0, padding), value=0.0)


def pad_or_trim_random_crop(
    waveform: torch.Tensor,
    target_length: int = 64000,
) -> torch.Tensor:
    """
    Giống pad_or_trim nhưng khi file dài hơn target_length,
    sẽ random crop thay vì luôn lấy phần đầu.
    Dùng cho data augmentation trong quá trình training.
    
    Args:
        waveform: Tensor 1D (num_samples,)
        target_length: Số samples mục tiêu
    
    Returns:
        Tensor 1D có đúng target_length samples
    """
    current_length = waveform.shape[0]
    
    if current_length == target_length:
        return waveform
    elif current_length > target_length:
        # Random crop: chọn ngẫu nhiên điểm bắt đầu
        max_start = current_length - target_length
        start = np.random.randint(0, max_start + 1)
        return waveform[start:start + target_length]
    else:
        # Pad: thêm zeros vào cuối
        padding = target_length - current_length
        return torch.nn.functional.pad(waveform, (0, padding), value=0.0)


def apply_pre_emphasis(
    waveform: torch.Tensor,
    coefficient: float = 0.97,
) -> torch.Tensor:
    """
    Áp dụng pre-emphasis filter để tăng cường dải cao tần.
    
    Công thức: y[n] = x[n] - coefficient * x[n-1]
    
    Đây là bước quan trọng cho speech processing vì giọng nói con người
    có năng lượng tập trung ở dải thấp. Pre-emphasis cân bằng lại phổ
    để CNN có thể "nhìn" rõ hơn ở dải cao — nơi chứa artifact của AI.
    
    Args:
        waveform: Tensor 1D (num_samples,)
        coefficient: Hệ số pre-emphasis (mặc định 0.97)
    
    Returns:
        Tensor 1D đã được pre-emphasize
    """
    # y[0] = x[0], y[n] = x[n] - coeff * x[n-1]
    emphasized = torch.cat([
        waveform[:1],
        waveform[1:] - coefficient * waveform[:-1]
    ])
    return emphasized


def load_and_preprocess(
    file_path: str | Path,
    target_sr: int = 16000,
    target_length: int = 64000,
    pre_emphasis_coeff: Optional[float] = 0.97,
    random_crop: bool = False,
) -> torch.Tensor:
    """
    Pipeline hoàn chỉnh: Load → Resample → Pad/Trim → Pre-emphasis.
    
    Đây là hàm convenience gom tất cả các bước lại thành 1 lệnh duy nhất.
    
    Args:
        file_path: Đường dẫn đến file audio
        target_sr: Tần số lấy mẫu mục tiêu
        target_length: Số samples mục tiêu
        pre_emphasis_coeff: Hệ số pre-emphasis (None = bỏ qua)
        random_crop: Dùng random crop thay vì trim (cho training)
    
    Returns:
        Tensor 1D đã được chuẩn hóa
    """
    # Step 1: Load + resample
    waveform, _ = load_audio(file_path, target_sr)
    
    # Step 2: Pad/Trim
    if random_crop:
        waveform = pad_or_trim_random_crop(waveform, target_length)
    else:
        waveform = pad_or_trim(waveform, target_length)
    
    # Step 3: Pre-emphasis (optional)
    if pre_emphasis_coeff is not None:
        waveform = apply_pre_emphasis(waveform, pre_emphasis_coeff)
    
    return waveform
