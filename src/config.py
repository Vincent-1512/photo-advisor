"""
Audio Deepfake Detection — Project Configuration
=================================================
Tập trung toàn bộ tham số cấu hình của dự án.
Hỗ trợ chạy trên cả local machine và Google Colab.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field


def _detect_environment() -> str:
    """Tự động phát hiện môi trường chạy (Colab hay Local)."""
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "local"


@dataclass
class PathConfig:
    """Cấu hình đường dẫn — tự động thay đổi theo môi trường."""
    
    environment: str = field(default_factory=_detect_environment)
    
    def __post_init__(self):
        if self.environment == "colab":
            # Google Colab: mount Google Drive
            self.project_root = Path("/content/drive/MyDrive/audio_deepfake")
        else:
            # Local machine
            self.project_root = Path(__file__).resolve().parent.parent
        
        # Dataset paths
        self.data_dir = self.project_root / "data"
        self.asvspoof_dir = self.data_dir / "asvspoof2019"
        self.wavefake_dir = self.data_dir / "wavefake"
        self.processed_dir = self.data_dir / "processed"
        self.processed_lfcc_dir = self.processed_dir / "lfcc"
        self.processed_melspec_dir = self.processed_dir / "melspec"
        
        # ASVspoof 2019 LA specific paths
        self.asvspoof_la_dir = self.asvspoof_dir / "LA"
        self.asvspoof_train_audio = self.asvspoof_la_dir / "ASVspoof2019_LA_train" / "flac"
        self.asvspoof_dev_audio = self.asvspoof_la_dir / "ASVspoof2019_LA_dev" / "flac"
        self.asvspoof_eval_audio = self.asvspoof_la_dir / "ASVspoof2019_LA_eval" / "flac"
        self.asvspoof_protocol_dir = self.asvspoof_la_dir / "ASVspoof2019_LA_cm_protocols"
        
        # Protocol file paths
        self.train_protocol = self.asvspoof_protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt"
        self.dev_protocol = self.asvspoof_protocol_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
        self.eval_protocol = self.asvspoof_protocol_dir / "ASVspoof2019.LA.cm.eval.trl.txt"
        
        # Output paths
        self.output_dir = self.project_root / "outputs"
        self.eda_output_dir = self.output_dir / "eda"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
    
    def ensure_dirs(self):
        """Tạo tất cả thư mục cần thiết nếu chưa tồn tại."""
        dirs_to_create = [
            self.data_dir,
            self.processed_lfcc_dir,
            self.processed_melspec_dir,
            self.eda_output_dir,
            self.checkpoint_dir,
            self.log_dir,
        ]
        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class AudioConfig:
    """Tham số xử lý âm thanh."""
    
    sample_rate: int = 16000          # Hz — chuẩn cho speech processing
    duration: float = 4.0             # Giây — độ dài cố định mỗi mẫu
    target_length: int = 64000        # = sample_rate * duration (samples)
    
    # STFT parameters
    n_fft: int = 512                  # FFT window size
    hop_length: int = 160             # = 10ms at 16kHz → phân giải thời gian cao
    win_length: int = 400             # = 25ms at 16kHz — chuẩn speech frame
    
    # Mel-Spectrogram
    n_mels: int = 80                  # Số Mel filterbank
    fmin: float = 0.0                 # Hz
    fmax: float = 8000.0              # Hz — Nyquist frequency at 16kHz
    
    # LFCC
    n_lfcc: int = 60                  # Số LFCC coefficients
    n_filter: int = 128               # Số linear filterbank
    
    # Pre-emphasis
    pre_emphasis: float = 0.97


@dataclass
class TrainingConfig:
    """Tham số huấn luyện."""
    
    # Dataloader
    batch_size: int = 32
    num_workers: int = 4              # Colab thường dùng 2
    pin_memory: bool = True
    
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    
    # Training
    epochs: int = 30
    early_stopping_patience: int = 5
    gradient_clip_max_norm: float = 1.0
    
    # Model
    backbone: str = "lcnn"            # "lcnn" | "resnet18" | "rawnet2"
    feature_type: str = "lfcc"        # "lfcc" | "melspec"
    
    # Multi-task loss weights (Phase 2+)
    alpha_binary: float = 1.0         # Level 1 weight
    beta_attack: float = 0.5          # Level 2 weight
    gamma_model: float = 0.3          # Level 3 weight
    delta_center: float = 0.01        # Center loss weight
    
    # Reproducibility
    seed: int = 42


@dataclass
class Config:
    """Master config — gom tất cả config con lại."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def setup(self):
        """Khởi tạo môi trường: tạo thư mục, set seed."""
        self.paths.ensure_dirs()
        self._set_seed()
        return self
    
    def _set_seed(self):
        """Đảm bảo reproducibility."""
        import random
        import numpy as np
        import torch
        
        random.seed(self.training.seed)
        np.random.seed(self.training.seed)
        torch.manual_seed(self.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.training.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def summary(self) -> str:
        """In tóm tắt cấu hình."""
        import torch
        
        lines = [
            "=" * 60,
            "  AUDIO DEEPFAKE DETECTION — CONFIG SUMMARY",
            "=" * 60,
            f"  Environment:     {self.paths.environment}",
            f"  Project root:    {self.paths.project_root}",
            f"  Device:          {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}",
            f"  Sample rate:     {self.audio.sample_rate} Hz",
            f"  Duration:        {self.audio.duration}s ({self.audio.target_length} samples)",
            f"  Feature type:    {self.training.feature_type}",
            f"  Backbone:        {self.training.backbone}",
            f"  Batch size:      {self.training.batch_size}",
            f"  Learning rate:   {self.training.learning_rate}",
            f"  Epochs:          {self.training.epochs}",
            f"  Seed:            {self.training.seed}",
            "=" * 60,
        ]
        return "\n".join(lines)


# Global config instance — import từ bất kỳ đâu
cfg = Config()
