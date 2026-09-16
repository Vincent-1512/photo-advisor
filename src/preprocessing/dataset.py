"""
PyTorch Dataset — ASVspoof 2019 LA Dataset
============================================
Class kế thừa torch.utils.data.Dataset, phục vụ cho DataLoader.
Chịu trách nhiệm:
  1. Đọc protocol file → danh sách (file_path, label)
  2. Load audio → trích xuất features on-the-fly
  3. Trả về (feature_tensor, label_dict) cho training loop
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Literal

from .audio_loader import load_and_preprocess
from .feature_extraction import extract_features


class ASVspoofDataset(Dataset):
    """
    PyTorch Dataset cho ASVspoof 2019 LA.
    
    Mỗi sample trả về:
        features: Tensor (1, F, T) — ảnh quang phổ (1 channel)
        labels: Dict chứa nhãn cho cả 3 cấp độ
            - "binary": 0 (bonafide) hoặc 1 (spoof)
            - "attack_type": 0-3 (bonafide/TTS/VC/Replay)
            - "system_id": string ID của attack system
    
    Usage:
        from src.preprocessing.dataset import ASVspoofDataset
        from src.config import cfg
        
        dataset = ASVspoofDataset(
            protocol_path=cfg.paths.train_protocol,
            audio_dir=cfg.paths.asvspoof_train_audio,
            feature_type="lfcc",
        )
        features, labels = dataset[0]
        print(features.shape)  # torch.Size([1, 60, 401])
        print(labels)          # {"binary": 0, "attack_type": 0, "system_id": "-"}
    """
    
    def __init__(
        self,
        protocol_path: str | Path,
        audio_dir: str | Path,
        feature_type: Literal["lfcc", "melspec"] = "lfcc",
        sample_rate: int = 16000,
        duration: float = 4.0,
        pre_emphasis: Optional[float] = 0.97,
        random_crop: bool = False,
        normalize: bool = True,
    ):
        """
        Args:
            protocol_path: Đường dẫn đến file protocol (.txt)
            audio_dir: Thư mục chứa file audio (.flac)
            feature_type: Loại đặc trưng: "lfcc" hoặc "melspec"
            sample_rate: Tần số lấy mẫu mục tiêu
            duration: Độ dài mỗi mẫu (giây)
            pre_emphasis: Hệ số pre-emphasis (None = bỏ qua)
            random_crop: Dùng random crop (True cho training, False cho eval)
            normalize: Z-score normalize features
        """
        super().__init__()
        
        self.audio_dir = Path(audio_dir)
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration)
        self.pre_emphasis = pre_emphasis
        self.random_crop = random_crop
        self.normalize = normalize
        
        # Parse protocol file
        self.samples = self._parse_protocol(protocol_path)
        
        # Mapping system_id → attack type
        from ..utils.protocol_parser import SYSTEM_TO_ATTACK_TYPE
        self.system_to_attack = SYSTEM_TO_ATTACK_TYPE
    
    def _parse_protocol(self, protocol_path: str | Path) -> list:
        """Đọc protocol file thành danh sách samples."""
        protocol_path = Path(protocol_path)
        if not protocol_path.exists():
            raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
        
        samples = []
        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                
                speaker_id = parts[0]
                filename = parts[1]
                system_id = parts[3]
                label_str = parts[4] if len(parts) > 4 else "bonafide"
                
                audio_path = self.audio_dir / f"{filename}.flac"
                
                # Bỏ qua nếu file audio không tồn tại
                if not audio_path.exists():
                    continue
                
                binary_label = 1 if label_str == "spoof" else 0
                
                # Attack type mapping
                if label_str == "bonafide":
                    attack_type_id = 0
                else:
                    attack_type = self.system_to_attack.get(system_id, "unknown") if hasattr(self, 'system_to_attack') else "unknown"
                    attack_mapping = {"bonafide": 0, "TTS": 1, "VC": 2, "Replay": 3}
                    attack_type_id = attack_mapping.get(attack_type, -1)
                
                samples.append({
                    "audio_path": str(audio_path),
                    "speaker_id": speaker_id,
                    "filename": filename,
                    "system_id": system_id,
                    "binary_label": binary_label,
                    "attack_type_id": attack_type_id,
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int):
        """
        Lấy 1 sample: load audio → trích xuất features → trả về tensor + labels.
        
        Returns:
            features: Tensor (1, F, T) — 1 channel, F frequency bins, T time frames
            labels: Dict với keys "binary", "attack_type", "system_id"
        """
        sample = self.samples[index]
        
        # Step 1: Load + preprocess audio → waveform 1D
        waveform = load_and_preprocess(
            file_path=sample["audio_path"],
            target_sr=self.sample_rate,
            target_length=self.target_length,
            pre_emphasis_coeff=self.pre_emphasis,
            random_crop=self.random_crop,
        )
        
        # Step 2: Trích xuất features → tensor 2D (F, T)
        features = extract_features(
            waveform=waveform,
            feature_type=self.feature_type,
            sample_rate=self.sample_rate,
        )
        
        # Step 3: Thêm channel dimension → (1, F, T) cho CNN
        features = features.unsqueeze(0)
        
        # Step 4: Labels
        labels = {
            "binary": torch.tensor(sample["binary_label"], dtype=torch.long),
            "attack_type": torch.tensor(sample["attack_type_id"], dtype=torch.long),
            "system_id": sample["system_id"],
        }
        
        return features, labels
    
    def get_label_distribution(self) -> dict:
        """Thống kê phân bổ nhãn trong dataset."""
        from collections import Counter
        
        binary_dist = Counter(s["binary_label"] for s in self.samples)
        attack_dist = Counter(s["attack_type_id"] for s in self.samples)
        system_dist = Counter(s["system_id"] for s in self.samples)
        
        return {
            "binary": dict(binary_dist),
            "attack_type": dict(attack_dist),
            "system_id": dict(system_dist),
            "total": len(self.samples),
        }


def create_dataloaders(
    train_protocol: str | Path,
    train_audio_dir: str | Path,
    dev_protocol: str | Path,
    dev_audio_dir: str | Path,
    feature_type: str = "lfcc",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Tạo DataLoader cho train và dev set.
    
    Convenience function gom việc tạo Dataset + DataLoader.
    
    Returns:
        train_loader: DataLoader cho training (shuffle=True, random_crop=True)
        dev_loader: DataLoader cho validation (shuffle=False)
    """
    from torch.utils.data import DataLoader
    
    # Train dataset: random crop + shuffle
    train_dataset = ASVspoofDataset(
        protocol_path=train_protocol,
        audio_dir=train_audio_dir,
        feature_type=feature_type,
        random_crop=True,   # Data augmentation
    )
    
    # Dev dataset: deterministic
    dev_dataset = ASVspoofDataset(
        protocol_path=dev_protocol,
        audio_dir=dev_audio_dir,
        feature_type=feature_type,
        random_crop=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,     # Bỏ batch cuối nếu không đủ batch_size
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    print(f"Train: {len(train_dataset):,} samples → {len(train_loader):,} batches")
    print(f"Dev:   {len(dev_dataset):,} samples → {len(dev_loader):,} batches")
    
    return train_loader, dev_loader
