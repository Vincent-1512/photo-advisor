"""
Metrics — Hàm tính các chỉ số đánh giá mô hình
=================================================
Chỉ số chính: EER (Equal Error Rate)
  - Đây là metric CHUẨN cho bài toán Anti-Spoofing
  - EER = điểm giao nhau giữa False Acceptance Rate và False Rejection Rate
  - EER càng thấp càng tốt (0% = hoàn hảo, 50% = đoán ngẫu nhiên)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from typing import Tuple, Optional
from pathlib import Path


def compute_eer(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """
    Tính Equal Error Rate (EER).
    
    EER là điểm mà tại đó:
        False Acceptance Rate (FAR) == False Rejection Rate (FRR)
    
    - FAR = Tỷ lệ mẫu giả bị hệ thống chấp nhận nhầm (tưởng là thật)
    - FRR = Tỷ lệ mẫu thật bị hệ thống từ chối nhầm (tưởng là giả)
    
    Args:
        scores: Mảng 1D chứa confidence scores (xác suất là spoof)
                Giá trị cao hơn → model cho rằng là spoof
        labels: Mảng 1D chứa ground truth (0 = bonafide, 1 = spoof)
    
    Returns:
        eer: Equal Error Rate (0.0 → 1.0), ví dụ 0.03 = 3% EER
        threshold: Ngưỡng tối ưu tại EER
    """
    # Tính FAR và FRR tại mọi threshold bằng sklearn roc_curve
    # roc_curve trả về: fpr (= FAR), tpr (= 1 - FRR)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr  # FRR = 1 - True Positive Rate
    
    # Tìm điểm giao nhau FAR == FRR
    # Dùng nội suy (interpolation) để tìm chính xác
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    threshold = thresholds[eer_index]
    
    return float(eer), float(threshold)


def compute_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Tính Accuracy đơn giản.
    
    Args:
        predictions: Mảng 1D chứa predicted labels (0 hoặc 1)
        labels: Mảng 1D chứa ground truth labels (0 hoặc 1)
    
    Returns:
        Accuracy (0.0 → 1.0)
    """
    return float(accuracy_score(labels, predictions))


def compute_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    target_names: Optional[list] = None,
) -> str:
    """
    Tạo báo cáo phân loại chi tiết (Precision, Recall, F1-Score).
    
    Args:
        predictions: Mảng 1D chứa predicted labels
        labels: Mảng 1D chứa ground truth labels
        target_names: Tên các class (ví dụ: ["Bonafide", "Spoof"])
    
    Returns:
        Chuỗi text chứa báo cáo
    """
    if target_names is None:
        target_names = ["Bonafide", "Spoof"]
    
    return classification_report(
        labels, predictions,
        target_names=target_names,
        digits=4,
    )


def plot_det_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str | Path] = None,
    title: str = "DET Curve — Audio Deepfake Detection",
) -> None:
    """
    Vẽ DET Curve (Detection Error Tradeoff).
    
    DET curve giống ROC curve nhưng vẽ FRR vs FAR (thay vì TPR vs FPR).
    Đây là chuẩn visualization cho bài toán speaker verification / anti-spoofing.
    
    Args:
        scores: Mảng 1D chứa confidence scores
        labels: Mảng 1D chứa ground truth labels
        save_path: Đường dẫn lưu ảnh (None = chỉ hiển thị)
        title: Tiêu đề biểu đồ
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    eer, _ = compute_eer(scores, labels)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Vẽ DET curve
    ax.plot(fpr * 100, fnr * 100, linewidth=2, color="#2196F3", label="DET Curve")
    
    # Vẽ đường EER (FAR = FRR)
    ax.plot([0, 100], [0, 100], linestyle="--", color="#9E9E9E", alpha=0.5, label="Random")
    
    # Đánh dấu điểm EER
    ax.plot(eer * 100, eer * 100, "ro", markersize=10, label=f"EER = {eer*100:.2f}%")
    
    ax.set_xlabel("False Acceptance Rate (%)", fontsize=12)
    ax.set_ylabel("False Rejection Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"DET curve saved to: {save_path}")
    
    plt.close(fig)


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[list] = None,
    save_path: Optional[str | Path] = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    Vẽ Confusion Matrix.
    
    Args:
        predictions: Mảng 1D chứa predicted labels
        labels: Mảng 1D chứa ground truth labels
        class_names: Tên các class
        save_path: Đường dẫn lưu ảnh
        title: Tiêu đề
    """
    if class_names is None:
        class_names = ["Bonafide", "Spoof"]
    
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )
    
    # Hiển thị số lượng trong mỗi ô
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close(fig)
