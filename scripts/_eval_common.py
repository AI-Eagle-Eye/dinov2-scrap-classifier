"""Shared evaluation helpers for evaluate_test.py and tune_threshold.py.

두 평가 스크립트의 모델 빌드/추론 로직을 한곳에 모아 동기화 누락 버그를 방지한다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import DANGER, HazardDataset
from src.models.hazard_model import HazardModel, ModelConfig, VPTConfig

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DINOV2_PATCH: int = 14

# 출력 레이어 전용 표시명. 코드 내부 키는 절대 변경하지 않고, 리포트/CSV/PNG로 내보낼 때만 변환한다.
DISPLAY_NAME_MAP: dict[str, str] = {
    "danger_as_safe_rate": "Miss Rate (Danger)",
    "danger_precision":    "Precision (Danger)",
    "danger_recall":       "Recall (Danger)",
    "cut_precision":       "Precision (Cut)",
    "cut_recall":          "Recall (Cut)",
    "excluded_precision":  "Precision (Excluded)",
    "excluded_recall":     "Recall (Excluded)",
    "f1_macro":            "Macro F1",
    "f2_macro":            "Macro F2",
    "f1_cut":              "F1 (Cut)",
    "f1_danger":           "F1 (Danger)",
    "f1_excluded":         "F1 (Excluded)",
    "coverage":            "Coverage (자동판정율)",
    # evaluator.compute_metrics가 내보내는 precision_<cls>/recall_<cls> 형태 별칭
    "precision_danger":    "Precision (Danger)",
    "recall_danger":       "Recall (Danger)",
    "precision_cut":       "Precision (Cut)",
    "recall_cut":          "Recall (Cut)",
    "precision_excluded":  "Precision (Excluded)",
    "recall_excluded":     "Recall (Excluded)",
    "safe_precision":      "Precision (Cut)",
    "accuracy":            "Accuracy",
}


def display_name(key: str) -> str:
    """내부 지표 키를 출력용 표시명으로 변환. 매핑이 없으면 키를 그대로 반환."""
    return DISPLAY_NAME_MAP.get(key, key)


def assert_patch_compatible(image_size: int, backbone_name: str = "dinov2_vitb14") -> None:
    """DINOv2 patch14 백본은 image_size가 14의 배수여야 한다."""
    if "14" in backbone_name:
        assert image_size % _DINOV2_PATCH == 0, (
            f"image_size {image_size} must be divisible by {_DINOV2_PATCH}"
        )


def detect_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        sys.exit(f"[ERROR] config not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_config_from_cfg(cfg: dict[str, Any]) -> ModelConfig:
    """config dict → ModelConfig. backbone/vpt/head 필드를 빠짐없이 매핑하는 단일 출처."""
    m = cfg["model"]
    vpt_raw = m.get("vpt", {})
    backbone_raw = m.get("backbone", {})
    return ModelConfig(
        backbone_name=m.get("backbone_name", "dinov2_vitb14"),
        backbone_frozen=backbone_raw.get("frozen", True),
        unfreeze_last_n=backbone_raw.get("unfreeze_last_n", 0),
        head_type=m.get("head_type", "mlp"),
        vpt=VPTConfig(
            enabled=vpt_raw.get("enabled", False),
            num_tokens=vpt_raw.get("num_tokens", 10),
            insert_from_layer=vpt_raw.get("insert_from_layer", 0),
        ),
        dropout=m.get("dropout", 0.3),
        num_classes=m.get("num_classes", 3),
        use_grad_checkpoint=m.get("use_grad_checkpoint", True),
        class_aware_init_weights=m.get("class_aware_init_weights", None),
        head_use_cls=m.get("head_use_cls", False),
    )


def build_model(cfg: dict[str, Any], ckpt_path: Path | None, device: torch.device) -> HazardModel:
    """config로 HazardModel을 만들고 ckpt_path가 있으면 가중치를 로드한다.

    ckpt_path=None이면 사전학습 backbone만 가진 모델(평가 'before' 상태)을 반환한다.
    전체 학습 체크포인트(trainer.py 포맷)와 plain state_dict(.pth) 모두 지원한다.
    """
    model = HazardModel(model_config_from_cfg(cfg))
    if ckpt_path is not None:
        raw = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
        model.load_state_dict(state)
    model.to(device).eval()
    return model


def read_checkpoint_epoch(ckpt_path: Path, device: torch.device) -> int | None:
    """전체 학습 체크포인트의 epoch 메타 (plain state_dict면 None)."""
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    return raw.get("epoch") if isinstance(raw, dict) else None


def resolve_checkpoint(cfg: dict[str, Any], explicit: Path | None) -> Path:
    """explicit 우선; 없으면 experiments/<name>/checkpoints의 최신 best_val_loss_*.ckpt."""
    if explicit is not None:
        if not explicit.exists():
            sys.exit(f"[ERROR] checkpoint not found: {explicit}")
        return explicit
    ckpt_dir = _PROJECT_ROOT / "experiments" / cfg["experiment"]["name"] / "checkpoints"
    candidates = sorted(ckpt_dir.glob("best_val_loss_*.ckpt"))
    if not candidates:
        sys.exit(f"[ERROR] best_val_loss_*.ckpt not found in {ckpt_dir}")
    return candidates[-1]


def dataset_kwargs(d: dict[str, Any], use_eval_label_col: bool = False) -> dict[str, Any]:
    """HazardDataset 공통 kwargs. use_eval_label_col=True면 eval_label_col을 우선한다."""
    label_col = (
        d.get("eval_label_col", d.get("label_col", "confirmed_label"))
        if use_eval_label_col
        else d.get("label_col", "confirmed_label")
    )
    kw: dict[str, Any] = {
        "unk_label": d.get("unk_label", "excluded"),
        "label_col": label_col,
        "split_col": d.get("split_col", "split"),
    }
    if "csv_path" in d:
        kw["csv_path"] = Path(d["csv_path"])
    return kw


def collect_cases(
    dataset: HazardDataset,
    probs: np.ndarray,
    true_cls: int,
    pred_cls: int,
    n: int,
) -> list[tuple[Path, int, int, np.ndarray]]:
    """(img_path, true_idx, pred_idx, prob_vec) n개 수집, danger_prob 내림차순 정렬."""
    results: list[tuple[Path, int, int, np.ndarray]] = []
    labels = dataset.labels
    paths = [p for p, _ in dataset._samples]
    for i, true in enumerate(labels):
        pred = int(probs[i].argmax())
        if true == true_cls and pred == pred_cls:
            results.append((paths[i], true, pred, probs[i]))
    results.sort(key=lambda r: -r[3][DANGER])
    return results[:n]


@torch.inference_mode()
def collect_probs(
    model: HazardModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """전체 추론 → (probs [N, 3], labels [N])."""
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []
    for images, labels in loader:
        logits = model(images.to(device))
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.tolist())
    return np.concatenate(all_probs, axis=0), np.array(all_labels, dtype=int)


@torch.inference_mode()
def collect_outputs(
    model: HazardModel,
    loader: DataLoader,
    device: torch.device,
    tta: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """전체 추론 → (logits [N, 3], probs [N, 3], labels [N]).

    tta=True: 원본 + hflip의 logits/softmax를 평균한다 (probs는 softmax 평균).
    """
    all_logits: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        if tta:
            logits_flip = model(torch.flip(images, dims=[3]))
            probs = (torch.softmax(logits, dim=1) + torch.softmax(logits_flip, dim=1)) / 2
            logits = (logits + logits_flip) / 2
        else:
            probs = torch.softmax(logits, dim=1)
        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.extend(labels.tolist())
    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_probs, axis=0),
        np.array(all_labels, dtype=int),
    )


def apply_threshold(probs: np.ndarray, threshold: float) -> list[int]:
    """danger prob >= threshold → danger(1); 그 외 argmax (list 반환, 하위 스크립트 호환)."""
    from src.evaluation.threshold import apply_danger_threshold  # noqa: PLC0415

    return apply_danger_threshold(probs, threshold).tolist()
