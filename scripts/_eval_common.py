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

from src.models.hazard_model import HazardModel, ModelConfig, VPTConfig

_DANGER_IDX: int = 1


def detect_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        sys.exit(f"[ERROR] config not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict[str, Any], ckpt_path: Path, device: torch.device) -> HazardModel:
    m = cfg["model"]
    vpt_raw = m.get("vpt", {})
    model_cfg = ModelConfig(
        backbone_name=m.get("backbone_name", "dinov2_vitb14"),
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
    )
    model = HazardModel(model_cfg)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


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


def apply_threshold(probs: np.ndarray, threshold: float) -> list[int]:
    """danger prob >= threshold → danger(1); 그 외 argmax."""
    preds: list[int] = probs.argmax(axis=1).tolist()
    for i in range(len(preds)):
        if probs[i, _DANGER_IDX] >= threshold:
            preds[i] = _DANGER_IDX
    return preds
