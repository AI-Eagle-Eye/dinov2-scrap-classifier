from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# training.md에 정의된 5종 체크포인트
CHECKPOINT_METRICS: dict[str, str | None] = {
    "val_loss": "min",
    "f1": "max",
    "f2": "max",
    "safe_precision": "max",
    "last": None,
}


class CheckpointManager:
    """5종 체크포인트(val_loss, f1, f2, safe_precision, last)를 저장/로드."""

    def __init__(self, save_dir: str | Path) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._best: dict[str, float] = {
            "val_loss": float("inf"),
            "f1": 0.0,
            "f2": 0.0,
            "safe_precision": 0.0,
        }

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: dict[str, float],
        config_dict: dict[str, Any],
        seed: int,
    ) -> list[str]:
        """지표가 개선된 체크포인트를 저장하고 저장된 파일 경로 목록을 반환."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": config_dict,
            "seed": seed,
        }
        saved: list[str] = []

        for metric_name, direction in CHECKPOINT_METRICS.items():
            if metric_name == "last":
                path = self.save_dir / f"last_ep{epoch:03d}.ckpt"
                torch.save(checkpoint, path)
                saved.append(str(path))
                self._cleanup_old("last_ep")
                continue

            value = metrics.get(metric_name)
            if value is None:
                continue

            improved = (
                (direction == "min" and value < self._best[metric_name])
                or (direction == "max" and value > self._best[metric_name])
            )
            if improved:
                self._best[metric_name] = value
                fname = f"best_{metric_name}_ep{epoch:03d}_{value:.4f}.ckpt"
                path = self.save_dir / fname
                torch.save(checkpoint, path)
                saved.append(str(path))
                self._cleanup_old(f"best_{metric_name}_ep")

        return saved

    def _cleanup_old(self, prefix: str) -> None:
        """동일 metric의 이전 체크포인트를 제거해 디스크 낭비를 방지."""
        files = sorted(self.save_dir.glob(f"{prefix}*.ckpt"), key=lambda p: p.stat().st_mtime)
        for f in files[:-1]:
            f.unlink(missing_ok=True)

    @staticmethod
    def load(ckpt_path: str | Path) -> dict[str, Any]:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
