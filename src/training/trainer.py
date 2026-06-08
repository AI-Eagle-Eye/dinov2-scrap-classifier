from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..evaluation.evaluator import compute_metrics, save_confusion_matrix
from .checkpoint import CheckpointManager
from .danger_recall_loss import DangerRecallLoss, FocalDangerRecallLoss
from .focal_loss import FocalLoss, LearnableClassWeight
from .supcon_loss import SupConLoss

# BF16이 FP16보다 수치 안정적이고 RTX 4060/4090에서 GradScaler 불필요
_AMP_DTYPE = torch.bfloat16

# danger_as_safe_rate가 이 값 미만일 때만 best 갱신/early stopping 카운트 진행
DANGER_AS_SAFE_LIMIT = 0.15


def set_seed(seed: int = 42) -> None:
    """전역 시드 고정으로 실험 재현성 보장."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(slots=True)
class EarlyStopping:
    patience: int = 10
    min_delta: float = 1e-4
    best_value: float = field(default=float("-inf"))  # higher is better (danger_precision)
    counter: int = 0

    def update(self, value: float, das: float) -> bool:
        """danger_as_safe < DANGER_AS_SAFE_LIMIT일 때만 danger_precision(높을수록 좋음) 개선 여부 확인.

        das >= DANGER_AS_SAFE_LIMIT면 카운트 증가 없이 False 반환.
        """
        if das >= DANGER_AS_SAFE_LIMIT:
            return False
        if value > self.best_value + self.min_delta:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def _build_optimizer(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    learnable_weight: LearnableClassWeight | None = None,
) -> tuple[Lion, bool]:
    """backbone/head 분리 파라미터 그룹으로 Lion 옵티마이저 생성.

    backbone_lr=0이면 backbone을 frozen(requires_grad=False)하고 optimizer에서 제외.
    Returns (optimizer, has_backbone_group).
    """
    backbone: nn.Module | None = getattr(model, "backbone", None)

    if backbone is not None:
        for p in backbone.parameters():
            p.requires_grad = backbone_lr > 0
        backbone_ids = {id(p) for p in backbone.parameters()}
        head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    else:
        head_params = [p for p in model.parameters() if p.requires_grad]

    has_backbone_group = backbone is not None and backbone_lr > 0

    if has_backbone_group:
        param_groups: list[dict[str, Any]] = [
            {"params": list(backbone.parameters()), "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
    else:
        param_groups = [{"params": head_params, "lr": head_lr}]

    if learnable_weight is not None:
        param_groups.append({
            "params": list(learnable_weight.parameters()),
            "lr": head_lr,
            "weight_decay": 0.0,
        })

    return Lion(param_groups, weight_decay=weight_decay), has_backbone_group


def _build_scheduler(
    optimizer: Lion,
    epochs: int,
    warmup_ratio: float,
) -> SequentialLR:
    """Linear warmup + CosineAnnealingLR."""
    warmup_epochs = max(1, int(epochs * warmup_ratio))
    cosine_epochs = max(1, epochs - warmup_epochs)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


class Trainer:
    """학습 루프, early stopping, CSV 로깅을 담당."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        ckpt_manager: CheckpointManager,
        log_dir: str | Path,
        config_dict: dict[str, Any],
        class_weights: torch.Tensor | None = None,
        device: torch.device | str = "cuda",
        backbone_lr: float = 0.0,
        lr: float = 1e-5,
        weight_decay: float = 1e-2,
        epochs: int = 30,
        warmup_ratio: float = 0.1,
        early_stopping_patience: int = 10,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
        seed: int = 42,
        use_learnable_weight: bool = False,
        loss_type: Literal["focal", "supcon", "ce", "ce_dr", "focal_dr"] = "focal",
        lambda_dr: float = 0.0,
        num_classes: int = 3,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ckpt_manager = ckpt_manager
        self.log_dir = Path(log_dir)
        self.config_dict = config_dict
        self.device = torch.device(device) if isinstance(device, str) else device
        self.epochs = epochs
        self.seed = seed
        self._focal_gamma = focal_gamma
        self._label_smoothing = label_smoothing

        self._use_amp: bool = self.device.type == "cuda"

        if use_learnable_weight:
            init_w = class_weights.tolist() if class_weights is not None else [1.0] * num_classes
            self.learnable_weight: LearnableClassWeight | None = (
                LearnableClassWeight(num_classes, init_w).to(self.device)
            )
            self.criterion: FocalLoss | LearnableClassWeight | DangerRecallLoss | FocalDangerRecallLoss = self.learnable_weight
        else:
            self.learnable_weight = None
            w = class_weights.to(self.device) if class_weights is not None else None
            if loss_type in ("ce", "ce_dr"):
                _lambda = lambda_dr if loss_type == "ce_dr" else 0.0
                self.criterion = DangerRecallLoss(lambda_dr=_lambda, label_smoothing=label_smoothing)
            elif loss_type == "focal_dr":
                self.criterion = FocalDangerRecallLoss(
                    lambda_dr=lambda_dr, gamma=focal_gamma, label_smoothing=label_smoothing, weight=w
                )
            else:
                self.criterion = FocalLoss(gamma=focal_gamma, weight=w, label_smoothing=label_smoothing)

        self.supcon: SupConLoss | None = SupConLoss() if loss_type == "supcon" else None

        self.optimizer, self._has_backbone_group = _build_optimizer(
            model, backbone_lr, lr, weight_decay, self.learnable_weight
        )
        self.scheduler = _build_scheduler(self.optimizer, epochs, warmup_ratio)

        self.early_stopping = EarlyStopping(patience=early_stopping_patience)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.log_dir / "train_log.csv"
        self._cm_path = self.log_dir / "confusion_matrix.csv"
        self._best_model_path = self.ckpt_manager.save_dir / "best_model.pth"
        self._best_danger_prec: float = float("-inf")
        self._best_acc: float = 0.0
        self._best_epoch: int = 0
        self._init_log_csv()

    def _get_lrs(self) -> tuple[float, float]:
        """(backbone_lr, head_lr) 반환. backbone group이 없으면 backbone_lr=0."""
        groups = self.optimizer.param_groups
        if self._has_backbone_group:
            return groups[0]["lr"], groups[1]["lr"]
        return 0.0, groups[0]["lr"]

    def _call_criterion(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """FocalLoss / LearnableClassWeight 통합 호출 — 설정된 gamma/label_smoothing 사용."""
        if self.learnable_weight is not None:
            return self.learnable_weight(logits, labels, self._focal_gamma, self._label_smoothing)
        return self.criterion(logits, labels)

    def _init_log_csv(self) -> None:
        with self._csv_path.open("w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss", "accuracy",
                "f1_macro", "danger_precision", "danger_as_safe", "lr_backbone", "lr_head", "timestamp",
            ])

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        for images, labels in tqdm(self.train_loader, desc="train", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=_AMP_DTYPE, enabled=self._use_amp):
                if self.supcon is not None:
                    features, logits = self.model.forward_features(images)  # type: ignore[operator]
                    loss = self.supcon(features, labels) + self._call_criterion(logits, labels) * 0.5
                else:
                    logits = self.model(images)
                    loss = self._call_criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.extend(logits.detach().argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        metrics = compute_metrics(all_labels, all_preds)
        metrics["loss"] = total_loss / len(self.train_loader)
        return metrics

    @torch.inference_mode()
    def _validate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        for images, labels in tqdm(self.val_loader, desc="val", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=_AMP_DTYPE, enabled=self._use_amp):
                logits = self.model(images)
                total_loss += self._call_criterion(logits, labels).item()
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        save_confusion_matrix(all_labels, all_preds, self._cm_path)
        metrics = compute_metrics(all_labels, all_preds)
        metrics["loss"] = total_loss / len(self.val_loader)
        return metrics

    def _update_best_model(self, val_m: dict[str, float]) -> bool:
        """danger_as_safe < DANGER_AS_SAFE_LIMIT 조건 하에 danger_precision 최대 기준 best 갱신. 갱신 여부 반환."""
        if val_m["danger_as_safe_rate"] >= DANGER_AS_SAFE_LIMIT:
            return False
        acc = val_m["accuracy"]
        danger_prec = val_m["precision_danger"]
        improved = danger_prec > self._best_danger_prec or (
            danger_prec == self._best_danger_prec and acc > self._best_acc
        )
        if improved:
            self._best_danger_prec = danger_prec
            self._best_acc = acc
            torch.save(self.model.state_dict(), self._best_model_path)
        return improved

    def _save_plots(self) -> None:
        """train_log.csv를 읽어 4개 학습 그래프를 plots/ 폴더에 저장."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = self.log_dir.parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        epochs: list[int] = []
        train_losses: list[float] = []
        val_losses: list[float] = []
        val_accs: list[float] = []
        val_f1s: list[float] = []
        danger_precs: list[float] = []
        danger_as_safes: list[float] = []

        with self._csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]))
                train_losses.append(float(row["train_loss"]))
                val_losses.append(float(row["val_loss"]))
                val_accs.append(float(row["accuracy"]))
                val_f1s.append(float(row["f1_macro"]))
                danger_precs.append(float(row["danger_precision"]))
                danger_as_safes.append(float(row["danger_as_safe"]))

        def _vline(ax: plt.Axes) -> None:
            if self._best_epoch > 0:
                ax.axvline(
                    x=self._best_epoch, color="green", linestyle="--",
                    linewidth=1.0, label=f"best (ep{self._best_epoch})",
                )

        # ── 1. loss_curve ─────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, train_losses, color="blue", label="train_loss")
        ax.plot(epochs, val_losses, color="orange", label="val_loss")
        _vline(ax)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Loss Curve")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(plots_dir / "loss_curve.png", dpi=100)
        plt.close(fig)

        # ── 2. accuracy_curve ─────────────────────────────────────────────────
        # CSV에 train_acc 컬럼 없음 → val_acc만 표시
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, val_accs, color="orange", label="val_acc")
        _vline(ax)
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy Curve")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(plots_dir / "accuracy_curve.png", dpi=100)
        plt.close(fig)

        # ── 3. f1_curve ───────────────────────────────────────────────────────
        # CSV에 train_f1 컬럼 없음 → val_f1만 표시
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, val_f1s, color="orange", label="val_f1_macro")
        _vline(ax)
        ax.set_xlabel("epoch")
        ax.set_ylabel("f1_macro")
        ax.set_title("F1 Macro Curve")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(plots_dir / "f1_curve.png", dpi=100)
        plt.close(fig)

        # ── 4. danger_metrics ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, danger_precs, color="blue", label="danger_precision")
        ax.plot(epochs, danger_as_safes, color="red", label="danger_as_safe_rate")
        ax.axhline(y=0.10, color="gray", linestyle=":", linewidth=1.0, label="threshold=0.10")
        _vline(ax)
        ax.set_xlabel("epoch")
        ax.set_ylabel("rate")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Danger Metrics")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(plots_dir / "danger_metrics.png", dpi=100)
        plt.close(fig)

    def _log_epoch(self, epoch: int, train_m: dict[str, float], val_m: dict[str, float]) -> None:
        lr_bb, lr_head = self._get_lrs()
        with self._csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_m['loss']:.4f}",
                f"{val_m['loss']:.4f}",
                f"{val_m['accuracy']:.4f}",
                f"{val_m['f1_macro']:.4f}",
                f"{val_m['precision_danger']:.4f}",
                f"{val_m['danger_as_safe_rate']:.4f}",
                f"{lr_bb:.2e}",
                f"{lr_head:.2e}",
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ])

    def fit(self, verbose: bool = True) -> dict[str, float]:
        """학습 루프 실행. 최종 best val 지표를 반환."""
        set_seed(self.seed)
        self.model.to(self.device)
        best_val_metrics: dict[str, float] = {}
        last_val_metrics: dict[str, float] = {}

        for epoch in range(1, self.epochs + 1):
            train_m = self._train_epoch()
            val_m = self._validate()
            last_val_metrics = val_m
            self.scheduler.step()
            self._log_epoch(epoch, train_m, val_m)

            ckpt_metrics = {
                **val_m,
                "val_loss": val_m["loss"],
                "f1": val_m["f1_macro"],
                "f2": val_m["f2_macro"],
            }
            self.ckpt_manager.save(
                self.model, self.optimizer, self.scheduler,
                epoch, ckpt_metrics, self.config_dict, self.seed,
            )

            if self._update_best_model(val_m):
                best_val_metrics = val_m
                self._best_epoch = epoch

            if verbose:
                _, lr_head = self._get_lrs()
                das = val_m["danger_as_safe_rate"]
                das_flag = "✓" if das < DANGER_AS_SAFE_LIMIT else "✗"
                class_w_str = ""
                if self.learnable_weight is not None:
                    cw = self.learnable_weight.get_weights().detach().cpu().tolist()
                    class_w_str = f" | class_w: [{cw[0]:.2f}, {cw[1]:.2f}, {cw[2]:.2f}]"
                print(
                    f"[Epoch {epoch}]"
                    f" loss: {val_m['loss']:.4f}"
                    f" | acc: {val_m['accuracy']:.4f}"
                    f" | danger_prec: {val_m['precision_danger'] * 100:.2f}%"
                    f" | danger_as_safe: {das * 100:.2f}% {das_flag}"
                    f" | lr_head: {lr_head:.2e}"
                    f"{class_w_str}"
                )

            if self.early_stopping.update(val_m["precision_danger"], val_m["danger_as_safe_rate"]):
                if verbose:
                    print(f"[train] Early stop at epoch {epoch} (patience={self.early_stopping.patience})")
                break

        self._save_plots()
        return best_val_metrics or last_val_metrics
