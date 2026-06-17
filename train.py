"""Training entry point for the hazard detection system.

Usage:
    python train.py --config configs/exp_a_mlp.yaml
    python train.py --config configs/exp_b_att.yaml --device cuda
    python train.py --config configs/exp_a_mlp.yaml --epochs 50
    python train.py --config configs/exp_a_mlp.yaml \\
                    --resume experiments/exp_a_mlp/checkpoints/best_val_loss_ep010_0.312.ckpt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader

from scripts._eval_common import build_model, detect_device, load_config
from scripts.evaluate_test import evaluate_checkpoint
from src.data.dataset import HazardDataset, compute_class_weights
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.hazard_model import HazardModel
from src.training.checkpoint import CheckpointManager
from src.training.trainer import Trainer, set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hazard detection model training")
    p.add_argument("--config", type=Path, required=True, help="YAML config file path")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Experiment output root (default: experiments/<exp_name>)")
    p.add_argument("--device", default="auto", help="cuda / cpu / auto (default: auto)")
    p.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume from")
    p.add_argument("--epochs", type=int, default=None, help="Override training.epochs in config")
    p.add_argument("--testset-root", type=Path, default=None,
                   help="외부 testset 루트 (미지정 시 config의 data.testset_root 사용). "
                        "Ctrl+C 중단 시 best checkpoint 평가에 사용")
    return p.parse_args()


def _build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    d = cfg["data"]
    image_size: int = d.get("image_size", 336)
    batch_size: int = d.get("batch_size", 32)
    num_workers: int = d.get("num_workers", 4)
    unk_label: str = d.get("unk_label", "excluded")
    label_col: str = d.get("label_col", "confirmed_label")
    eval_label_col: str = d.get("eval_label_col", label_col)
    split_col: str = d.get("split_col", "split")
    ds_kwargs: dict[str, Any] = {"unk_label": unk_label, "label_col": label_col, "split_col": split_col}
    eval_ds_kwargs: dict[str, Any] = {**ds_kwargs, "label_col": eval_label_col}
    if "csv_path" in d:
        ds_kwargs["csv_path"] = Path(d["csv_path"])
        eval_ds_kwargs["csv_path"] = Path(d["csv_path"])

    aug_cfg: dict[str, Any] | None = cfg.get("data_augmentation") or None
    padding_color: str = d.get("padding_color", "black")
    padding_variants: list[str] | None = d.get("padding_variants")

    if padding_variants:
        img_base = Path("dataset/classification")
        per_variant: list[HazardDataset] = [
            HazardDataset(
                "train",
                transform=get_train_transforms(image_size, aug_cfg, padding_color),
                img_root=img_base / variant,
                **ds_kwargs,
            )
            for variant in padding_variants
        ]
        train_ds: HazardDataset | ConcatDataset = ConcatDataset(per_variant)
        # 모든 variant는 동일 파일셋이므로 첫 번째로 class_weights 계산
        weights_ds = per_variant[0]
    else:
        train_ds = HazardDataset(
            "train", transform=get_train_transforms(image_size, aug_cfg, padding_color), **ds_kwargs
        )
        weights_ds = train_ds

    # val/test는 crops_25pct 단일 기준 유지 (평가 일관성)
    val_ds = HazardDataset(
        "val", transform=get_val_transforms(image_size, padding_color), **eval_ds_kwargs
    )

    if len(train_ds) == 0:
        sys.exit("[ERROR] train split이 비어 있습니다. CSV 경로와 split 컬럼을 확인하세요.")

    class_weights = compute_class_weights(weights_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=len(train_ds) > batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, class_weights


def _maybe_resume(args: argparse.Namespace, model: HazardModel, trainer: Trainer) -> None:
    if args.resume is None:
        return
    if not args.resume.exists():
        sys.exit(f"[ERROR] resume checkpoint not found: {args.resume}")

    ckpt = CheckpointManager.load(args.resume)
    model.load_state_dict(ckpt["model_state_dict"])
    trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ckpt.get("scheduler_state_dict"):
        trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"[train] Resumed from epoch {ckpt['epoch']} ({args.resume.name})")


def _resolve_testset_root(args: argparse.Namespace, cfg: dict[str, Any]) -> Path | None:
    if args.testset_root is not None:
        return args.testset_root
    raw = cfg.get("data", {}).get("testset_root")
    return Path(raw) if raw else None


def _find_best_checkpoint(ckpt_dir: Path) -> Path | None:
    """best_model.pth 우선, 없으면 가장 최근에 저장된 best_*.ckpt를 반환."""
    primary = ckpt_dir / "best_model.pth"
    if primary.exists():
        return primary
    candidates = sorted(ckpt_dir.glob("best_*.ckpt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _evaluate_on_interrupt(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    trainer: Trainer,
    ckpt_dir: Path,
    exp_name: str,
) -> None:
    """Ctrl+C 중단 시 best checkpoint로 4타겟 평가를 직접 실행."""
    print("\n[train] 학습 중단됨. best checkpoint로 평가를 실행합니다 …")

    testset_root = _resolve_testset_root(args, cfg)
    if testset_root is None:
        print("[train] ⚠ testset_root가 config(data.testset_root)나 --testset-root 인자에 없습니다. "
              "평가를 건너뜁니다.")
        return
    if not testset_root.exists():
        print(f"[train] ⚠ testset_root 경로가 존재하지 않습니다: {testset_root}. 평가를 건너뜁니다.")
        return

    best_ckpt = _find_best_checkpoint(ckpt_dir)
    if best_ckpt is None:
        print(f"[train] ⚠ best checkpoint를 찾지 못했습니다: {ckpt_dir}. 평가를 건너뜁니다.")
        return
    print(f"[train] best checkpoint: {best_ckpt.name}")

    # EMA 활성 시 저장된 best checkpoint는 이미 ema_model.module 가중치로 기록된다(trainer.py).
    if trainer.ema_model is not None:
        print("[train] EMA 활성 — best checkpoint에 EMA 가중치가 반영되어 있습니다.")

    # 두 번째 Ctrl+C는 여기서 캐치하지 않는다 → KeyboardInterrupt 전파로 즉시 강제 종료.
    summary = evaluate_checkpoint(
        cfg, best_ckpt, testset_root,
        output_dir=Path("results") / exp_name,
        device=args.device,
        batch_size=cfg.get("data", {}).get("batch_size", 32),
    )
    print(f"[train] 중단 평가 완료 → {summary}")


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    exp_name: str = cfg.get("experiment", {}).get("name", args.config.stem)
    output_dir = args.output_dir or Path("experiments") / exp_name
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    t_cfg = cfg["training"]
    seed: int = t_cfg.get("seed", 42)
    set_seed(seed)

    device = detect_device(args.device)
    device_label = (
        f"cuda ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "cpu"
    )
    print(f"[train] Exp       : {exp_name}")
    print(f"[train] Config    : {args.config}")
    print(f"[train] Device    : {device_label}")
    print(f"[train] Output    : {output_dir}")

    m_cfg = cfg["model"]
    print(f"[train] Building model ({m_cfg.get('backbone_name', 'dinov2_vitb14')}, "
          f"head={m_cfg.get('head_type', 'mlp')}) …")
    model = build_model(cfg, None, device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[train] Params    : trainable={n_trainable:,} / frozen={n_frozen:,}")

    print("[train] Loading data …")
    train_loader, val_loader, class_weights = _build_dataloaders(cfg)
    print(f"[train] Data      : train={len(train_loader.dataset)} / val={len(val_loader.dataset)} samples")
    print(f"[train] class_weights: {class_weights.tolist()}")

    ckpt_manager = CheckpointManager(ckpt_dir)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        ckpt_manager=ckpt_manager,
        log_dir=log_dir,
        config_dict=cfg,
        class_weights=class_weights,
        device=device,
        backbone_lr=t_cfg.get("backbone_lr", 0.0),
        lr=t_cfg.get("head_lr", t_cfg.get("lr", 1e-5)),
        weight_decay=t_cfg.get("weight_decay", 1e-2),
        epochs=t_cfg.get("epochs", 30),
        warmup_ratio=t_cfg.get("warmup_ratio", 0.1),
        warmup_type=t_cfg.get("warmup_type", "linear"),
        early_stopping_patience=t_cfg.get("early_stopping_patience", 10),
        label_smoothing=t_cfg.get("label_smoothing", 0.1),
        focal_gamma=t_cfg.get("focal_gamma", 2.0),
        seed=seed,
        use_learnable_weight=t_cfg.get("use_learnable_weight", False),
        loss_type=t_cfg.get("loss_type", "focal"),
        lambda_dr=t_cfg.get("lambda_dr", 0.0),
        num_classes=cfg["model"].get("num_classes", 3),
        best_metric=t_cfg.get("best_metric", "danger_precision"),
        das_constraint=t_cfg.get("das_constraint", 0.15),
        llrd_decay=t_cfg.get("llrd_decay", 1.0),
        wd_exclude=t_cfg.get("wd_exclude", False),
        use_ema=t_cfg.get("use_ema", False),
        ema_decay=t_cfg.get("ema_decay", 0.9995),
    )

    _maybe_resume(args, model, trainer)

    print(
        f"[train] Training  : epochs={t_cfg.get('epochs', 30)}"
        f" | lr={t_cfg.get('head_lr', t_cfg.get('lr', 1e-5)):.1e}"
        f" | patience={t_cfg.get('early_stopping_patience', 10)}"
    )
    print("[train] " + "-" * 72)

    try:
        best = trainer.fit(verbose=True)
    except KeyboardInterrupt:
        _evaluate_on_interrupt(args, cfg, trainer, ckpt_dir, exp_name)
        return

    print("[train] " + "-" * 72)
    if best:
        print(
            f"[train] Best val  | loss={best.get('loss', float('nan')):.4f}"
            f" | f1={best.get('f1_macro', 0):.4f}"
            f" | safe_prec={best.get('safe_precision', 0):.4f}"
            f" | danger_as_safe={best.get('danger_as_safe_rate', float('nan')):.4f}"
        )
    print(f"[train] Checkpoints: {ckpt_dir}")
    print(f"[train] Log        : {log_dir / 'train_log.csv'}")
    print("[train] Done.")


if __name__ == "__main__":
    main()
