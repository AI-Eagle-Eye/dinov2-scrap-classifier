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
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from src.data.dataset import HazardDataset, compute_class_weights
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.hazard_model import HazardModel, ModelConfig, VPTConfig
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
    return p.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        sys.exit(f"[ERROR] config not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model_config(cfg: dict[str, Any]) -> ModelConfig:
    m = cfg["model"]
    vpt_raw = m.get("vpt", {})
    return ModelConfig(
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


def _detect_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    d = cfg["data"]
    image_size: int = d.get("image_size", 336)
    batch_size: int = d.get("batch_size", 32)
    num_workers: int = d.get("num_workers", 4)
    unk_label: str = d.get("unk_label", "excluded")
    label_col: str = d.get("label_col", "confirmed_label")
    split_col: str = d.get("split_col", "split")
    ds_kwargs: dict[str, Any] = {"unk_label": unk_label, "label_col": label_col, "split_col": split_col}
    if "csv_path" in d:
        ds_kwargs["csv_path"] = Path(d["csv_path"])

    aug_cfg: dict[str, Any] | None = cfg.get("data_augmentation") or None
    padding_variants: list[str] | None = d.get("padding_variants")

    if padding_variants:
        img_base = Path("dataset/classification")
        per_variant: list[HazardDataset] = [
            HazardDataset(
                "train",
                transform=get_train_transforms(image_size, aug_cfg),
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
            "train", transform=get_train_transforms(image_size, aug_cfg), **ds_kwargs
        )
        weights_ds = train_ds

    # val/test는 crops_25pct 단일 기준 유지 (평가 일관성)
    val_ds = HazardDataset("val", transform=get_val_transforms(image_size), **ds_kwargs)

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


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)

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

    device = _detect_device(args.device)
    device_label = (
        f"cuda ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "cpu"
    )
    print(f"[train] Exp       : {exp_name}")
    print(f"[train] Config    : {args.config}")
    print(f"[train] Device    : {device_label}")
    print(f"[train] Output    : {output_dir}")

    model_cfg = _build_model_config(cfg)
    print(f"[train] Building model ({model_cfg.backbone_name}, head={model_cfg.head_type}) …")
    model = HazardModel(model_cfg)

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
    )

    _maybe_resume(args, model, trainer)

    print(
        f"[train] Training  : epochs={t_cfg.get('epochs', 30)}"
        f" | lr={t_cfg.get('lr', 1e-5):.1e}"
        f" | patience={t_cfg.get('early_stopping_patience', 10)}"
    )
    print("[train] " + "-" * 72)

    best = trainer.fit(verbose=True)

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
