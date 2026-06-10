"""Grad-CAM 시각화: TP danger / FN danger→cut / FP cut→danger 각 8장.

backbone patch_tokens에 대한 클래스 스코어 기울기로 공간 중요도 히트맵 생성.

Usage:
    python scripts/visualize_gradcam.py \
        --config configs/exp_v_bicubic.yaml \
        --checkpoint experiments/exp_v_bicubic/checkpoints/best_val_loss_ep010_0.312.ckpt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._eval_common import build_model, collect_probs, detect_device, load_config
from src.data.dataset import CLASS_NAMES, HazardDataset
from src.data.transforms import get_val_transforms

_DEFAULT_CONFIG = Path("configs/exp_v_bicubic.yaml")
_DEFAULT_EXP = "exp_v_bicubic"
_SAMPLES_PER_CASE = 8
_COLS = 4
_DPI = 150
_ALPHA = 0.55

CUT, DANGER, EXCLUDED = 0, 1, 2


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grad-CAM visualization")
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(f"experiments/{_DEFAULT_EXP}/visualizations/gradcam"),
    )
    return p.parse_args()


def _resolve_checkpoint(cfg: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            sys.exit(f"[ERROR] checkpoint not found: {explicit}")
        return explicit
    ckpt_dir = _PROJECT_ROOT / "experiments" / cfg["experiment"]["name"] / "checkpoints"
    candidates = sorted(ckpt_dir.glob("best_val_loss_*.ckpt"))
    if not candidates:
        sys.exit(f"[ERROR] best_val_loss_*.ckpt not found in {ckpt_dir}")
    return candidates[-1]


def _dataset_kwargs(d: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "unk_label": d.get("unk_label", "excluded"),
        "label_col": d.get("label_col", "confirmed_label"),
        "split_col": d.get("split_col", "split"),
    }
    if "csv_path" in d:
        kw["csv_path"] = Path(d["csv_path"])
    return kw


def _collect_cases(
    dataset: HazardDataset,
    probs: np.ndarray,
    true_cls: int,
    pred_cls: int,
    n: int,
) -> list[tuple[Path, int, int, np.ndarray]]:
    results: list[tuple[Path, int, int, np.ndarray]] = []
    labels = dataset.labels
    paths = [p for p, _ in dataset._samples]
    for i, true in enumerate(labels):
        pred = int(probs[i].argmax())
        if true == true_cls and pred == pred_cls:
            results.append((paths[i], true, pred, probs[i]))
    results.sort(key=lambda r: -r[3][DANGER])
    return results[:n]


def _gradcam(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    target_cls: int,
    device: torch.device,
    patch_grid: int,
) -> np.ndarray:
    """patch_tokens 기준 Grad-CAM → [patch_grid, patch_grid] 히트맵."""
    model.eval()
    x = img_tensor.unsqueeze(0).to(device).requires_grad_(False)

    # backbone에서 patch_tokens를 중간 변수로 캡처해 grad 계산
    patch_tokens_captured: list[torch.Tensor] = []

    def _fwd_hook(_module: Any, _input: Any, output: Any) -> None:
        # backbone forward returns (cls_token, patch_tokens)
        patch_tokens_captured.clear()
        t = output[1].detach().requires_grad_(True)
        patch_tokens_captured.append(t)

    hook = model.backbone.register_forward_hook(_fwd_hook)

    # backbone forward (원본 patch_tokens 캡처)
    with torch.no_grad():
        cls_token, _ = model.backbone(x)
    hook.remove()

    if not patch_tokens_captured:
        return np.zeros((patch_grid, patch_grid), dtype=np.float32)

    patch_tokens = patch_tokens_captured[0]  # [1, N, D]

    # head forward: patch_tokens에 grad 활성화해서 재실행
    patch_tokens.requires_grad_(True)
    logits = model.head(cls_token, patch_tokens)          # [1, 3]
    score = logits[0, target_cls]
    score.backward()

    grad = patch_tokens.grad  # [1, N, D]
    if grad is None:
        return np.zeros((patch_grid, patch_grid), dtype=np.float32)

    # Grad-CAM: ReLU(mean_D(grad * activation))
    cam = (grad * patch_tokens.detach()).mean(dim=-1)     # [1, N]
    cam = F.relu(cam)[0].cpu().numpy()                    # [N]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.reshape(patch_grid, patch_grid)


def _overlay(img: Image.Image, heat: np.ndarray, size: int, alpha: float = _ALPHA) -> np.ndarray:
    img_resized = img.resize((size, size))
    heat_pil = Image.fromarray((heat * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    heat_rgb = np.array(cm.jet(np.array(heat_pil) / 255.0))[:, :, :3]
    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    return np.clip((1 - alpha) * img_arr + alpha * heat_rgb, 0, 1)


def _save_grid(
    cases: list[tuple[Path, int, int, np.ndarray]],
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    patch_grid: int,
    transform: Any,
    target_cls: int,
    title: str,
    out_path: Path,
) -> None:
    n = len(cases)
    if n == 0:
        print(f"  skip (no samples): {out_path.name}")
        return
    rows = (n + _COLS - 1) // _COLS
    fig, axes = plt.subplots(rows, _COLS, figsize=(_COLS * 2.8, rows * 3.0), dpi=_DPI)
    if rows == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()
    fig.suptitle(title, fontsize=10, fontweight="bold")

    for idx, ax in enumerate(axes_flat):
        if idx >= n:
            ax.axis("off")
            continue
        img_path, true_idx, pred_idx, prob_vec = cases[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except OSError:
            ax.axis("off")
            continue
        img_tensor = transform(img)
        heat = _gradcam(model, img_tensor, target_cls, device, patch_grid)
        overlay = _overlay(img, heat, image_size)
        ax.imshow(overlay)
        ax.axis("off")
        cut_p, dan_p, exc_p = prob_vec
        ax.set_title(
            f"T={CLASS_NAMES[true_idx]} P={CLASS_NAMES[pred_idx]}\n"
            f"cut={cut_p:.2f} dan={dan_p:.2f} exc={exc_p:.2f}",
            fontsize=6, pad=2,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}  ({n} samples)")


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    ckpt = _resolve_checkpoint(cfg, args.checkpoint)
    device = detect_device(args.device)

    print(f"[gradcam] config     : {args.config}")
    print(f"[gradcam] checkpoint : {ckpt.name}")
    print(f"[gradcam] device     : {device}")

    d = cfg["data"]
    image_size: int = d.get("image_size", 336)
    backbone_name: str = cfg["model"].get("backbone_name", "dinov2_vitb14")
    patch_size = 14 if "14" in backbone_name else 16
    patch_grid = image_size // patch_size

    model = build_model(cfg, ckpt, device)
    transform = get_val_transforms(image_size)

    ds = HazardDataset("test", transform=transform, **_dataset_kwargs(d))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=d.get("num_workers", 4), pin_memory=device.type == "cuda")
    print(f"[gradcam] test n={len(ds)}")

    from scripts._eval_common import collect_probs  # noqa: PLC0415
    probs, _ = collect_probs(model, loader, device)

    # Grad-CAM target: 예측 클래스(pred_cls) 기준
    cases_spec: list[tuple[str, str, int, int, int]] = [
        ("tp_danger.png",    "TP danger (T=danger, P=danger)",  DANGER, DANGER, DANGER),
        ("fn_danger2cut.png","FN danger→cut (T=danger, P=cut)", DANGER, CUT,    DANGER),
        ("fp_cut2danger.png","FP cut→danger (T=cut, P=danger)", CUT,    DANGER, DANGER),
    ]

    for fname, title, true_cls, pred_cls, target_cls in cases_spec:
        cases = _collect_cases(ds, probs, true_cls, pred_cls, _SAMPLES_PER_CASE)
        _save_grid(cases, model, device, image_size, patch_grid, transform,
                   target_cls, title, args.output_dir / fname)

    print("[gradcam] done.")


if __name__ == "__main__":
    main()
