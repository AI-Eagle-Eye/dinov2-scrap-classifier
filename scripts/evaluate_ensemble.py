"""Multi-scale logit 앙상블 평가 — 두 모델(예: 336 + 448)의 logit을 평균해 4타겟 평가.

각 모델은 자기 config의 image_size/padding_color로 추론하고, 샘플 정렬이 동일한
두 모델의 logit을 평균(→ softmax)해 단일 결정 규칙(apply_danger_threshold)을 적용한다.
threshold/temperature는 앙상블 val logit에서만 선택한다(no-leakage). 산출물 writer와
threshold 선택 규칙은 evaluate_test.py와 완전히 동일하다.

Usage:
    python scripts/evaluate_ensemble.py \\
        --config configs/exp_ap_f1best.yaml \\
        --checkpoint experiments/exp_ap_f1best/checkpoints/best_model.pth \\
        --config configs/exp_ar_448.yaml \\
        --checkpoint experiments/exp_ar_448/checkpoints/best_f1_ep014_0.8198.ckpt \\
        --exp-name exp_ensemble_336_448
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._eval_common import (
    assert_patch_compatible,
    build_model,
    collect_outputs,
    dataset_kwargs,
    detect_device,
    load_config,
)
from scripts.evaluate_test import (
    _ALL_VARIANTS,
    _DEFAULT_DAS_LIMIT,
    _Inference,
    _SWEEP_THRS,
    _evaluate_target,
    _roots_for_target,
)
from src.data.dataset import HazardDataset, TestsetFolderDataset
from src.data.transforms import get_val_transforms
from src.evaluation import report_artifacts as ra
from src.evaluation.evaluator import compute_metrics
from src.evaluation.threshold import apply_danger_threshold, select_best_threshold


@dataclass(slots=True)
class _Member:
    """앙상블 멤버 1개 — 로드된 모델 + 전처리 해상도/패딩."""
    model: Any
    image_size: int
    padding_color: str
    tag: str


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _mean_logits(per_member: list[np.ndarray]) -> np.ndarray:
    """멤버별 logit [N, C]를 평균. shape 불일치는 정렬 깨짐을 의미하므로 즉시 실패."""
    ref = per_member[0].shape
    for lg in per_member[1:]:
        if lg.shape != ref:
            sys.exit(f"[ERROR] logit shape 불일치 {lg.shape} != {ref} — 멤버 간 샘플 정렬이 어긋남")
    return np.mean(per_member, axis=0)


def _ensemble_testset(
    target: str, members: list[_Member], device: Any, testset_root: Path, batch_size: int,
) -> _Inference:
    """타겟의 testset를 멤버별로 추론 → logit 평균 → _Inference(앙상블) 반환."""
    roots = _roots_for_target(testset_root, target)
    per_member_logits: list[np.ndarray] = []
    ref_labels: np.ndarray | None = None
    ref_paths: list[Path] = []
    ref_crop: list[int] = []
    for m in members:
        ds = TestsetFolderDataset(roots, transform=get_val_transforms(m.image_size, m.padding_color))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=device.type == "cuda")
        logits, _probs, labels = collect_outputs(m.model, loader, device, tta=False)
        per_member_logits.append(logits)
        if ref_labels is None:
            ref_labels, ref_paths, ref_crop = labels, ds.paths, ds.crop_pcts
        elif not np.array_equal(labels, ref_labels):
            sys.exit(f"[ERROR] target={target} 멤버 간 label 정렬 불일치 — 앙상블 평균 불가")

    ens_logits = _mean_logits(per_member_logits)
    ens_probs = _softmax(ens_logits)
    assert ref_labels is not None
    return _Inference(ens_logits, ens_probs, ref_labels, ref_paths, ref_crop, len(ref_labels))


def _ensemble_val(
    members: list[_Member], val_data_cfg: dict[str, Any], device: Any, batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """앙상블 val 추론 → (val_logits, val_probs, val_labels). threshold/temperature 선택용."""
    per_member_logits: list[np.ndarray] = []
    ref_labels: np.ndarray | None = None
    for m in members:
        val_ds = HazardDataset("val", transform=get_val_transforms(m.image_size, m.padding_color),
                               **dataset_kwargs(val_data_cfg, use_eval_label_col=True))
        loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=device.type == "cuda")
        logits, _probs, labels = collect_outputs(m.model, loader, device, tta=False)
        per_member_logits.append(logits)
        if ref_labels is None:
            ref_labels = labels
        elif not np.array_equal(labels, ref_labels):
            sys.exit("[ERROR] val 멤버 간 label 정렬 불일치 — 앙상블 평균 불가")
    ens_logits = _mean_logits(per_member_logits)
    assert ref_labels is not None
    return ens_logits, _softmax(ens_logits), ref_labels


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-scale logit ensemble evaluation (4 targets)")
    p.add_argument("--config", type=Path, action="append", required=True,
                   help="멤버 config (모델 수만큼 반복). --checkpoint와 순서 매칭")
    p.add_argument("--checkpoint", type=Path, action="append", required=True,
                   help="멤버 checkpoint (--config와 순서 매칭)")
    p.add_argument("--exp-name", default="exp_ensemble", help="결과 폴더/summary exp_name")
    p.add_argument("--testset_root", type=Path, default=Path("dataset/testset"))
    p.add_argument("--output_dir", type=Path, default=None,
                   help="기본값: results/{exp-name}")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if len(args.config) != len(args.checkpoint):
        sys.exit(f"[ERROR] --config({len(args.config)})와 --checkpoint({len(args.checkpoint)}) 개수 불일치")
    if len(args.config) < 2:
        sys.exit("[ERROR] 앙상블은 최소 2개 멤버가 필요합니다")

    device = detect_device(args.device)
    out_root: Path = args.output_dir or Path("results") / args.exp_name

    # ── 멤버 로드 ──────────────────────────────────────────────────────────────
    members: list[_Member] = []
    val_data_cfg: dict[str, Any] | None = None
    das_limit: float = _DEFAULT_DAS_LIMIT
    tags: list[str] = []
    for cfg_path, ckpt_path in zip(args.config, args.checkpoint):
        cfg = load_config(cfg_path)
        d = cfg["data"]
        image_size = d.get("image_size", 336)
        padding_color = d.get("padding_color", "black")
        backbone_name = cfg["model"].get("backbone_name", "dinov2_vitb14")
        assert_patch_compatible(image_size, backbone_name)
        if not ckpt_path.exists():
            sys.exit(f"[ERROR] checkpoint not found: {ckpt_path}")
        tag = f"{cfg['experiment']['name']}@{image_size}"
        print(f"[ens] load {tag}  ckpt={ckpt_path.name}")
        members.append(_Member(build_model(cfg, ckpt_path, device), image_size, padding_color, tag))
        tags.append(f"{cfg['experiment']['name']}_{image_size}")
        if val_data_cfg is None:
            val_data_cfg = d
            das_limit = cfg.get("training", {}).get("das_constraint", _DEFAULT_DAS_LIMIT)

    assert val_data_cfg is not None
    res_label = "+".join(str(m.image_size) for m in members)
    # 산출물 파일명(stem)과 summary checkpoint 컬럼에 사용할 합성 이름
    synthetic_ckpt = Path(f"ensemble_{'_'.join(tags)}")

    print("=" * 68)
    print(f"  evaluate_ensemble.py  |  exp={args.exp_name}  res={res_label}")
    print(f"  members={[m.tag for m in members]}  targets={_ALL_VARIANTS}")
    print("=" * 68)

    # ── val에서 threshold 선택 (no-leakage) ──────────────────────────────────────
    val_logits, val_probs, val_labels = _ensemble_val(
        members, val_data_cfg, device, args.batch_size)
    applied_thr = select_best_threshold(val_probs, val_labels, _SWEEP_THRS, das_limit)
    thr_source = "val"
    print(f"[ens] applied_thr={applied_thr:.2f}  (source={thr_source}, das_limit={das_limit})")
    val_preds = apply_danger_threshold(val_probs, applied_thr)
    val_metrics = compute_metrics(val_labels.tolist(), val_preds.tolist())

    # ── 4타겟 평가 (기존 _evaluate_target 재사용) ────────────────────────────────
    rows: list[dict[str, Any]] = []
    for target in _ALL_VARIANTS:
        inf = _ensemble_testset(target, members, device, args.testset_root, args.batch_size)
        rows.append(_evaluate_target(
            target, inf, val_logits, val_labels, val_metrics,
            applied_thr, thr_source, args.exp_name, res_label, out_root, synthetic_ckpt))

    ra.write_summary_csv(out_root / "summary.csv", rows)
    print(f"\n[ens] summary.csv → {out_root / 'summary.csv'}  ({len(rows)} targets)")
    print("[ens] Done.")


if __name__ == "__main__":
    main()
