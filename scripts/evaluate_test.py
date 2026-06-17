"""외부 testset 평가 — 4타겟(all / crops_0pct / crops_25pct / crops_50pct) × 전체 산출물.

단일 모델을 지정하면 4타겟을 항상 자동 평가한다. threshold/margin 선택은 항상
val(또는 내부 test)에서 수행한다 — 외부 testset으로 thr을 선택하면 데이터 누수이므로
predictions.csv의 applied_thr_source에 선택 출처를 기록한다.

타겟별 results/{exp_name}/{eval_target}/metrics.csv가 이미 있으면 스킵하며(--force로 무시),
스킵된 타겟도 기존 metrics.csv에서 읽어 summary.csv에 항상 반영한다.

Usage:
    # 단일 체크포인트 → 4타겟 자동
    python scripts/evaluate_test.py \\
        --config configs/exp_ap_f1best.yaml \\
        --checkpoint checkpoints/exp_ap/best.ckpt

    # 디렉토리 내 모든 .ckpt 순회 (체크포인트별 하위 폴더로 분리 저장)
    python scripts/evaluate_test.py \\
        --config configs/exp_ap_f1best.yaml \\
        --checkpoint_dir checkpoints/

    # 스킵 무시하고 전체 재실행
    python scripts/evaluate_test.py ... --force
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
    DISPLAY_NAME_MAP,
    assert_patch_compatible,
    build_model,
    collect_outputs,
    dataset_kwargs,
    detect_device,
    load_config,
)
from src.data.dataset import CLASS_NAME_LIST, HazardDataset, TestsetFolderDataset
from src.data.transforms import get_val_transforms
from src.evaluation import report_artifacts as ra
from src.evaluation.evaluator import compute_metrics
from src.evaluation.threshold import apply_danger_threshold, select_best_threshold

_CLASS_NAMES: list[str] = CLASS_NAME_LIST
_VARIANTS: list[str] = ["crops_0pct", "crops_25pct", "crops_50pct"]
_ALL_VARIANTS: list[str] = ["all", *_VARIANTS]  # 항상 평가하는 4타겟
_GAP_WARN: float = 0.05  # val vs test 절대 차이 > 5%p 경고
_DEFAULT_DAS_LIMIT: float = 0.15
_SWEEP_THRS: np.ndarray = np.round(np.arange(0.10, 0.901, 0.05), 2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="External testset evaluation (4 targets + summary)")
    p.add_argument("--config", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path, default=None, help="단일 체크포인트")
    g.add_argument("--checkpoint_dir", type=Path, default=None,
                   help="디렉토리 내 모든 .ckpt 순회 (체크포인트별 하위 폴더로 분리)")
    p.add_argument("--testset_root", type=Path, default=Path("dataset/testset"))
    p.add_argument("--output_dir", type=Path, default=None,
                   help="기본값: results/{experiment.name}")
    p.add_argument("--threshold", type=float, default=None,
                   help="미지정 시 val에서 자동 선택 (no-leakage)")
    p.add_argument("--thr-source", default="manual",
                   help="--threshold 직접 지정 시 기록할 선택 출처 (val/internal_test/manual)")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--tta", action="store_true",
                   help="원본 + hflip softmax 평균 (default: False)")
    p.add_argument("--force", action="store_true",
                   help="기존 metrics.csv 무시하고 전체 재실행")
    return p.parse_args()


@dataclass(slots=True)
class _Inference:
    """타겟 1회 추론 결과 — 'all' 타겟을 캐시 concat으로 재구성하기 위한 컨테이너."""
    logits: np.ndarray
    probs: np.ndarray
    labels: np.ndarray
    paths: list[Path]
    crop_pcts: list[int]
    n: int


def _roots_for_target(testset_root: Path, target: str) -> list[Path]:
    if target == "all":
        return [testset_root / v for v in _VARIANTS]
    return [testset_root / target]


def _infer_variant(
    variant: str, cfg: dict[str, Any], args: argparse.Namespace, model: Any, device: Any,
    image_size: int, padding_color: str,
) -> _Inference:
    """단일 crops_<N>pct variant 추론 (paths/crop_pcts 포함)."""
    d = cfg["data"]
    ds = TestsetFolderDataset([args.testset_root / variant],
                              transform=get_val_transforms(image_size, padding_color))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=d.get("num_workers", 4), pin_memory=device.type == "cuda")
    logits, probs, labels = collect_outputs(model, loader, device, tta=args.tta)
    return _Inference(logits, probs, labels, ds.paths, ds.crop_pcts, len(ds))


def _inference_for_target(
    target: str, cache: dict[str, _Inference], cfg: dict[str, Any], args: argparse.Namespace,
    model: Any, device: Any, image_size: int, padding_color: str,
) -> _Inference:
    """타겟 추론 결과 반환. variant는 캐시하고 'all'은 variant 캐시를 concat해 재구성한다."""
    def variant(v: str) -> _Inference:
        if v not in cache:
            cache[v] = _infer_variant(v, cfg, args, model, device, image_size, padding_color)
        return cache[v]

    if target != "all":
        return variant(target)

    parts = [variant(v) for v in _VARIANTS]
    return _Inference(
        logits=np.concatenate([p.logits for p in parts], axis=0),
        probs=np.concatenate([p.probs for p in parts], axis=0),
        labels=np.concatenate([p.labels for p in parts], axis=0),
        paths=[path for p in parts for path in p.paths],
        crop_pcts=[c for p in parts for c in p.crop_pcts],
        n=sum(p.n for p in parts),
    )


def _print_metrics(metrics: dict[str, float], coverage: float, target: str) -> None:
    print(f"\n[{target}] Accuracy={metrics['accuracy']:.4f}  "
          f"Macro F1={metrics['f1_macro']:.4f}  Macro F2={metrics['f2_macro']:.4f}")
    print(f"  {'class':>10} | {'precision':>9} | {'recall':>6} | {'f1':>6}")
    print("  " + "-" * 42)
    for cls in _CLASS_NAMES:
        print(f"  {cls:>10} | {metrics[f'precision_{cls}'] * 100:>8.2f}% |"
              f" {metrics[f'recall_{cls}'] * 100:>5.2f}% | {metrics[f'f1_{cls}'] * 100:>5.2f}%")
    print(f"  Miss Rate (Danger) = {metrics['danger_as_safe_rate'] * 100:.2f}%   "
          f"Coverage = {coverage * 100:.2f}%")


def _print_gap(val_m: dict[str, float], test_m: dict[str, float], target: str) -> None:
    rows = [
        ("accuracy", "Accuracy"),
        ("f1_macro", "Macro F1"),
        ("danger_as_safe_rate", "Miss Rate (Danger)"),
        ("precision_danger", "Precision (Danger)"),
        ("recall_danger", "Recall (Danger)"),
    ]
    print(f"\n[{target}] val vs test gap (>5%p 경고):")
    for key, label in rows:
        v, t = val_m[key], test_m[key]
        flag = "  <-- gap > 5%p" if abs(v - t) > _GAP_WARN else ""
        print(f"  {label:<20} val={v * 100:>6.2f}%  test={t * 100:>6.2f}%{flag}")


def _evaluate_target(
    target: str,
    inf: _Inference,
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    val_metrics: dict[str, float],
    applied_thr: float,
    thr_source: str,
    exp_name: str,
    image_size: int,
    out_root: Path,
    ckpt_path: Path,
) -> dict[str, Any]:
    logits, probs, labels = inf.logits, inf.probs, inf.labels
    print(f"\n{'=' * 68}\n  TARGET: {target}  (n={inf.n})\n{'=' * 68}")

    preds = apply_danger_threshold(probs, applied_thr)
    metrics = compute_metrics(labels.tolist(), preds.tolist())
    coverage = ra.coverage_of(preds)
    support = ra.support_counts(labels)

    _print_metrics(metrics, coverage, target)
    _print_gap(val_metrics, metrics, target)

    target_dir = out_root / target
    stem = ckpt_path.stem
    res = image_size

    def png(kind: str) -> Path:
        return target_dir / f"{target}_{kind}_{stem}_{res}.png"

    ra.write_predictions_csv(
        target_dir / "predictions.csv", inf.paths, labels, preds, probs, logits,
        inf.crop_pcts, applied_thr, thr_source,
    )
    ra.write_metrics_csv(target_dir / "metrics.csv", metrics, support, coverage, DISPLAY_NAME_MAP)
    ra.write_threshold_sweep_csv(target_dir / "threshold_sweep.csv", probs, labels, _SWEEP_THRS)
    ra.write_margin_sweep_csv(target_dir / "margin_sweep.csv", probs, labels)
    ra.write_confusion_matrix(target_dir / "confusion_matrix.csv", png("confusion_matrix"),
                              labels, preds)
    ra.write_pr_curve(png("pr_curve"), probs, labels)
    ra.write_roc_curve(png("roc_curve"), probs, labels)
    ra.write_calibration(
        target_dir / "calibration.json", png("reliability_diagram"),
        val_logits, val_labels, logits, probs, labels,
    )
    ra.write_merged_metrics_csv(target_dir / "merged_metrics.csv", labels, preds, DISPLAY_NAME_MAP)
    # 실무/납품 지표
    ra.write_risk_coverage_curve(
        target_dir / "risk_coverage_curve.csv", target_dir / "risk_coverage_curve.png",
        probs, labels, applied_thr)
    ra.write_operating_points(target_dir / "operating_points.csv", probs, labels)
    ra.write_cost_matrix(target_dir / "cost_matrix.csv", labels, preds)
    ra.write_bootstrap_ci(target_dir / "metrics.csv", labels, preds)  # metrics.csv에 CI 행 append
    ra.write_confidence_histogram(target_dir / "confidence_histogram.png", probs, labels)
    print(f"  artifacts → {target_dir}")

    return ra.build_summary_row(
        exp_name, target, res, ckpt_path.name, metrics, coverage, support, applied_thr,
    )


def _run_val(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    model: Any,
    device: Any,
    image_size: int,
    padding_color: str,
    das_limit: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float, str]:
    """val 추론 → threshold 선택 + calibration/gap 기준 (모두 no-leakage)."""
    d = cfg["data"]
    val_ds = HazardDataset("val", transform=get_val_transforms(image_size, padding_color),
                           **dataset_kwargs(d, use_eval_label_col=True))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=d.get("num_workers", 4), pin_memory=device.type == "cuda")
    print(f"[eval] val samples: {len(val_ds)} — 추론 중 …")
    val_logits, val_probs, val_labels = collect_outputs(model, val_loader, device, tta=args.tta)

    if args.threshold is not None:
        applied_thr, thr_source = float(args.threshold), args.thr_source
    else:
        applied_thr = select_best_threshold(val_probs, val_labels, _SWEEP_THRS, das_limit)
        thr_source = "val"
    print(f"[eval] applied_thr={applied_thr:.2f}  (source={thr_source}, das_limit={das_limit})")

    val_preds = apply_danger_threshold(val_probs, applied_thr)
    val_metrics = compute_metrics(val_labels.tolist(), val_preds.tolist())
    return val_logits, val_labels, val_metrics, applied_thr, thr_source


def _write_model_profile(
    model: Any, ckpt_path: Path, out_root: Path, image_size: int, device: Any,
) -> None:
    """배치별 latency/VRAM + 모델 정보 → results/{exp}/model_profile.csv. 실패해도 평가 무중단.

    profiling은 모델 dtype을 바꾸므로 반드시 4타겟 평가가 끝난 뒤 호출한다.
    """
    try:
        from src.export.check_vram import measure_model_info, run_batch_sweep  # noqa: PLC0415
    except ImportError as e:
        print(f"  [profile] skipped (import: {e})")
        return
    use_fp16 = device.type == "cuda"
    try:
        info = measure_model_info(model, ckpt_path)
        run_batch_sweep(model, image_size, device=device, fp16=use_fp16,
                        out_csv=out_root / "model_profile.csv", model_info=info)
        print(f"  model_profile → {out_root / 'model_profile.csv'}")
    except RuntimeError as e:  # CUDA OOM 등 — 프로파일은 보조 산출물이므로 평가를 막지 않는다
        print(f"  [profile] skipped (runtime: {e})")


def _process_checkpoint(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    ckpt_path: Path,
    out_root: Path,
    exp_name: str,
    image_size: int,
    padding_color: str,
    das_limit: float,
) -> None:
    """단일 체크포인트에 대해 4타겟 평가(+스킵) 후 summary.csv 항상 갱신."""
    needs_run = {t: args.force or not (out_root / t / "metrics.csv").exists() for t in _ALL_VARIANTS}

    print("=" * 68)
    print(f"  checkpoint={ckpt_path.name}  → out_root={out_root}")
    print("=" * 68)

    rows: list[dict[str, Any]] = []
    if any(needs_run.values()):
        device = detect_device(args.device)
        model = build_model(cfg, ckpt_path, device)
        val_logits, val_labels, val_metrics, applied_thr, thr_source = _run_val(
            cfg, args, model, device, image_size, padding_color, das_limit)
        cache: dict[str, _Inference] = {}  # variant 추론 1회만 — 'all'은 concat 재구성
        for target in _ALL_VARIANTS:
            if needs_run[target]:
                inf = _inference_for_target(
                    target, cache, cfg, args, model, device, image_size, padding_color)
                rows.append(_evaluate_target(
                    target, inf, val_logits, val_labels, val_metrics,
                    applied_thr, thr_source, exp_name, image_size, out_root, ckpt_path))
            else:
                print(f"⏭ SKIP: {exp_name}/{target} 이미 존재")
                rows.append(ra.read_summary_row_from_metrics(
                    out_root / target, exp_name, target, image_size, ckpt_path.name,
                    DISPLAY_NAME_MAP, applied_thr_fallback=applied_thr))
        _write_model_profile(model, ckpt_path, out_root, image_size, device)
    else:
        # 4타겟 모두 스킵 — 모델 로드 없이 기존 metrics.csv로 summary만 재생성
        for target in _ALL_VARIANTS:
            print(f"⏭ SKIP: {exp_name}/{target} 이미 존재")
            rows.append(ra.read_summary_row_from_metrics(
                out_root / target, exp_name, target, image_size, ckpt_path.name,
                DISPLAY_NAME_MAP, applied_thr_fallback=None))

    ra.write_summary_csv(out_root / "summary.csv", rows)
    print(f"\n[eval] summary.csv → {out_root / 'summary.csv'}  ({len(rows)} targets)")


def evaluate_checkpoint(
    cfg: dict[str, Any],
    ckpt_path: Path,
    testset_root: Path,
    *,
    output_dir: Path | None = None,
    device: str = "auto",
    batch_size: int = 32,
    tta: bool = False,
    threshold: float | None = None,
    thr_source: str = "manual",
    force: bool = False,
) -> Path:
    """단일 체크포인트를 4타겟 자동 평가 (train.py 등에서 직접 호출). summary.csv 경로 반환."""
    d = cfg["data"]
    image_size: int = d.get("image_size", 336)
    padding_color: str = d.get("padding_color", "black")
    backbone_name: str = cfg["model"].get("backbone_name", "dinov2_vitb14")
    assert_patch_compatible(image_size, backbone_name)
    exp_name: str = cfg["experiment"]["name"]
    das_limit: float = cfg.get("training", {}).get("das_constraint", _DEFAULT_DAS_LIMIT)
    out_root: Path = output_dir if output_dir is not None else Path("results") / exp_name

    args = argparse.Namespace(
        testset_root=testset_root, batch_size=batch_size, tta=tta,
        threshold=threshold, thr_source=thr_source, device=device, force=force,
    )
    _process_checkpoint(cfg, args, ckpt_path, out_root, exp_name,
                        image_size, padding_color, das_limit)
    return out_root / "summary.csv"


def _resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint_dir is not None:
        if not args.checkpoint_dir.is_dir():
            sys.exit(f"[ERROR] checkpoint_dir not found: {args.checkpoint_dir}")
        ckpts = sorted(args.checkpoint_dir.glob("*.ckpt"))
        if not ckpts:
            sys.exit(f"[ERROR] no .ckpt files in {args.checkpoint_dir}")
        return ckpts
    if not args.checkpoint.exists():
        sys.exit(f"[ERROR] checkpoint not found: {args.checkpoint}")
    return [args.checkpoint]


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    d = cfg["data"]
    image_size: int = d.get("image_size", 336)
    padding_color: str = d.get("padding_color", "black")
    backbone_name: str = cfg["model"].get("backbone_name", "dinov2_vitb14")
    assert_patch_compatible(image_size, backbone_name)

    exp_name: str = cfg["experiment"]["name"]
    das_limit: float = cfg.get("training", {}).get("das_constraint", _DEFAULT_DAS_LIMIT)
    base_out: Path = args.output_dir if args.output_dir is not None else Path("results") / exp_name

    checkpoints = _resolve_checkpoints(args)
    multi = args.checkpoint_dir is not None

    print("=" * 68)
    print(f"  evaluate_test.py  |  exp={exp_name}  res={image_size}")
    print(f"  targets={_ALL_VARIANTS}  tta={args.tta}  force={args.force}")
    print(f"  checkpoints={len(checkpoints)}  (mode={'dir' if multi else 'single'})")
    print("=" * 68)

    # 다중 체크포인트는 같은 exp_name 으로 충돌하므로 stem 하위 폴더로 분리한다.
    for ckpt_path in checkpoints:
        out_root = base_out / ckpt_path.stem if multi else base_out
        _process_checkpoint(cfg, args, ckpt_path, out_root, exp_name,
                            image_size, padding_color, das_limit)

    print("\n[eval] Done.")


if __name__ == "__main__":
    main()
