"""results/*/summary.csv를 모아 실험 비교표(콘솔 + CSV + 막대그래프)를 생성한다.

폴더명을 실험 식별자로 사용한다(내부 exp_name은 tta 변형에서 충돌 가능). 추론을 재실행하지
않고 기존 summary.csv만 읽는다.

Usage:
    python scripts/compare_experiments.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _PROJECT_ROOT / "results"
# summary.csv 컬럼 → 출력 컬럼 (miss_rate_danger를 miss_rate로 노출)
_COLS: list[str] = [
    "exp_name", "precision_danger", "recall_danger", "miss_rate", "accuracy",
    "f1_macro", "coverage",
]
_BAR_METRICS: list[tuple[str, str]] = [
    ("precision_danger", "#2ecc71"),
    ("miss_rate", "#e74c3c"),
    ("f1_macro", "#3498db"),
]
_DPI = 150


def _load_target_rows(target: str) -> pd.DataFrame:
    """모든 summary.csv에서 해당 eval_target 행을 모아 폴더명을 exp_name으로 부여."""
    records: list[dict[str, object]] = []
    for summary in sorted(_RESULTS_DIR.glob("*/summary.csv")):
        df = pd.read_csv(summary)
        row = df[df["eval_target"] == target]
        if row.empty:
            continue
        r = row.iloc[0]
        records.append({
            "exp_name": summary.parent.name,
            "precision_danger": float(r["precision_danger"]),
            "recall_danger": float(r["recall_danger"]),
            "miss_rate": float(r["miss_rate_danger"]),
            "accuracy": float(r["accuracy"]),
            "f1_macro": float(r["f1_macro"]),
            "coverage": float(r["coverage"]),
        })
    out = pd.DataFrame(records, columns=_COLS)
    return out.sort_values("precision_danger", ascending=False).reset_index(drop=True)


def _print_console(df: pd.DataFrame, target: str) -> None:
    print(f"\n{'=' * 96}")
    print(f"  실험 비교 — eval_target={target}  (precision_danger 내림차순, n={len(df)})")
    print("=" * 96)
    header = (f"{'exp_name':<24} {'prec_dgr':>9} {'recall_dgr':>10} {'miss_rate':>9} "
              f"{'accuracy':>9} {'f1_macro':>9} {'coverage':>9}")
    print(header)
    print("-" * 96)
    for _, r in df.iterrows():
        print(f"{r['exp_name']:<24} {r['precision_danger'] * 100:>8.2f}% "
              f"{r['recall_danger'] * 100:>9.2f}% {r['miss_rate'] * 100:>8.2f}% "
              f"{r['accuracy'] * 100:>8.2f}% {r['f1_macro']:>9.4f} {r['coverage'] * 100:>8.2f}%")
    print("-" * 96)


def _save_csv(frames: dict[str, pd.DataFrame], out_path: Path) -> None:
    parts: list[pd.DataFrame] = []
    for target, df in frames.items():
        tagged = df.copy()
        tagged.insert(0, "eval_target", target)
        parts.append(tagged)
    combined = pd.concat(parts, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\n[compare] CSV → {out_path}  ({len(combined)} rows, targets={list(frames)})")


def _save_bar(df: pd.DataFrame, target: str, out_path: Path) -> None:
    """precision_danger / miss_rate / f1_macro 3개 지표 묶음 막대그래프."""
    exps = df["exp_name"].tolist()
    x = np.arange(len(exps))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(exps) * 1.1), 6), dpi=_DPI)
    for i, (metric, color) in enumerate(_BAR_METRICS):
        ax.bar(x + (i - 1) * width, df[metric].to_numpy(), width, label=metric, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(exps, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("value (0–1)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Experiment comparison (eval_target={target})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] PNG → {out_path}  (x={len(exps)} exps, metrics={[m for m, _ in _BAR_METRICS]})")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="results/*/summary.csv 실험 비교표 생성")
    p.add_argument("--console-target", default="crops_25pct", help="콘솔 표 기준 eval_target")
    p.add_argument("--csv-out", type=Path, default=_RESULTS_DIR / "experiments_comparison.csv")
    p.add_argument("--png-out", type=Path, default=_RESULTS_DIR / "experiments_comparison.png")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    frames = {t: _load_target_rows(t) for t in ("all", "crops_25pct")}

    # 출력 1: 콘솔 표 (crops_25pct)
    _print_console(frames[args.console_target], args.console_target)

    # 출력 2: CSV (all + crops_25pct)
    _save_csv(frames, args.csv_out)

    # 출력 3: 막대그래프 (콘솔과 동일 타겟 기준)
    _save_bar(frames[args.console_target], args.console_target, args.png_out)
    print("[compare] done.")


if __name__ == "__main__":
    main()
