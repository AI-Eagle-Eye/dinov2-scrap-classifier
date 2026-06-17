#!/usr/bin/env python3
"""dataset/ 라벨 CSV(v4 vs baseline) 호환성 진단 — 읽기 전용.

dataset/ 폴더는 절대 수정하지 않으며 콘솔 출력만 수행한다.
src/data/dataset.py 의 필터/샘플 빌드 로직이 요구하는 컬럼을 기준으로,
v4 CSV 를 그대로 학습 파이프라인에 넣었을 때 발생할 문제를 사전 점검한다.

실행:
    python scripts/diagnose_v4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import CLASS_NAME_LIST, _read_csv
_BASELINE_CSV = _PROJECT_ROOT / "dataset" / "label_with_split_full.csv"
_V4_CSV = _PROJECT_ROOT / "dataset" / "label_split_v4.csv"

# src/data/dataset.py 가 요구하는 필수 컬럼.
#   _filter()           → short_side, split_col, (label_col==confirmed_label 일 때만) unk
#   _build_sample_list()→ label_col, original_label, fname
_ALWAYS_REQUIRED: tuple[str, ...] = ("short_side", "original_label", "fname")
_CONFIRMED_ONLY_REQUIRED: tuple[str, ...] = ("unk",)
# KeyError 가 터지는 함수 위치 (점검 4 안내용)
_KEYERROR_LOC: dict[str, str] = {
    "short_side": "_filter()",
    "original_label": "_build_sample_list()",
    "fname": "_build_sample_list()",
    "unk": "_filter() (confirmed_label 사용 시)",
}

_VALID_LABELS: frozenset[str] = frozenset(CLASS_NAME_LIST)
_SHORT_SIDE_MIN: int = 32
_SPLITS: tuple[str, ...] = ("train", "val", "test")
_CLUSTER_PATTERNS: tuple[str, ...] = ("cluster", "group", "fold", "stratif")
# 자동 탐지 시 버전 신규 컬럼을 우선 선택한다.
_LABEL_PRIORITY: tuple[str, ...] = (
    "v4_label", "v2_label", "confirmed_label", "custom_label", "original_label",
)
_SPLIT_PRIORITY: tuple[str, ...] = ("split_v4", "split_v2", "split")


def _pick(candidates: list[str], priority: tuple[str, ...]) -> str | None:
    """후보 중 우선순위 목록에서 첫 일치를 고르고, 없으면 첫 후보."""
    if not candidates:
        return None
    for name in priority:
        if name in candidates:
            return name
    return candidates[0]


def _label_candidates(columns: list[str]) -> list[str]:
    return [c for c in columns if "label" in c.lower()]


def _split_candidates(columns: list[str]) -> list[str]:
    return [c for c in columns if "split" in c.lower()]


def _path_candidates(columns: list[str]) -> list[str]:
    keys = ("path", "fname", "image")
    return [c for c in columns if any(k in c.lower() for k in keys)]


def _cluster_candidates(columns: list[str]) -> list[str]:
    return [c for c in columns if any(p in c.lower() for p in _CLUSTER_PATTERNS)]


def _missing_required(columns: list[str], label_col: str | None) -> list[str]:
    cols = set(columns)
    missing = [c for c in _ALWAYS_REQUIRED if c not in cols]
    if label_col == "confirmed_label":
        missing += [c for c in _CONFIRMED_ONLY_REQUIRED if c not in cols]
    return missing


def _label_counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    """label_col 값을 strip 후 클래스별 건수로 집계 (NaN → 'nan')."""
    series = df[col].astype(str).str.strip()
    return {str(k): int(v) for k, v in series.value_counts().items()}


def _short_side_filtered(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """short_side >= 32 필터 적용; 컬럼 부재 시 원본과 False 반환."""
    if "short_side" not in df.columns:
        return df, False
    return df[df["short_side"] >= _SHORT_SIDE_MIN], True


def _hr(n: int) -> None:
    print(f"\n=== 점검 {n} ===")


def check1_columns(
    base_df: pd.DataFrame,
    v4_df: pd.DataFrame,
    base_label: str | None,
    base_split: str | None,
    v4_label: str | None,
    v4_split: str | None,
) -> None:
    _hr(1)
    print("[컬럼 구조 호환성]")
    base_cols, v4_cols = set(base_df.columns), set(v4_df.columns)
    only_v4 = sorted(v4_cols - base_cols)
    only_base = sorted(base_cols - v4_cols)
    print(f"  only_in_v4       : {only_v4 or '(없음)'}")
    print(f"  only_in_baseline : {only_base or '(없음)'}")

    print("\n[split / label / path 컬럼 후보 자동 탐지]")
    for name, df in (("baseline", base_df), ("v4", v4_df)):
        cols = list(df.columns)
        print(f"  {name:>8}: split={_split_candidates(cols)} "
              f"label={_label_candidates(cols)} path={_path_candidates(cols)}")

    print("\n[dataset.py 필수 컬럼 존재 여부]")
    for name, df, label_col in (("baseline", base_df, base_label), ("v4", v4_df, v4_label)):
        missing = _missing_required(list(df.columns), label_col)
        if missing:
            print(f"  {name:>8}: ✗ 누락 {missing}")
        else:
            print(f"  {name:>8}: ✓ 필수 컬럼 모두 존재")

    print("\n[탐지된 label_col / split_col]")
    print(f"  baseline: label_col={base_label}, split_col={base_split}")
    print(f"        v4: label_col={v4_label}, split_col={v4_split}")
    if v4_label != base_label or v4_split != base_split:
        print(f"⚠ config 수정 필요: label_col={v4_label}, split_col={v4_split}")


def check2_distribution(
    base_df: pd.DataFrame, v4_df: pd.DataFrame, base_label: str | None, v4_label: str | None,
) -> None:
    _hr(2)
    print("[short_side >= 32 필터 전/후 행 수]")
    base_filt, base_ok = _short_side_filtered(base_df)
    v4_filt, v4_ok = _short_side_filtered(v4_df)
    print(f"  baseline: 전 {len(base_df):>6} → 후 {len(base_filt):>6} "
          f"({'적용' if base_ok else 'short_side 없음 — 미적용'})")
    print(f"        v4: 전 {len(v4_df):>6} → 후 {len(v4_filt):>6} "
          f"({'적용' if v4_ok else 'short_side 없음 — 미적용'})")
    if not v4_ok:
        print("  ⚠ v4 에 short_side 컬럼이 없어 필터 후 분포 = 필터 전 분포")

    print("\n[클래스 분포 비교 — 필터 적용 후 기준]")
    if base_label is None or v4_label is None:
        print("  label_col 탐지 실패 — 분포 비교 생략")
        return
    base_counts = _label_counts(base_filt, base_label)
    v4_counts = _label_counts(v4_filt, v4_label)
    base_total = sum(base_counts.values()) or 1
    v4_total = sum(v4_counts.values()) or 1
    print(f"  {'label':<12}{'base n':>9}{'base %':>9}{'v4 n':>9}{'v4 %':>9}{'Δ%p':>9}")
    for lab in sorted(set(base_counts) | set(v4_counts)):
        bn, vn = base_counts.get(lab, 0), v4_counts.get(lab, 0)
        bp, vp = bn / base_total * 100, vn / v4_total * 100
        print(f"  {lab:<12}{bn:>9}{bp:>8.1f}%{vn:>9}{vp:>8.1f}%{vp - bp:>+8.1f}")

    unknown = {
        lab: v4_counts[lab] for lab in v4_counts if lab not in _VALID_LABELS
    }
    for lab, n in unknown.items():
        print(f"⚠ 미지 라벨 발견: {lab} ({n}건) → _build_sample_list에서 silent drop 위험")


def check3_leakage(name: str, df: pd.DataFrame, split_col: str | None) -> None:
    print(f"\n[{name}] split / cluster 누수")
    cluster_cands = _cluster_candidates(list(df.columns))
    cluster_col = cluster_cands[0] if cluster_cands else None

    if split_col is None or split_col not in df.columns:
        print(f"  ⚠ split 컬럼({split_col}) 미탐지 — split 분포 검증 불가")
    else:
        vc = df[split_col].value_counts(dropna=False)
        for sp in _SPLITS:
            print(f"  split={sp:<6}: {int(vc.get(sp, 0))}")
        na = int(df[split_col].isna().sum())
        extra = sorted(set(df[split_col].dropna().astype(str)) - set(_SPLITS))
        if na:
            print(f"  ⚠ split 누락값(NaN) {na}건")
        if extra:
            print(f"  ⚠ 예상 밖 split 값: {extra}")

    if cluster_col is None:
        print("  ⚠ cluster 컬럼 미탐지 — cluster 기반 누수 검증 불가")
        return
    if split_col is None or split_col not in df.columns:
        print(f"  cluster 컬럼={cluster_col} 탐지됨 — split 컬럼 부재로 누수 검증 불가")
        return
    per_cluster = df.groupby(cluster_col)[split_col].nunique()
    n_leaky = int((per_cluster > 1).sum())
    print(f"  cluster 컬럼={cluster_col}: 복수 split 에 걸친 cluster {n_leaky}개")
    if n_leaky > 0:
        print(f"⚠ 누수 의심: {n_leaky}개 cluster가 복수 split에 걸침")


def check4_summary(
    v4_df: pd.DataFrame,
    base_label: str | None,
    base_split: str | None,
    v4_label: str | None,
    v4_split: str | None,
) -> None:
    _hr(4)
    print("[수정 필요 항목]")
    print("  - csv_path : dataset/label_split_v4.csv (항상)")
    if v4_label != base_label:
        print(f"  - label_col: {base_label} → {v4_label}")
    if v4_split != base_split:
        print(f"  - split_col: {base_split} → {v4_split}")

    print("\n[필수 컬럼 부재 시 KeyError 발생 위치]")
    missing = _missing_required(list(v4_df.columns), v4_label)
    if not missing:
        print("  (없음) v4 에 필수 컬럼 모두 존재")
    for col in missing:
        print(f"  - {col} 없음 → {_KEYERROR_LOC.get(col, '?')} 에서 KeyError")

    print("\n※ 이미지 파일 실존 여부(_resolve_image_path)는 별도 확인 권장")


def main() -> None:
    for path in (_BASELINE_CSV, _V4_CSV):
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

    base_df = _read_csv(_BASELINE_CSV)
    v4_df = _read_csv(_V4_CSV)

    base_label = _pick(_label_candidates(list(base_df.columns)), _LABEL_PRIORITY)
    base_split = _pick(_split_candidates(list(base_df.columns)), _SPLIT_PRIORITY)
    v4_label = _pick(_label_candidates(list(v4_df.columns)), _LABEL_PRIORITY)
    v4_split = _pick(_split_candidates(list(v4_df.columns)), _SPLIT_PRIORITY)

    print(f"baseline: {_BASELINE_CSV.name} ({len(base_df)} rows)")
    print(f"      v4: {_V4_CSV.name} ({len(v4_df)} rows)")

    check1_columns(base_df, v4_df, base_label, base_split, v4_label, v4_split)
    check2_distribution(base_df, v4_df, base_label, v4_label)
    _hr(3)
    check3_leakage("baseline", base_df, base_split)
    check3_leakage("v4", v4_df, v4_split)
    check4_summary(v4_df, base_label, base_split, v4_label, v4_split)


if __name__ == "__main__":
    main()
