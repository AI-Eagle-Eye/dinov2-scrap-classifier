#!/usr/bin/env python3
"""발표용 시각화 단일 레이어 — 추론 재실행 없이 results/{exp}/ 산출물만 읽어 렌더한다.

기준 모듈은 src/evaluation/ 이다. evaluate_test.py / report_artifacts.py 가 만든
predictions.csv · summary.csv · threshold_sweep.csv · embeddings.npy 를 재사용하며,
공통 계산은 라이브러리(calibration.compute_ece, report_artifacts.danger_risk_coverage_arrays)를
재사용해 중복을 두지 않는다. report_artifacts 의 인라인 분석 PNG는 그대로 두고(다른 스크립트 공유),
이 모듈은 다크 테마 발표본을 results/{exp}/presentation/ 에 따로 생성한다.

전 그림은 다크 테마(#0f1117 배경 · 흰 텍스트 · 14pt+ · dpi 170)로 통일된다. 산출물:

  - KPI board                  → results/{exp}/presentation/{best_target}/kpi_board.png
  - Risk–Coverage (4-target)   → results/{exp}/presentation/risk_coverage_curve.png
  - summary_comparison (개선)  → results/{exp}/presentation/summary_comparison.png
  - 타겟별 → results/{exp}/presentation/{target}/ :
      confusion_matrix · pr_roc_curve · threshold_sweep · tsne_class · tsne_correct ·
      missed/false-alarm 갤러리(9장, conf=1.00 강조) · success_failure_comparison

threshold_sweep 은 report_artifacts.py 가 CSV만 남기므로 그 CSV를 읽어 PNG로 그린다(재추론 없음).
t-SNE 는 evaluate_test.py 가 저장한 embeddings.npy를 읽는다(없으면 안내 후 스킵 — 재실행 필요).

실행은 thin shim 으로 기존 경로를 유지한다:

    python evaluation/visualize_extra.py \
        --results_dir results/exp_ar_448_tta \
        --testset_root dataset/testset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ..data.dataset import CLASS_NAME_LIST, CUT, DANGER
from . import report_artifacts as ra
from .calibration import _DEFAULT_ECE_BINS, compute_ece, confidence_bin_masks

REPO_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES: list[str] = CLASS_NAME_LIST
CUT_IDX: int = CUT
DANGER_IDX: int = DANGER
_DANGER_COLOR = "#ff5c5c"  # 다크 배경 대비 가독성을 위해 약간 밝게
_OK_COLOR = "#3ddc84"
_HIGHLIGHT_COLOR = "#f4d35e"
# 다크 테마 팔레트 (#0f1117 배경 + 흰 텍스트)
_BG_COLOR = "#0f1117"
_FG_COLOR = "#ffffff"
_MUTED_COLOR = "#9aa0aa"  # 캡션/부제목 등 보조 텍스트
_PANEL_COLOR = "#1a1d27"  # 범례/카드 면 색
_GRID_COLOR = "#2a2e38"
_CONF_FULL = 0.995  # conf 표시값이 1.00 으로 반올림되는 경계 (별도 강조 대상)
_CORRECT_COLOR = "#4c9be8"  # t-SNE 정답 점 (파랑)
_TSNE_SEED = 42
_TARGET_ORDER = ["all", "crops_0pct", "crops_25pct", "crops_50pct"]
_BEST_TARGET = "crops_0pct"  # external testset에서 danger miss-rate가 가장 낮은 조건
_SUMMARY_INDEX = "eval_target"  # 내 summary.csv 인덱스 컬럼 (팀원은 'target')
_GALLERY_N = 9  # 3x3 그리드 — 썸네일을 크고 정렬되게 유지
_GALLERY_COLS = 3
_PANEL_TOP_N = 6
_ECE_BINS = _DEFAULT_ECE_BINS  # calibration 단일 출처 — 모든 산출물 ECE bin 수 통일
_SAVE_DPI = 170
_PROB_COLS = [f"prob_{c}" for c in CLASS_NAMES]


def _register_korean_font() -> str | None:
    """한글 캡션이 두부(tofu)로 깨지지 않게 폰트 탐색; family 또는 None 반환."""
    import matplotlib.font_manager as fm  # noqa: PLC0415

    for family in ("NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR", "Malgun Gothic"):
        try:
            fm.findfont(family, fallback_to_default=False)
            return family
        except (ValueError, FileNotFoundError):
            continue
    return None


def _apply_style() -> None:
    """모든 그림 공통 발표 테마 (다크 #0f1117 배경, 흰 텍스트, 14pt+, dpi 170, tight 저장)."""
    ko = _register_korean_font()
    family = [ko, "DejaVu Sans"] if ko else ["DejaVu Sans"]
    if ko is None:
        print("[warn] 한글 폰트를 찾지 못했습니다 → 한글 캡션이 깨질 수 있습니다.")
    plt.rcParams.update({
        "figure.dpi": _SAVE_DPI,
        "savefig.dpi": _SAVE_DPI,
        "savefig.bbox": "tight",
        "font.family": family,
        "axes.unicode_minus": False,
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.titlesize": 17,
        # 다크 테마
        "figure.facecolor": _BG_COLOR,
        "axes.facecolor": _BG_COLOR,
        "savefig.facecolor": _BG_COLOR,
        "savefig.edgecolor": _BG_COLOR,
        "text.color": _FG_COLOR,
        "axes.edgecolor": _GRID_COLOR,
        "axes.labelcolor": _FG_COLOR,
        "axes.titlecolor": _FG_COLOR,
        "xtick.color": _FG_COLOR,
        "ytick.color": _FG_COLOR,
        "grid.color": _GRID_COLOR,
        "legend.facecolor": _PANEL_COLOR,
        "legend.edgecolor": _GRID_COLOR,
        "legend.labelcolor": _FG_COLOR,
    })


def _save(fig: plt.Figure, out: Path, name: str) -> Path:
    """공통 저장 헬퍼 (dpi/bbox 고정)."""
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.savefig(p, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def _probs(df: pd.DataFrame) -> np.ndarray:
    return df[_PROB_COLS].to_numpy()


def _labels(df: pd.DataFrame, col: str) -> np.ndarray:
    idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    return df[col].map(idx).to_numpy()


def _resolve_image(row: pd.Series, testset_root: Path | None) -> Path | None:
    """predictions.csv의 image_path 우선 사용; 없으면 testset_root로 재구성."""
    stored = Path(str(row.image_path))
    candidates = [stored, REPO_ROOT / stored]
    if testset_root is not None:
        candidates.append(
            testset_root / f"crops_{int(row.crop_pct)}pct" / row.true_label / stored.name)
    for c in candidates:
        if c.exists():
            return c
    return None


def _target_color(target: str) -> str:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    idx = _TARGET_ORDER.index(target) if target in _TARGET_ORDER else 0
    return cycle[idx % len(cycle)]


def _ece_from_df(df: pd.DataFrame, n_bins: int = _ECE_BINS) -> float:
    """predictions.csv prob → max-softmax ECE (계산은 calibration.compute_ece 재사용)."""
    return compute_ece(_probs(df), _labels(df, "true_label"), n_bins)


# ---------------------------------------------------------------- per target

def plot_confusion(df: pd.DataFrame, out: Path) -> Path:
    """개선 confusion matrix: counts + row-normalized, colorbar 부제목 + danger→cut 강조."""
    yt, yp = _labels(df, "true_label"), _labels(df, "pred_label")
    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    row = cm.sum(1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros_like(cm, float), where=row > 0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panels = ((axes[0], cm, "Confusion (counts)", "d", "샘플 수 (count)"),
              (axes[1], norm, "Confusion (row-normalized)", ".2f", "행 정규화 비율 (0~1)"))
    for ax, data, title, fmt, cbar_label in panels:
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=cm.max() if fmt == "d" else 1.0)
        ax.set_xticks(range(n), CLASS_NAMES)
        ax.set_yticks(range(n), CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        thr = data.max() / 2
        for i in range(n):
            for j in range(n):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > thr else "black", fontsize=13)
        # danger→cut (치명적 오분류) 셀 빨간 테두리 강조
        ax.add_patch(Rectangle((CUT_IDX - 0.5, DANGER_IDX - 0.5), 1, 1,
                               fill=False, edgecolor=_DANGER_COLOR, lw=3))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=12, color=_FG_COLOR)
        cbar.ax.tick_params(colors=_FG_COLOR)
        cbar.outline.set_edgecolor(_GRID_COLOR)
    miss, dtot = int(cm[DANGER_IDX, CUT_IDX]), int(cm[DANGER_IDX].sum())
    fig.suptitle(
        f"danger → cut (critical miss) = {miss}/{dtot} = {miss / max(dtot, 1):.1%}\n"
        "왼쪽 = 샘플 수,  오른쪽 = 행 정규화 비율",
        color=_DANGER_COLOR, fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out, "confusion_matrix.png")


def plot_pr_roc(df: pd.DataFrame, out: Path) -> Path:
    """PR + ROC one-vs-rest 결합. PR에 클래스별 무작위 baseline(prevalence) 수평선 + 불균형 캡션."""
    probs, yt = _probs(df), _labels(df, "true_label")
    onehot = np.eye(len(CLASS_NAMES))[yt]
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, cls in enumerate(CLASS_NAMES):
        c = _DANGER_COLOR if i == DANGER_IDX else cycle[i % len(cycle)]
        ap = average_precision_score(onehot[:, i], probs[:, i])
        pr, rc, _ = precision_recall_curve(onehot[:, i], probs[:, i])
        axes[0].plot(rc, pr, color=c, lw=2, label=f"{cls} (AP={ap:.3f})")
        # 무작위 분류기 baseline = 해당 클래스 양성 비율(prevalence)
        axes[0].axhline(float(onehot[:, i].mean()), color=c, ls=":", lw=1.2, alpha=0.7)
        auc = roc_auc_score(onehot[:, i], probs[:, i])
        fpr, tpr, _ = roc_curve(onehot[:, i], probs[:, i])
        axes[1].plot(fpr, tpr, color=c, lw=2, label=f"{cls} (AUC={auc:.3f})")
    axes[0].plot([], [], color=_MUTED_COLOR, ls=":", lw=1.2, label="random baseline (prevalence)")
    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[0].set_title("PR (one-vs-rest)")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="lower left", framealpha=0.9)  # PR 곡선은 우상단 → 범례를 좌하단에
    axes[1].plot([0, 1], [0, 1], ls="--", color=_MUTED_COLOR, lw=1, label="random")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC (one-vs-rest)")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(loc="lower right", framealpha=0.9)  # ROC 곡선은 좌상단 → 범례를 우하단에
    fig.text(0.5, -0.02,
             "클래스 불균형 데이터라 ROC보다 PR이 신뢰도 높음 "
             "(점선 = 무작위 분류기 기준선 = 클래스 양성 비율)",
             ha="center", va="top", fontsize=12, style="italic", color=_MUTED_COLOR)
    fig.tight_layout()
    return _save(fig, out, "pr_roc_curve.png")


def plot_threshold_sweep(sweep_csv: Path, out: Path) -> Path | None:
    """threshold_sweep.csv(재추론 없음) → danger precision/recall/F1 곡선 + precision=recall 교차점 수직선(없으면 best-F1 폴백)."""
    if not sweep_csv.exists():
        print(f"[skip] threshold_sweep: {sweep_csv.name} 없음")
        return None
    s = pd.read_csv(sweep_csv).sort_values("thr")
    thr = s["thr"].to_numpy()
    prec = s["precision_danger"].to_numpy()
    rec = s["recall_danger"].to_numpy()
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom), where=denom > 0)
    mark_t, mark_label = ra.precision_recall_crossing(thr, prec, rec, f1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thr, prec, lw=2, label="danger precision")
    ax.plot(thr, rec, color=_DANGER_COLOR, lw=2, label="danger recall")
    ax.plot(thr, f1, lw=2, label="danger F1")
    ax.axvline(mark_t, color=_MUTED_COLOR, ls="--", lw=1.4, label=mark_label)
    ax.set_xlabel("danger one-vs-rest threshold")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Danger threshold sweep\n제안적 분석 (실제 평가는 threshold decision layer 사용)")
    ax.legend(loc="lower center", framealpha=0.9)
    fig.tight_layout()
    return _save(fig, out, "threshold_sweep.png")


def plot_tsne(df: pd.DataFrame, emb_path: Path, out: Path) -> list[Path]:
    """CLS 임베딩 t-SNE 2종: (1) 클래스별 색(범례 n 표기), (2) 정답=파랑/오답=빨강."""
    if not emb_path.exists():
        print(f"[skip] t-SNE: {emb_path} 없음 — evaluate_test.py 재실행 필요")
        return []
    feats = np.load(emb_path)
    yt, yp = _labels(df, "true_label"), _labels(df, "pred_label")
    if len(feats) != len(yt):
        print(f"[skip] t-SNE: 임베딩({len(feats)})과 predictions({len(yt)}) 길이 불일치 — 재실행 필요")
        return []
    perp = min(30, max(5, (len(feats) - 1) // 3))  # 표본 수에 맞춘 perplexity
    emb = TSNE(n_components=2, init="pca", perplexity=perp,
               random_state=_TSNE_SEED).fit_transform(feats)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    saved: list[Path] = []

    # 버전 1 — 클래스별 색
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, cls in enumerate(CLASS_NAMES):
        m = yt == i
        color = _DANGER_COLOR if i == DANGER_IDX else cycle[i % len(cycle)]
        ax.scatter(emb[m, 0], emb[m, 1], s=20, alpha=0.7, linewidths=0.3,
                   edgecolors=_BG_COLOR, color=color, label=f"{cls} (n={int(m.sum())})")
    ax.set_title("t-SNE — CLS 임베딩 (true class)")
    ax.legend(markerscale=1.5, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    saved.append(_save(fig, out, "tsne_class.png"))

    # 버전 2 — 정답/오답 색
    correct = yt == yp
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(emb[correct, 0], emb[correct, 1], s=18, alpha=0.6, linewidths=0.3,
               edgecolors=_BG_COLOR, color=_CORRECT_COLOR, label=f"correct (n={int(correct.sum())})")
    ax.scatter(emb[~correct, 0], emb[~correct, 1], s=30, alpha=0.8, linewidths=0.3,
               edgecolors=_BG_COLOR, color=_DANGER_COLOR, label=f"wrong (n={int((~correct).sum())})")
    ax.set_title("t-SNE — CLS 임베딩 (correct vs wrong)")
    ax.legend(markerscale=1.4, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    saved.append(_save(fig, out, "tsne_correct.png"))
    return saved


# confidence-bin 다지표 reliability: (key, 축 제목, 한국어 캡션).
# accuracy는 고전적 보정(대각선+ECE) 해석, 나머지는 "성능 vs 확신" 으로 읽는다.
_REL_METRICS: list[tuple[str, str, str]] = [
    ("accuracy", "accuracy", "전체 정확도 (보정)"),
    ("danger_precision", "danger precision", "danger 예측의 정밀도"),
    ("danger_recall", "danger recall (1 - miss)", "danger 놓치지 않은 비율"),
    ("f1_macro", "macro F1", "클래스 균형 성능"),
]


def _bin_metric(metric: str, yt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """confidence bin 하나의 지표 값 (해당 bin에서 정의 불가면 NaN)."""
    t, p = yt[mask], pred[mask]
    if metric == "accuracy":
        return float((p == t).mean())
    if metric == "danger_precision":
        sel = p == DANGER_IDX
        return float((t[sel] == DANGER_IDX).mean()) if sel.any() else np.nan
    if metric == "danger_recall":
        pos = t == DANGER_IDX
        return float((p[pos] == DANGER_IDX).mean()) if pos.any() else np.nan
    return float(f1_score(t, p, average="macro", zero_division=0))  # f1_macro


def plot_reliability(df: pd.DataFrame, out: Path, n_bins: int = _ECE_BINS) -> Path:
    """2x2 confidence-bin 패널: accuracy(보정)+danger P/R+macro F1, bin 표본수 막대 오버레이."""
    probs, yt = _probs(df), _labels(df, "true_label")
    conf, pred = probs.max(1), probs.argmax(1)
    _, masks = confidence_bin_masks(conf, n_bins)
    masks = [m for m in masks if m.sum() > 0]
    bin_conf = [conf[m].mean() for m in masks]
    bin_n = [int(m.sum()) for m in masks]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, (key, ylabel, caption) in zip(axes.ravel(), _REL_METRICS):
        ys = [_bin_metric(key, yt, pred, m) for m in masks]
        twin = ax.twinx()  # 희소 bin이 보이도록 옅은 표본수 막대
        twin.bar(bin_conf, bin_n, width=0.045, color=_MUTED_COLOR, alpha=0.30, zorder=0)
        twin.set_ylabel("bin 표본 수", fontsize=11, color=_MUTED_COLOR)
        twin.tick_params(axis="y", labelcolor=_MUTED_COLOR, labelsize=10)
        if key == "accuracy":
            ax.plot([0, 1], [0, 1], ls="--", color=_MUTED_COLOR, label="perfect", zorder=2)
            ece = compute_ece(probs, yt, n_bins)  # calibration 단일 출처 재사용 (중복 계산 제거)
            ax.set_title(f"{ylabel}  (ECE={ece:.3f})", fontsize=15)
        else:
            ax.set_title(ylabel, fontsize=15)
        ax.plot(bin_conf, ys, "o-", color=_DANGER_COLOR, zorder=3, label="model")
        ax.set_xlabel("confidence")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_zorder(twin.get_zorder() + 1)  # 지표 선을 막대 위로
        ax.patch.set_visible(False)
        ax.text(0.5, -0.16, caption, transform=ax.transAxes, ha="center", va="top",
                fontsize=11, style="italic", color=_MUTED_COLOR)
        if key == "accuracy":
            ax.legend(loc="lower right", fontsize=10)
    fig.suptitle("Reliability / confidence-bin metrics (x = 모델 확신도)", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out, "reliability_multi.png")


def plot_umap(df: pd.DataFrame, emb_path: Path, out: Path) -> list[Path]:
    """저장된 embeddings.npy 재사용 UMAP 2종 (클래스별 / 정답·오답) — 재추론 없음.

    plot_tsne와 동일 입력/레이아웃. umap-learn은 optional(requirements 참고)이라
    미설치 시 안내 후 스킵한다. embeddings 길이 불일치도 스킵.
    """
    if not emb_path.exists():
        print(f"[skip] UMAP: {emb_path} 없음 — evaluate_test.py 재실행 필요")
        return []
    try:
        import umap  # noqa: PLC0415
    except ImportError:
        print("[skip] UMAP: umap-learn 미설치 (pip install umap-learn)")
        return []
    feats = np.load(emb_path)
    yt, yp = _labels(df, "true_label"), _labels(df, "pred_label")
    if len(feats) != len(yt):
        print(f"[skip] UMAP: 임베딩({len(feats)})과 predictions({len(yt)}) 길이 불일치 — 재실행 필요")
        return []
    n_neighbors = min(15, max(2, len(feats) - 1))  # 표본 수에 맞춘 이웃 수
    emb = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                    random_state=_TSNE_SEED).fit_transform(feats)
    caption = "t-SNE/UMAP 모두 비선형 축소 — 거리·밀도 해석 금지 (군집의 상대적 분리만 참고)"
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    saved: list[Path] = []

    # 버전 1 — 클래스별 색
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, cls in enumerate(CLASS_NAMES):
        m = yt == i
        color = _DANGER_COLOR if i == DANGER_IDX else cycle[i % len(cycle)]
        ax.scatter(emb[m, 0], emb[m, 1], s=20, alpha=0.7, linewidths=0.3,
                   edgecolors=_BG_COLOR, color=color, label=f"{cls} (n={int(m.sum())})")
    ax.set_title("UMAP — CLS 임베딩 (true class)")
    ax.legend(markerscale=1.5, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, -0.06, caption, transform=ax.transAxes, ha="center", va="top",
            fontsize=10, style="italic", color=_MUTED_COLOR)
    fig.tight_layout()
    saved.append(_save(fig, out, "umap_class.png"))

    # 버전 2 — 정답/오답 색 (신규)
    correct = yt == yp
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(emb[correct, 0], emb[correct, 1], s=18, alpha=0.6, linewidths=0.3,
               edgecolors=_BG_COLOR, color=_CORRECT_COLOR, label=f"correct (n={int(correct.sum())})")
    ax.scatter(emb[~correct, 0], emb[~correct, 1], s=30, alpha=0.8, linewidths=0.3,
               edgecolors=_BG_COLOR, color=_DANGER_COLOR, label=f"wrong (n={int((~correct).sum())})")
    ax.set_title("UMAP — CLS 임베딩 (correct vs wrong)")
    ax.legend(markerscale=1.4, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, -0.06, caption, transform=ax.transAxes, ha="center", va="top",
            fontsize=10, style="italic", color=_MUTED_COLOR)
    fig.tight_layout()
    saved.append(_save(fig, out, "umap_correct.png"))
    return saved


def _gallery(df: pd.DataFrame, true_i: int, pred_i: int, sort_col: str,
             testset_root: Path | None, out: Path, fname: str, title: str) -> Path | None:
    """오분류 케이스 top-N(9) 썸네일 그리드 (confidence 높은 순)."""
    sub = df[(df.true_label == CLASS_NAMES[true_i]) & (df.pred_label == CLASS_NAMES[pred_i])]
    if sub.empty:
        print(f"[skip] {fname}: 해당 케이스 없음")
        return None
    sub = sub.sort_values(sort_col, ascending=False).head(_GALLERY_N)
    cols = _GALLERY_COLS
    rows = int(np.ceil(_GALLERY_N / cols))  # 케이스가 9개 미만이어도 3x3 격자 고정
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.6))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    loaded = 0
    for ax, (_, r) in zip(axes, sub.iterrows()):
        img_path = _resolve_image(r, testset_root)
        if img_path is not None:
            with Image.open(img_path) as im:
                ax.imshow(im.convert("RGB"), aspect="auto")  # aspect="auto" → 정렬된 그리드
                loaded += 1
        else:
            ax.text(0.5, 0.5, "img missing", ha="center", va="center", fontsize=13)
        conf = float(r[f"prob_{r.pred_label}"])
        full = conf >= _CONF_FULL  # conf=1.00 케이스 별도 강조
        prefix = "★ " if full else ""
        ax.set_title(f"{prefix}T:{r.true_label}  P:{r.pred_label}\nconf={conf:.2f}",
                     fontsize=14, color=_HIGHLIGHT_COLOR if full else _DANGER_COLOR,
                     fontweight="bold" if full else "normal")
        if full:  # 확신 오분류 → 노란 테두리로 시선 유도
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                   edgecolor=_HIGHLIGHT_COLOR, lw=4, zorder=5))
    if loaded == 0:
        plt.close(fig)
        print(f"[skip] {fname}: 이미지를 하나도 찾지 못함 (--testset_root 확인)")
        return None
    fig.suptitle(f"{title}  (top {len(sub)},  ★ = conf 1.00)", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out, fname)


def _panel_thumb(ax: plt.Axes, testset_root: Path | None, r: pd.Series, color: str) -> bool:
    ax.axis("off")
    img_path = _resolve_image(r, testset_root)
    ok = False
    if img_path is not None:
        with Image.open(img_path) as im:
            ax.imshow(im.convert("RGB"), aspect="auto")
            ok = True
    else:
        ax.text(0.5, 0.5, "img missing", ha="center", va="center", fontsize=11)
    ax.set_title(f"T:{r.true_label}  P:{r.pred_label}\nconf={r[f'prob_{r.pred_label}']:.2f}",
                 fontsize=11, color=color)
    return ok


def plot_success_failure(df: pd.DataFrame, testset_root: Path | None, out: Path) -> Path | None:
    """성공(danger→danger) top-6 vs 실패(danger→cut) top-6 비교 패널."""
    correct = (df[(df.true_label == "danger") & (df.pred_label == "danger")]
               .sort_values("prob_danger", ascending=False).head(_PANEL_TOP_N))
    missed = (df[(df.true_label == "danger") & (df.pred_label == "cut")]
              .sort_values("prob_cut", ascending=False).head(_PANEL_TOP_N))
    if correct.empty and missed.empty:
        print("[skip] success_failure_comparison: 해당 케이스 없음")
        return None
    fig, axes = plt.subplots(3, 4, figsize=(4 * 3.2, 3 * 3.4))
    for ax in axes.ravel():
        ax.axis("off")
    loaded = 0
    for i, (_, r) in enumerate(correct.iterrows()):
        loaded += _panel_thumb(axes[i // 2, i % 2], testset_root, r, _OK_COLOR)
    for i, (_, r) in enumerate(missed.iterrows()):
        loaded += _panel_thumb(axes[i // 2, 2 + i % 2], testset_root, r, _DANGER_COLOR)
    if loaded == 0:
        plt.close(fig)
        print("[skip] success_failure_comparison: 이미지를 찾지 못함 (--testset_root 확인)")
        return None
    fig.suptitle("성공 vs 실패 비교 (danger, confidence 높은 순 top-6)", fontsize=18)
    fig.text(0.30, 0.94, "정답 danger → danger", ha="center", fontsize=15,
             fontweight="bold", color=_OK_COLOR)
    fig.text(0.72, 0.94, "놓친 danger → cut", ha="center", fontsize=15,
             fontweight="bold", color=_DANGER_COLOR)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.add_artist(Line2D([0.5, 0.5], [0.02, 0.90], transform=fig.transFigure,
                          color="#bbbbbb", lw=1.5))
    return _save(fig, out, "success_failure_comparison.png")


# ---------------------------------------------------------------- cross target

def plot_summary(summary_csv: Path, out: Path) -> Path:
    """4-target 비교 막대 (miss_rate ↓ 주석 + best 조건 강조)."""
    df = pd.read_csv(summary_csv).set_index(_SUMMARY_INDEX)
    order = [t for t in _TARGET_ORDER if t in df.index]
    df = df.reindex(order)
    metrics = ["accuracy", "f1_macro", "recall_danger", "miss_rate_danger"]
    metric_labels = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "recall_danger": "recall_danger",
        "miss_rate_danger": "miss_rate_danger  (↓ lower is better)",
    }
    x = np.arange(len(order))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6.5))
    if _BEST_TARGET in order:
        best_x = order.index(_BEST_TARGET)
        ax.axvspan(best_x - 0.5, best_x + 0.5, color=_HIGHLIGHT_COLOR, alpha=0.22, zorder=0)
        ax.text(best_x, 1.02, "★ best (lowest danger miss)", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=_HIGHLIGHT_COLOR)
    for k, m in enumerate(metrics):
        off = (k - (len(metrics) - 1) / 2) * width
        color = _DANGER_COLOR if m == "miss_rate_danger" else None
        bars = ax.bar(x + off, df[m].values, width, label=metric_labels[m], color=color, zorder=3)
        for b, v in zip(bars, df[m].values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10, zorder=4)
    ax.set_xticks(x, order)
    if _BEST_TARGET in order:
        ax.get_xticklabels()[order.index(_BEST_TARGET)].set_fontweight("bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title("4-target comparison (external testset)")
    ax.legend(ncol=2, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return _save(fig, out, "summary_comparison.png")


def plot_risk_coverage(results_dir: Path, targets: list[str], out: Path) -> Path:
    """4-target overlay Risk–Coverage 곡선 (precision 좌축 / miss-rate 우축).

    sweep 계산은 report_artifacts.danger_risk_coverage_arrays 를 재사용한다(중복 제거).
    """
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax2 = ax.twinx()
    for target in targets:
        df = pd.read_csv(results_dir / target / "predictions.csv")
        cov, prec, miss = ra.danger_risk_coverage_arrays(_probs(df), _labels(df, "true_label"))
        c = _target_color(target)
        ax.plot(cov, prec, color=c, lw=2.2, label=target)
        ax2.plot(cov, miss, color=c, lw=1.6, ls="--", alpha=0.9)
    ax.set_xlabel("Coverage (danger로 판정한 비율)")
    ax.set_ylabel("Precision (Danger)")
    ax2.set_ylabel("Miss Rate (Danger)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax2.set_ylim(0, 1.02)
    ax.set_title("Risk–Coverage curve (danger threshold sweep)")
    target_leg = ax.legend(loc="lower left", title="target", framealpha=0.9)
    target_leg.get_title().set_color(_FG_COLOR)
    ax.add_artist(target_leg)
    style_handles = [Line2D([], [], color=_MUTED_COLOR, lw=2.2, label="Precision (left axis)"),
                     Line2D([], [], color=_MUTED_COLOR, lw=1.6, ls="--", label="Miss Rate (right axis)")]
    ax.legend(handles=style_handles, loc="upper center", framealpha=0.9)
    fig.text(0.5, -0.02,
             "임계값↑ → coverage↓·precision↑·miss-rate↑. 운영점(자동 판정 비율) 선택에 사용.",
             ha="center", va="top", fontsize=12, style="italic", color=_MUTED_COLOR)
    fig.tight_layout()
    return _save(fig, out, "risk_coverage_curve.png")


def _draw_kpi_card(ax: plt.Axes, label: str, value: float | None,
                   target: float, lower_is_better: bool, emphasize: bool) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(label, fontsize=15, pad=10)
    if value is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=34, color="gray")
        return
    meets = value <= target if lower_is_better else value >= target
    num_color = _DANGER_COLOR if emphasize else (_OK_COLOR if meets else "#e08a00")
    ax.text(0.5, 0.60, f"{value:.2f}", ha="center", va="center",
            fontsize=38, fontweight="bold", color=num_color)
    x0, w, y0, h = 0.12, 0.76, 0.20, 0.07
    ax.add_patch(Rectangle((x0, y0), w, h, color="#e6e6e6", zorder=1))
    fill = float(np.clip(value, 0.0, 1.0))
    ax.add_patch(Rectangle((x0, y0), w * fill, h,
                           color=(_OK_COLOR if meets else num_color), zorder=2))
    tx = x0 + w * float(np.clip(target, 0.0, 1.0))
    ax.plot([tx, tx], [y0 - 0.03, y0 + h + 0.03], color=_FG_COLOR, lw=1.6, zorder=3)
    cmp = "≤" if lower_is_better else "≥"
    mark = "✓" if meets else "✗"
    ax.text(0.5, 0.05, f"목표 {cmp}{target:.2f}  {mark}", ha="center", va="center",
            fontsize=12, color=(_OK_COLOR if meets else _DANGER_COLOR))


def plot_kpi_board(summary_csv: Path, results_dir: Path, out: Path,
                   target_name: str = _BEST_TARGET) -> Path:
    """best 조건(crops_0pct) KPI 보드: Accuracy/Precision/Miss/F1/ECE 게이지 + 목표선."""
    sdf = pd.read_csv(summary_csv).set_index(_SUMMARY_INDEX)
    if target_name not in sdf.index:
        target_name = str(sdf.index[0])
    row = sdf.loc[target_name]
    preds = results_dir / target_name / "predictions.csv"
    ece = _ece_from_df(pd.read_csv(preds)) if preds.exists() else None
    # (label, value, target, lower_is_better, emphasize) — 목표값은 예시 goal
    cards = [
        ("Accuracy", float(row["accuracy"]), 0.80, False, False),
        ("Precision (Danger)", float(row["precision_danger"]), 0.85, False, False),
        ("Miss Rate (Danger)", float(row["miss_rate_danger"]), 0.10, True, True),
        ("Macro F1", float(row["f1_macro"]), 0.70, False, False),
        ("ECE", ece, 0.05, True, False),
    ]
    fig, axes = plt.subplots(1, len(cards), figsize=(3.2 * len(cards), 4.4))
    for ax, (label, value, tgt, lower, emph) in zip(np.atleast_1d(axes), cards):
        _draw_kpi_card(ax, label, value, tgt, lower, emph)
    fig.suptitle(f"KPI board — {target_name} (best 조건, external testset)", fontsize=18)
    fig.text(0.5, 0.02, "막대 = 측정값 · 흰선 = 목표선 · 색상 = 목표 달성 여부 (목표값은 예시)",
             ha="center", va="bottom", fontsize=11, style="italic", color=_MUTED_COLOR)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    return _save(fig, out, "kpi_board.png")


# ---------------------------------------------------------------- driver

def _resolve_results_dir(raw: str) -> Path:
    rd = Path(raw)
    if rd.exists():
        return rd
    alt = REPO_ROOT / raw
    if alt.exists():
        return alt
    sys.exit(f"[ERROR] 결과 폴더가 없습니다: {raw}\n"
             "        먼저 `python scripts/evaluate_test.py ...` 를 실행하세요.")


def _resolve_testset_root(raw: str) -> Path | None:
    tr = Path(raw)
    if not tr.exists() and (REPO_ROOT / raw).exists():
        tr = REPO_ROOT / raw
    return tr if tr.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Hazard-detection 발표용 시각화 (추론 재실행 없음)")
    ap.add_argument("--results_dir", required=True, help="예: results/exp_ar_448_tta")
    ap.add_argument("--testset_root", default="dataset/testset",
                    help="갤러리/패널 이미지 로드용 (predictions.csv의 image_path 우선)")
    ap.add_argument("--out", default=None, help="기본값: results/{exp}/presentation")
    args = ap.parse_args()

    _apply_style()

    results_dir = _resolve_results_dir(args.results_dir)
    # 기본 출력은 results/{exp}/presentation — exp namespace로 실험 간 덮어쓰기 방지
    plots_root = Path(args.out) if args.out else results_dir / "presentation"
    testset_root = _resolve_testset_root(args.testset_root)
    if testset_root is None:
        print(f"[warn] testset_root 미발견: {args.testset_root} → image_path 절대/상대 경로로만 시도")

    targets = [t for t in _TARGET_ORDER if (results_dir / t / "predictions.csv").exists()]
    if not targets:
        sys.exit(f"[ERROR] {results_dir} 하위에 predictions.csv 가 없습니다.")

    saved: list[Path] = []
    for target in targets:
        tdir = results_dir / target
        out = plots_root / target
        df = pd.read_csv(tdir / "predictions.csv")
        saved.append(plot_confusion(df, out))
        saved.append(plot_pr_roc(df, out))
        ts = plot_threshold_sweep(tdir / "threshold_sweep.csv", out)
        saved.append(plot_reliability(df, out))
        saved += plot_tsne(df, tdir / "embeddings.npy", out)
        saved += plot_umap(df, tdir / "embeddings.npy", out)
        g1 = _gallery(df, DANGER_IDX, CUT_IDX, "prob_cut", testset_root, out,
                      "missed_danger_gallery.png", "Missed danger (true danger → pred cut)")
        g2 = _gallery(df, CUT_IDX, DANGER_IDX, "prob_danger", testset_root, out,
                      "false_alarm_gallery.png", "False alarm (true cut → pred danger)")
        sf = plot_success_failure(df, testset_root, out)
        saved += [p for p in (ts, g1, g2, sf) if p is not None]
        print(f"[OK] {target} 발표 플롯 완료 → {out}/")

    summary_csv = results_dir / "summary.csv"
    if summary_csv.exists():
        saved.append(plot_summary(summary_csv, plots_root))
    saved.append(plot_risk_coverage(results_dir, targets, plots_root))
    if summary_csv.exists():
        best = _BEST_TARGET if _BEST_TARGET in targets else targets[0]
        saved.append(plot_kpi_board(summary_csv, results_dir, plots_root / best, target_name=best))

    print("\n[OK] 생성된 파일:")
    for p in saved:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
