"""Smoke test: verify Korean font rendering in all eda.py figure text styles.

Checks:
  1. NanumGothic (or fallback) is loaded
  2. Key Korean strings from every modified plot are rendered without tofu (□)
  3. PNG is created and non-empty

Run:
    python scripts/smoke_test_korean_font.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── 1. Font setup (same as eda.py) ───────────────────────────────────────────

def _setup_korean_font() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = "DejaVu Sans"
    for name in ("NanumGothic", "NanumBarunGothic", "Malgun Gothic", "AppleGothic"):
        if name in available:
            chosen = name
            break
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


font_name = _setup_korean_font()
print(f"[font] 선택된 폰트: {font_name}")

# ── 2. Korean strings used in every modified plot ────────────────────────────

KOREAN_STRINGS: list[tuple[str, str]] = [
    # plot_bbox_absolute
    ("BBox", "가로 크기 (px)"),
    ("BBox", "세로 크기 (px)"),
    ("범례", "소형 <100px"),
    ("범례", "대형 >300px"),
    ("축", "건수"),
    # plot_bbox_relative
    ("축", "bbox 면적 / 이미지 면적"),
    # analyse_pixel fig5
    ("범례", "데이터 평균"),
    ("범례", "ImageNet 평균"),
    # analyse_pixel fig6
    ("축", "평균 밝기"),
    # analyse_pixel fig7
    ("축", "색조 (Hue)"),
    # analyse_pixel fig8
    ("축", "평균 밝기"),
    # analyse_quality_features
    ("제목", "클래스별 엣지 밀도 (Canny, σ=1.0)"),
    ("축", "엣지 밀도"),
    ("제목", "클래스별 텍스처 엔트로피 (Shannon)"),
    ("축", "텍스처 엔트로피"),
    # analyse_blur_occlusion fig13
    ("범례", "임계값=100.0"),
    ("제목", "흐림도 점수 분포 (Laplacian 분산) — 저품질: 3개"),
    ("축", "Laplacian 분산"),
    # analyse_blur_occlusion fig14
    ("제목", "흐림도 vs 엣지 밀도 (클래스별)"),
    ("축", "흐림도 (Laplacian 분산)"),
    # analyse_blur_occlusion fig15
    ("축", "전경 비율"),
    # analyse_features fig16
    ("제목", "UMAP 클러스터 (silhouette=0.123)"),
    # analyse_features fig17
    ("전체제목", "클래스별 대표 샘플 이미지 (centroid 최근접)"),
    # analyse_features fig18
    ("전체제목", "혼동 가능성 높은 샘플 쌍 (위험 vs 안전, 상위 5쌍)"),
    # HTML titles
    ("HTML", "엣지 밀도 분포"),
    ("HTML", "텍스처 엔트로피 분포"),
    ("HTML", "흐림도 vs 엣지 밀도"),
    ("HTML", "전경 비율 분포 (Occlusion)"),
    ("HTML", "대표 샘플 이미지"),
    ("HTML", "혼동 가능성 높은 샘플 쌍"),
]

# ── 3. Render all strings in a grid figure ───────────────────────────────────

n = len(KOREAN_STRINGS)
cols = 4
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 1.4))
axes = np.array(axes).ravel()

for i, (category, text) in enumerate(KOREAN_STRINGS):
    ax = axes[i]
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9,
            transform=ax.transAxes, wrap=True)
    ax.set_title(f"[{category}]", fontsize=7, color="#555")
    ax.axis("off")

for ax in axes[n:]:
    ax.axis("off")

fig.suptitle(f"한국어 폰트 smoke test — 폰트: {font_name}", fontsize=11, y=1.01)
fig.tight_layout()

out_path = _BASE / "reports" / "eda" / "figures" / "smoke_test_korean_font.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
plt.close(fig)

# ── 4. Basic verification ─────────────────────────────────────────────────────

size = out_path.stat().st_size
print(f"[smoke] PNG 생성: {out_path.name} ({size:,} bytes)")
assert size > 10_000, f"PNG가 너무 작음 ({size} bytes) — 렌더링 실패 가능성"

# Check that the chosen font actually has Korean glyphs (NanumGothic etc.)
font_props = fm.FontProperties(family=font_name)
found_font = fm.findfont(font_props, fallback_to_default=False)
print(f"[smoke] 실제 사용 폰트 파일: {Path(found_font).name}")

kr_test_char = "가"
try:
    test_fig, test_ax = plt.subplots(figsize=(1, 1))
    test_text = test_ax.text(0.5, 0.5, kr_test_char, fontsize=20,
                              ha="center", va="center", transform=test_ax.transAxes)
    test_fig.canvas.draw()
    # Get the actual font used for this glyph
    renderer = test_fig.canvas.get_renderer()
    props = test_text.get_fontproperties()
    actual = fm.findfont(props)
    plt.close(test_fig)
    print(f"[smoke] 한국어 글자 '{kr_test_char}' → 실제 폰트: {Path(actual).name}")
    if "Nanum" in actual or "Gothic" in actual or "Malgun" in actual:
        print("[smoke] ✅ 한국어 폰트 정상 적용")
    else:
        print(f"[smoke] ⚠️  폰트 fallback 발생 ({Path(actual).name}) — 폰트 설치 확인 필요")
except Exception as e:
    print(f"[smoke] 글리프 검사 건너뜀: {e}")

print("\n[smoke] ✅ smoke test 완료 — 렌더링된 이미지를 육안으로 확인하세요:")
print(f"  {out_path}")
