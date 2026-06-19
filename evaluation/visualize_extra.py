#!/usr/bin/env python3
"""기존 실행 경로 유지용 thin shim — 발표용 시각화 로직은 src/evaluation/figures.py 로 이전됨.

    python evaluation/visualize_extra.py \
        --results_dir results/exp_ar_448_tta \
        --testset_root dataset/testset
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.figures import main  # noqa: E402

if __name__ == "__main__":
    main()
