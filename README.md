# dinov2-scrap-classifier

스크랩 야드에서 용광로로 투입되는 금속 스크랩 중 **밀폐 용기(LPG통, 산소통 등)를 자동으로 탐지**하는 보조 분류 모델입니다.

> 사람이 최종 판단하는 보조 시스템입니다. 모델 단독으로 통과/차단을 결정하지 않습니다.

---

## 배경

용광로에 밀폐 용기가 투입되면 폭발 사고로 이어질 수 있습니다. 기존에는 작업자가 육안으로 스크랩을 확인했으나, 대량 처리 환경에서는 누락이 발생합니다. 이 시스템은 카메라 영상에서 bbox crop된 이미지를 실시간으로 분류해 위험 후보를 작업자에게 알립니다.

---

## 태스크

| 항목 | 내용 |
|---|---|
| 입력 | bbox crop 이미지 (25% padding, 336×336 LetterBox) |
| 출력 | 3-class 분류 확률 + threshold 기반 최종 판정 |
| 추론 환경 | RTX 4090, ONNX Runtime, 파이프라인 전체 5GB 이하 |

### 클래스 정의

```
위험 (danger)   — 밀폐 상태 유지, 용광로 투입 시 폭발 위험
안전 (cut)      — 절단 확인, 통과 가능
제외 (excluded) — 판단 불가 또는 위험/안전 구분 어려움
```

---

## 핵심 지표

이 프로젝트에서 가장 중요한 실패는 **위험을 안전으로 오판하는 것**입니다.

| 지표 | 정의 | 방향 |
|---|---|---|
| **Danger-as-Safe Rate** | 실제 위험 중 안전으로 판정된 비율 | 최소화 |
| Safe Precision | 안전 판정 중 실제 안전 비율 | 최대화 |
| Danger Recall | 실제 위험을 위험으로 잡은 비율 | 높을수록 좋음 |

결정 방식은 argmax가 아닌 **threshold 기반 decision layer**를 사용합니다.  
`danger_prob >= 0.30`이면 danger로 판정하고, 그 외에는 argmax를 적용합니다.

---

## 아키텍처

```
DINOv2 ViT-B/14 (frozen + 마지막 4블록 LLRD 미세조정)
  └─ AttentionHead (use_cls)
       - nn.MultiheadAttention (embed_dim=768, num_heads=8)
       - [CLS + patch] self-attention → mean pool → LayerNorm
       - Linear(768 → 3)
```

- **Backbone**: `dinov2_vitb14` — 기본 frozen, 마지막 4개 block만 LLRD(decay 0.75)로 미세조정
- **Head**: `AttentionHead` — CLS token을 patch token 앞에 붙여(CLS+patch) self-attention 후 mean pool로 집계 (`src/models/head.py`)
- **입력 크기**: 336×336 (기본) / 448×448 (고해상도 변형), bbox crop + 25% padding, LetterBox, patch_size=14
- **Loss**: FocalLoss(γ=2.0) + inverse-frequency class weight, label smoothing 0.2
- **Optimizer**: AdamW + LLRD(블록별 lr 감쇠) + EMA(decay 0.999~0.9995)
- **Scheduler**: LinearWarmup(10%) → CosineAnnealingLR
- **결정 규칙**: argmax가 아닌 **threshold 기반** — `danger_prob >= thr`이면 danger, thr은 val에서 Danger-as-Safe ≤ 0.15 제약으로 선택(no-leakage)

### 아키텍처 선택 근거

Attention head가 MLP head 대비 patch token 전체를 활용해 절단면 위치와 맥락을 더 잘 포착합니다.  
backbone은 frozen을 기본으로 하며, 이 태스크에서 VPT/PAA 등의 추가 모듈은 유의미한 개선을 보이지 않았습니다 (상세 내용은 [Ablation 결과](#ablation-결과) 참조).

---

## 데이터셋

기업 제공 데이터만 사용합니다. 공개 데이터셋은 학습 파이프라인에서 완전히 제외합니다.

| 클래스 | 샘플 수 |
|---|---|
| cut (안전) | 9,713 |
| danger (위험) | 5,393 |
| excluded (제외) | 2,797 |
| **합계** | **17,901** |

- **Split**: train 70% / val 15% / test 15%
- **Split 방식**: DINOv2 feature DBSCAN 클러스터 단위 배정 (유사 프레임이 train/test에 분리되지 않도록 데이터 유출 방지)
- **전처리**: LetterBox(336) + 커스텀 정규화 (ImageNet 아닌 데이터 실측값)
- **데이터 경로**: `.gitignore` 필수 — 절대 커밋하지 않습니다

---

## 실험 결과

### 최종 모델 성능 (외부 testset, crops_25pct 기준)

| 모델 | Precision(Danger) | Miss Rate (Danger-as-Safe) | Macro F1 |
|---|---|---|---|
| AR+TTA (448) | 87.83% | 10.15% | 0.540 |
| 앙상블 336+448 | 87.31% | **8.65%** | 0.540 |

- 두 모델 모두 단일 baseline 대비 핵심 안전지표(Danger-as-Safe)를 억제
- **앙상블 336+448**이 Miss Rate 8.65%로 가장 보수적 — 납품 관점 권장 구성
- threshold는 val에서 Danger-as-Safe ≤ 0.15 제약으로 선택(no-leakage), 외부 testset에 고정 적용
- 전체 12개 실험 × 4타겟(all/crops_0/25/50pct) 비교는 `results/experiments_comparison.csv` / `results/experiments_comparison.png` 참조

### Ablation 결과

실험 A~Q를 통해 확인한 주요 결과입니다. 상세 수치는 `experiments/` 산출물(gitignore) 참조.

| 항목 | 결과 |
|---|---|
| Attention head > MLP head | patch token 전체 활용으로 danger 판별 개선 |
| FocalLoss γ=2.0 | 클래스 불균형 억제. γ=3.0 ablation에서 추가 이득 없음 |
| lr 분리 (head 5e-5 / backbone 1e-5) | head 빠른 수렴 + backbone 미세 적응 균형 |
| threshold=0.30 | val sweep에서 danger_as_safe < 10% 제약 만족하는 최적점 |

**효과 없었던 시도:**

| 항목 | 제외 이유 |
|---|---|
| VPT (전체 / Partial) | frozen backbone 대비 danger_as_safe 개선 없음, VRAM만 증가 |
| SupCon Loss | contrastive 표현이 최종 분류 지표로 이어지지 않음 |
| CE + DangerRecallLoss | danger recall 직접 최대화가 precision 과하게 희생 |
| LearnableClassWeight | 상수 inverse-freq 가중치와 거의 동일하게 수렴 |
| ClassAwareHead | 단순 attention head 대비 개선 없음 |
| EVA-02 backbone | DINOv2 대비 우위 없음, 입력 448로 비용만 증가 |
| WeightedRandomSampler | Focal Loss와 동시 적용 시 소수 클래스 이중 보정으로 불안정 |

---

## 실험 목록

`experiments/` 하위 학습 산출물(체크포인트·로그·plot). 가중치는 `.gitignore`로 제외하며, 평가 지표는 `results/`에 별도 보관합니다.

| 실험 | 요약 |
|---|---|
| `exp_ab_mean_pad` | LLRD decay=0.75 + cosine warmup, mean 패딩 baseline |
| `exp_ah_orig` | original_label 기준 학습·평가 |
| `exp_ah_regularize` | dropout 0.3 / wd 5e-2 / 마지막 4블록 unfrozen 강한 정규화 |
| `exp_aj_linear` | pure linear probe (CLS → 3) |
| `exp_ak_regularize2` | dropout 0.2 / unfreeze 2 / lr·wd 재튜닝 |
| `exp_al_ema` | EMA(decay 0.999) + v2_label |
| `exp_am_cls_att` | AttentionHead with CLS(use_cls) + EMA |
| `exp_an_tuned` | exp_am 튜닝 — dropout·unfreeze·lr·llrd 조정, f1_macro best |
| `exp_ap_f1best` | cls_att base, f1_macro best metric (336 최종 후보) |
| `exp_aq_multipad` | multipad(crops_0/10/25pct) 3배 확장 |
| `exp_ar_448` | exp_ap base를 448 해상도로, 강한 aug (고해상도 최종 후보) |

평가 전용 결과 디렉토리: `exp_ar_448_tta`(AR+TTA), `exp_ensemble_336_448`(336+448 logit 앙상블).

---

## 디렉토리 구조

```
.
├── src/
│   ├── models/
│   │   ├── backbone.py       # DINOv2 로드 및 forward (frozen)
│   │   ├── head.py           # AttentionHead — patch token self-attention + mean pool
│   │   ├── vpt.py            # VPT prompt tokens (ablation용, 미채택)
│   │   └── hazard_model.py   # 전체 모델 조합
│   ├── data/
│   │   ├── dataset.py        # HazardDataset, split 컬럼 기반 로드
│   │   └── transforms.py     # LetterBox, 커스텀 정규화
│   ├── training/
│   │   ├── trainer.py        # 학습 루프, early stopping
│   │   ├── focal_loss.py     # FocalLoss(γ=2.0)
│   │   └── checkpoint.py     # 5종 체크포인트 저장/로드
│   ├── evaluation/
│   │   ├── evaluator.py      # Danger-as-Safe Rate 등 지표 계산
│   │   ├── threshold.py      # threshold sweep, decision layer
│   │   └── calibration.py    # temperature scaling (val set only)
│   └── export/
│       ├── onnx_export.py    # ONNX export 및 출력 일치 검증
│       └── check_vram.py     # VRAM/latency 측정
├── configs/
│   ├── exp_f_learnable_w.yaml  # 최종 보고 모델 config
│   ├── exp_s_frozen.yaml       # fully frozen baseline
│   └── exp_*.yaml              # 전체 ablation configs
├── scripts/
│   ├── train.py / evaluate_test.py / margin_sweep.py
│   └── eda*.py                 # EDA 스크립트
├── skills/                     # 프로젝트 정책 문서
│   ├── architecture.md         # 확정 아키텍처 및 선택 근거
│   ├── experiments.md          # 실험 A~Q 결과 요약
│   ├── dataset.md              # 데이터 처리 규칙
│   ├── evaluation.md           # 평가 지표 정의
│   └── export.md               # ONNX export 정책
├── tests/                      # pytest 단위/E2E 테스트
├── train.py                    # 학습 엔트리포인트
├── dataset/                    # 기업 데이터 (gitignore)
└── experiments/                # 실험 산출물 (gitignore)
```

---

## 환경 설정

```bash
# Python 3.11+
pip install torch torchvision
pip install timm lion-pytorch
pip install onnxruntime scikit-learn pandas matplotlib
```

**실험 환경:**

| 환경 | GPU | 용도 |
|---|---|---|
| 로컬 A | RTX 4060, VRAM 8GB | DINOv2 ablation 실험 |
| 로컬 B | - | 팀원 보조 실험 |
| 기업 PC | RTX 4090 | 추론 전용 (ONNX Runtime) |

---

## 평가 / 실행

```bash
# 학습
python train.py --config configs/exp_ar_448.yaml

# 외부 testset 평가 — 4타겟 자동(all / crops_0pct / crops_25pct / crops_50pct)
python scripts/evaluate_test.py \
    --config configs/exp_ar_448.yaml \
    --checkpoint experiments/exp_ar_448/checkpoints/best_f1_ep014_0.8198.ckpt \
    --testset_root dataset/testset \
    --output_dir results/exp_ar_448_tta --tta

# multi-scale logit 앙상블 평가 (336 + 448)
python scripts/evaluate_ensemble.py \
    --config configs/exp_ap_f1best.yaml --checkpoint experiments/exp_ap_f1best/checkpoints/best_model.pth \
    --config configs/exp_ar_448.yaml   --checkpoint experiments/exp_ar_448/checkpoints/best_f1_ep014_0.8198.ckpt \
    --exp-name exp_ensemble_336_448

# 실험 비교표 (results/*/summary.csv 집계 → CSV + 막대그래프)
python scripts/compare_experiments.py
```

threshold는 `--threshold` 미지정 시 val에서 Danger-as-Safe ≤ 0.15 제약으로 자동 선택되며(no-leakage), 외부 testset에 고정 적용됩니다. 선택 출처는 `predictions.csv`의 `applied_thr_source`에 기록됩니다.

```bash
# 테스트 실행
python -m pytest -q
```

---

## 결과 (`results/`)

`results/<exp_name>/`에 실험별 평가 산출물을 저장합니다. CSV/summary는 커밋하고, `predictions.csv`(데이터 경로 노출)와 이미지 오버레이 PNG(attention/gallery — 기업 이미지 포함)는 `.gitignore`로 제외합니다.

```
results/
├── experiments_comparison.csv      # 12개 실험 × (all, crops_25pct) 비교
├── experiments_comparison.png      # precision_danger / miss_rate / f1_macro 묶음 막대
└── <exp_name>/
    ├── summary.csv                 # 4타겟 핵심 지표 요약
    └── <target>/                   # all / crops_0pct / crops_25pct / crops_50pct
        ├── metrics.csv             # per-class precision/recall/f1 + bootstrap CI
        ├── threshold_sweep.csv     # danger threshold sweep
        ├── margin_sweep.csv        # 2D decision layer sweep
        ├── risk_coverage_curve.*   # coverage ↔ precision/miss tradeoff
        ├── operating_points.csv    # PrecD≥95/90%, Miss≤10% 운영점
        ├── cost_matrix.csv         # cost-weighted 기대비용
        ├── calibration.json        # temperature scaling(ECE 전/후, val 적합)
        ├── confusion_matrix.csv / *_pr_curve / *_roc_curve / reliability_diagram
        └── predictions.csv         # 샘플별 prob/logit/applied_thr (gitignore)
```

---

## 주요 설계 결정

**왜 threshold 기반 decision인가?**  
argmax는 가장 높은 확률 클래스를 선택하지만, 이 태스크에서는 danger가 낮은 확률로 예측되더라도 안전으로 통과시키는 것이 더 위험합니다. `danger_prob >= 0.30`이면 무조건 danger로 판정해 recall을 확보합니다.

**왜 backbone을 frozen하는가?**  
DINOv2는 대규모 self-supervised 사전학습으로 강력한 visual feature를 보유합니다. fine-tuning이 이 feature를 오히려 훼손할 수 있고, frozen 상태에서도 attention head만으로 충분한 판별력을 보였습니다. VRAM 제약(8GB)에서도 안정적으로 동작합니다.

**왜 LetterBox인가?**  
단순 Resize는 세로로 긴 LPG통을 찌그러뜨려 절단면 형상을 왜곡합니다. LetterBox는 원본 비율을 유지하고 나머지를 패딩으로 채웁니다.

---

## 데이터 보안

```
기업 데이터를 외부(GitHub, Drive, S3 등)에 업로드 금지
데이터 경로를 로그/커밋 메시지에 노출 금지
dataset/, experiments/, checkpoints는 .gitignore로 차단
```
