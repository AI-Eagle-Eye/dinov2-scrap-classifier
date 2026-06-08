# skills/training.md
# 학습 환경 및 정책

---

## 환경

```text
GPU: RTX 4060, VRAM 8GB
학습: 로컬 전용
데이터: 기업 제공 데이터만
외부 로깅/외부 서버 사용 금지
```

---

## 학습 Precision 정책

```text
기본: BF16 mixed precision (torch.autocast, dtype=torch.bfloat16)
  - CUDA일 때만 활성화; CPU는 FP32 fallback
  - BF16은 FP16보다 수치 안정적 (exp/overflow 위험 낮음)
  - RTX 4060/4090 모두 BF16 Tensor Core 지원
  - GradScaler 불필요 → 코드 단순화

적용 위치:
  - Trainer._train_epoch(): forward + loss 구간
  - Trainer._validate(): forward + loss 구간
  - 상수: src/training/trainer.py::_AMP_DTYPE = torch.bfloat16

양자화(INT8/FP8):
  - 전용 연산기 없는 GPU에서 실패 가능성 — 이번 실험 범위 제외
  - ONNX export 이후 별도 검토

VRAM 추정 (RTX 4060, 8GB)
  - DINOv2-S + VPT + MLP head, batch 16, grad_checkpoint=True
    FP32 학습: ~4.5GB (추정)
    BF16 학습: ~2.5GB (추정) — 활성화 메모리 절반
  - DINOv2-B + VPT + MLP head, batch 8, grad_checkpoint=True
    FP32 학습: ~7.5GB (추정, 8GB 경계)
    BF16 학습: ~4.5GB (추정) — 안전 여유 확보
  * backbone은 frozen FP32, head/VPT/PAA만 BF16 autocast 적용
  * 실측 후 ROADMAP 갱신
```

---

## 입력 설정 (EDA 확정)

```yaml
data:
  image_size: 336          # EDA: danger 중앙값 W=285 H=231 → 224 입력 시 다운스케일 발생. 336으로 해상도 확보
  bbox_padding: [0.10, 0.25]  # train: random uniform, val/test: 0.175 고정
  dataset: crops_25pct     # 224미만 비율 29.7%로 3종 중 최소
  category_filter: [2, 3, 4]  # COCO cat_id: cut(2), danger(3), excluded(4)

augmentation:
  normalize:
    mean: [0.484, 0.483, 0.496]   # EDA: B-채널 ImageNet 대비 0.090 차이 → 커스텀 통계 적용
    std:  [0.188, 0.188, 0.194]

paa:
  topk: 36   # 336 입력 기준 576 patch의 6.25% (224 기준 16/256과 동일 비율)
```

resize 방식: LetterBox (aspect ratio 유지, 짧은 쪽 mean color 패딩)
  - 이유: 단순 Resize보다 국소 시각 단서 왜곡이 적음
  - fill=(123, 123, 126) = round(CUSTOM_MEAN * 255)

---

## 클래스 불균형 처리

```python
# EDA: cut 54.2% / danger 30.1% / excluded 15.6% — 불균형 명확
# WeightedRandomSampler를 기본 적용 (make_weighted_sampler 유틸 참고)
from src.data.dataset import make_weighted_sampler

train_ds = HazardDataset(..., is_train=True)
sampler = make_weighted_sampler(train_ds)
loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
# sampler 사용 시 DataLoader shuffle=False
```

---

## VRAM 관리

```yaml
default:
  batch_size: 16
  grad_accumulation: 4
  use_grad_checkpoint: true

dinov2_b_fallback:
  batch_size: 8
  grad_accumulation: 8
```

부족 시 대응:

```text
1. batch_size 감소
2. grad_accumulation 증가
3. num_workers 조정
4. image_size 192는 최후 수단
```

---

## 재현성

```python
def set_seed(seed: int = 42) -> None:
    """실험 재현성을 위한 전역 시드 고정."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 학습 파라미터

Backbone은 frozen으로 둡니다. VPT tokens, PAA 관련 모듈, attention head, classifier만 학습합니다.

```python
trainable = [p for n, p in model.named_parameters() if p.requires_grad]
optimizer = AdamW(trainable, lr=1e-4, weight_decay=1e-4)
```

Scheduler:

```text
cosine_warmup 사용.
PyTorch 기본 CosineAnnealingWarmRestarts에는 warmup_epochs 인자가 없으므로
custom warmup wrapper 또는 별도 scheduler 구현 필요.
```

---

## 체크포인트 정책

```python
CHECKPOINT_METRICS = {
    "val_loss": "min",
    "f1": "max",
    "f2": "max",
    "safe_precision": "max",
    "last": None,
}
```

주의:

```text
ablation 비교 기준은 best_val_loss 또는 best_f1를 우선 사용한다.
best_safe_precision은 threshold와 calibration의 영향을 받으므로
최종 납품 후보 선택에 주로 사용한다.
```

파일명:

```text
best_val_loss_ep023_0.3241.ckpt
best_f1_ep031_0.8912.ckpt
best_f2_ep028_0.8654.ckpt
best_safe_precision_ep035_0.9521.ckpt
last_ep050.ckpt
```

체크포인트 내용:

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "metrics": {
        "val_loss": val_loss,
        "f1": f1,
        "f2": f2,
        "safe_precision": safe_precision,
        "danger_as_safe": danger_as_safe_rate,
    },
    "config": cfg.to_dict(),
    "seed": cfg.training.seed,
}
```

---

## Early Stopping

```yaml
early_stopping:
  monitor: val_loss
  patience: 10
  min_delta: 1.0e-4
  restore_best: true
```

---

## 로깅

```text
experiments/{exp_name}_{timestamp}/
├── config.yaml
├── train_log.csv
├── checkpoints/
└── threshold_results.json
```

CSV:

```csv
epoch,train_loss,val_loss,accuracy,f1,f2,safe_precision,danger_as_safe,lr,timestamp
```

금지:

```text
wandb
MLflow remote tracking
외부 API 기반 로깅
데이터 경로 출력
```

---

## 학습 시간 추정

```text
DINOv2-S + VPT-Deep, 4,200 train samples, batch 16
  1 epoch 약 3~5분 예상
  early stopping 20~30 epoch 예상

DINOv2-B 계열
  batch 8 fallback 고려
  실측 후 ROADMAP 갱신
```

