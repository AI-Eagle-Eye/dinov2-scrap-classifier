# skills/architecture.md
# 확정 아키텍처 및 선택 근거

> 이 파일은 실험 A~Q를 거쳐 확정된 최종 모델 구성과 그 근거를 기록한다.
> 실험 결과 요약은 [skills/experiments.md], 데이터 정책은 [skills/dataset.md] 참조.

---

## 확정 아키텍처

```text
DINOv2 ViT-B/14 (frozen)
  └─ Attention head
       - nn.MultiheadAttention (embed_dim=768, num_heads=8, batch_first=True)
       - patch token self-attention → mean pool → LayerNorm
       - Linear(768 → 3)
```

- backbone: `dinov2_vitb14` (embed_dim=768, patch_size=14, num_heads=12), 전 구간 frozen + eval 유지
- head: `AttentionHead` (`src/models/head.py`) — patch token에 self-attention 적용 후 mean pool
- 입력: 336×336 LetterBox (patch_size=14 → 24×24=576 patch token)
- 출력: 3-class logits (cut=0 / danger=1 / excluded=2)
- 결정: argmax가 아니라 threshold 기반 decision (threshold=0.30)

---

## 학습 하이퍼파라미터 및 근거

### Learning rate: backbone_lr=1e-5 / head_lr=5e-5
```text
- backbone은 frozen이 기본이나, 미세 적응 시 head보다 1/5 수준의 낮은 lr로
  사전학습 표현 붕괴를 방지.
- head는 무작위 초기화 상태에서 학습하므로 더 큰 lr(5e-5)로 빠르게 수렴.
- backbone_lr=0이면 backbone을 완전 frozen하고 optimizer에서 제외 (기본 동작).
```

### Loss: FocalLoss(γ=2.0) + (옵션) LearnableClassWeight
```text
- 클래스 불균형(cut 9713 / danger 5393 / excluded 2797)에서
  easy-negative 비중을 낮추기 위해 Focal Loss 채택.
- γ=2.0: 표준값. γ=3.0 ablation(exp_a_mlp_gamma3, exp_b_att_gamma3)에서
  뚜렷한 개선 없어 2.0 유지.
- label_smoothing=0.1로 과신뢰 억제.
- focal weight는 softmax(logits.detach())로 계산 — gradient는 CE 경로로만 흐름.
```

### Optimizer: Lion(lr=5e-5, weight_decay=1e-2)
```text
- Lion은 AdamW 대비 메모리 절약(모멘텀 1개 상태만 유지) → VRAM 8GB(로컬 A) 환경에 유리.
- weight_decay=1e-2로 frozen backbone 위 head 정규화.
- backbone/head 분리 param group으로 서로 다른 lr 적용.
```

### Scheduler: LinearWarmup(10%) + CosineAnnealingLR
```text
- warmup_ratio=0.1: 초기 head 무작위 가중치에서의 불안정한 큰 step 방지.
- 이후 CosineAnnealingLR(eta_min=1e-7)로 부드럽게 감쇠.
- SequentialLR로 warmup→cosine 연결 (milestone=warmup_epochs).
```

### Decision threshold: 0.30
```text
- danger_prob >= 0.30이면 danger로 판정, 그 외 argmax.
- val set sweep(0.10~0.90)에서 danger_as_safe < 10% 제약을 만족하면서
  danger_precision을 최대화하는 지점으로 선택.
- 핵심 지표(Danger-as-Safe Rate 최소화)에 맞춰 낮은 threshold로 danger 민감도 확보.
```

### Early stopping 조건
```text
- danger_as_safe_rate < DANGER_AS_SAFE_LIMIT(0.15) 조건 하에서만
  danger_precision(높을수록 좋음)의 개선 여부를 카운트.
- das >= 0.15면 카운트 증가 없이 skip → 안전 제약을 만족하는 구간에서만 최적화.
- best 모델도 동일 조건(das<0.15 하 danger_precision 최대, 동률 시 accuracy)으로 저장.
```

---

## 시도했으나 효과 없었던 것 (제외 이유)

| 항목 | 제외 이유 |
|---|---|
| VPT 전체 (insert_from_layer=0) | frozen backbone 대비 danger_as_safe 개선 미미, 학습 파라미터·VRAM만 증가 |
| VPT Partial (insert_from_layer=k) | 중간 레이어부터 prompt 삽입해도 attention head 단독 대비 이득 없음 |
| SupCon Loss | contrastive 표현 정렬이 최종 분류 지표로 이어지지 않음, 튜닝 비용만 큼 |
| CE + DangerRecallLoss | danger_recall 직접 최대화가 precision을 과하게 희생 |
| LearnableClassWeight | 학습된 가중치가 inverse-freq 상수 가중치와 거의 동일하게 수렴 → 이득 없음 |
| ClassAwareHead | class-aware token attention이 단순 attention head 대비 개선 없음 |
| EVA-02 (eva02_base_patch14_448) | DINOv2 대비 분류 지표 우위 없음, 입력 448로 비용만 증가 |
| DINOv2 registers | register token 변형이 본 태스크에서 유의미한 차이 없음 |
| WeightedRandomSampler | Focal Loss와 동시 적용 시 소수 클래스 이중 보정 → 과보정으로 불안정 |

---

## 관련 코드 상수

```text
DANGER_AS_SAFE_LIMIT = 0.15   # src/training/trainer.py (best 갱신/early stopping 공통 기준)
CUSTOM_MEAN/STD               # src/data/transforms.py (데이터 실측 정규화값)
```

관련: [skills/experiments.md], [skills/code_decisions.md]
