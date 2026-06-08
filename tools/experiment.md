# tools/experiment.md
# 실험 실행 방법

---

## 필수 실험

```bash
# Exp B-S
python train.py --config configs/exp_b_s.yaml

# Exp B-B
python train.py --config configs/exp_b_b.yaml

# Exp C-S
python train.py --config configs/exp_c_s.yaml

# Exp C-B
python train.py --config configs/exp_c_b.yaml

# Exp D-S, Primary MVP
python train.py --config configs/exp_d_s.yaml
```

권장:

```bash
# Exp D-B, performance upper bound
python train.py --config configs/exp_d_b.yaml
```

최종:

```bash
python calibrate.py --exp experiments/exp_d_s_...
python evaluate.py --exp experiments/exp_d_s_... --sweep
python export.py --exp experiments/exp_d_s_... --checkpoint best_safe_precision
```

---

## Optional

```bash
# 336 입력 비교, D-S만
python train.py --config configs/exp_d_s.yaml \
  --override data.image_size=336 paa.topk=36

# MoE head, D-S만
python train.py --config configs/exp_d_s_moe.yaml
```

---

## Config 예시

```yaml
experiment:
  name: exp_d_s
  description: DINOv2-S + VPT + PAA + texture + attention head
  seed: 42

data:
  train_dir: data/processed/train
  val_dir: data/processed/val
  test_dir: data/processed/test
  image_size: 224
  bbox_padding: 0.15
  batch_size: 16
  num_workers: 4

model:
  backbone: dinov2_vits14
  frozen: true
  vpt:
    enabled: true
    depth: deep
    num_tokens: 10
    dropout: 0.1
  paa:
    enabled: true
    topk: 16
  texture:
    enabled: true
    statistics: [mean, std]
  head:
    type: attention_pooling
    hidden_dim: 256
    num_classes: 3
    dropout: 0.1

training:
  epochs: 50
  optimizer: adamw
  lr: 1.0e-4
  weight_decay: 1.0e-4
  scheduler: cosine_warmup
  warmup_epochs: 5
  early_stopping_patience: 10
  label_smoothing: 0.1
  grad_accumulation: 4
  use_grad_checkpoint: true
  mixup_alpha: 0.0
  seed: 42

threshold:
  safe_threshold: 0.70
  danger_threshold: 0.60
  margin: 0.20
  sweep:
    target: safe_threshold
    start: 0.30
    end: 0.90
    step: 0.05
```

---

## Ablation 비교

```text
Backbone scale:
  B-S vs B-B

VPT:
  B-S vs C-S
  B-B vs C-B

PAA + texture:
  C-S vs D-S
  C-B vs D-B

Decision:
  D-S/D-B vs E
```

판단 기준:

```text
Danger-as-Safe Rate
Safe Precision
Danger Miss Rate
per-class F1
latency / VRAM
```

