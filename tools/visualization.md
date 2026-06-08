# tools/visualization.md
# 시각화 정책

---

## 목적

```text
모델이 절단면/끝단/표면 질감에 집중하는지 확인
Danger-as-Safe 실패 사례를 정성 분석
발표 자료용 ablation 근거 생성
```

---

## Attention Map

```bash
python visualize.py --mode attention \
  --checkpoint experiments/exp_d_s_.../checkpoints/best_f1.ckpt \
  --image data/processed/test/danger/sample.jpg \
  --output reports/attention_sample.png
```

출력:

```text
원본 crop
attention map
overlay
예측 확률
decision 결과
```

---

## Ablation 비교 시각화

```bash
python visualize.py --mode ablation_compare \
  --image data/processed/test/danger/sample.jpg \
  --checkpoints \
    b_s=experiments/exp_b_s_.../best_f1.ckpt \
    c_s=experiments/exp_c_s_.../best_f1.ckpt \
    d_s=experiments/exp_d_s_.../best_f1.ckpt \
  --output reports/ablation_attention_compare.png
```

---

## 오분류 갤러리

```bash
python visualize.py --mode error_gallery \
  --exp experiments/exp_d_s_... \
  --error_type danger_as_safe \
  --output reports/danger_as_safe_gallery.html
```

우선순위:

```text
1. 위험 → 안전
2. 안전 → 위험
3. 위험 → 제외
4. 제외 → 안전
```

---

## 발표용 체크리스트

```text
- [ ] Danger-as-Safe 실패 사례
- [ ] 성공적으로 위험을 잡은 사례
- [ ] 안전 고신뢰 통과 사례
- [ ] B/C/D attention 변화
- [ ] calibration 전후 reliability diagram
```

