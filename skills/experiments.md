# skills/experiments.md
# 실험 결과 요약 (A~Q)

> 확정 아키텍처와 선택 근거는 [skills/architecture.md] 참조.

---

## 효과 있었던 것

```text
1. Attention head > MLP head
   - patch token self-attention + mean pool이 CLS-only MLP 대비 danger 판별 우수.

2. FocalLoss γ=2.0
   - 클래스 불균형에서 easy sample 억제. γ=3.0은 추가 이득 없음.

3. lr 분리: head 5e-5 + backbone 1e-5
   - head 빠른 수렴 + backbone 미세 적응의 균형.

4. threshold 튜닝 (0.30)
   - argmax 대신 danger_prob>=0.30. danger_as_safe를 핵심 제약으로 최적화.
```

## 효과 없었던 것 (architecture.md와 동일 목록)

```text
VPT 전체 / VPT Partial / SupCon Loss / CE+DangerRecallLoss /
LearnableClassWeight(상수와 거의 동일하게 수렴) / ClassAwareHead /
EVA-02 / DINOv2 registers / WeightedRandomSampler(Focal과 이중 보정)
```

각 항목 제외 이유는 [skills/architecture.md]의 표 참조.

---

## 최종 모델 — Exp F (learnable_w)

test set, threshold=0.30:

| 지표 | 값 |
|---|---|
| accuracy | 63.59% |
| danger_as_safe | 6.62% |
| danger_precision | 51.92% |
| danger_recall | 82.52% |
| cut_precision | 89.54% |
| F1 macro | 62.91% |

```text
해석:
- danger_as_safe 6.62% — 핵심 지표(위험을 안전으로 오판) 목표 달성 수준.
- danger_recall 82.52% — 위험 샘플의 대부분을 포착.
- cut_precision 89.54% — 안전 판정의 신뢰도 확보(사람 보조 시스템에서 중요).
- danger_precision 51.92% — danger 판정의 절반은 재검토 대상(보수적 운영 의도와 부합).
```

---

## 실험 A~Q 흐름 요약

```text
A (exp_a_mlp)          : MLP head baseline + FocalLoss γ=2.0
A' (exp_a_mlp_gamma3)  : MLP head, γ=3.0 ablation → 개선 없음
B (exp_b_att)          : Attention head 도입 → MLP 대비 개선 확인
B' (exp_b_att_gamma3)  : Attention head, γ=3.0 ablation → 개선 없음
C (exp_c_lr)           : learning rate 탐색 (head/backbone 분리 lr)
D (exp_d_vpt_frozen)   : VPT (backbone frozen) → 이득 없음
E (exp_e_vpt_unfreeze) : VPT + backbone 미세 unfreeze → 이득 없음
F (exp_f_learnable_w)  : LearnableClassWeight → 최종 보고 모델
                         (가중치는 상수와 거의 동일하게 수렴)
G (exp_g_supcon)       : SupCon Loss → 분류 지표로 이어지지 않음
H (exp_h_supcon_lw)    : SupCon + learnable weight → 개선 없음
I (exp_i_ce)           : 순수 CE baseline
J (exp_j_ce_dr05)      : CE + DangerRecall(λ=0.5) → precision 희생
K (exp_k_ce_dr10)      : CE + DangerRecall(λ=1.0) → precision 추가 희생
L (exp_l_focal_dr05)   : Focal + DangerRecall(λ=0.5)
M (exp_m_focal_dr10)   : Focal + DangerRecall(λ=1.0)
N (exp_n_eva02)        : EVA-02 backbone → DINOv2 대비 우위 없음
P (exp_p_class_aware)  : ClassAwareHead → 단순 attention 대비 개선 없음
Q (exp_q_partial_vpt)  : Partial VPT (insert_from_layer) → 이득 없음
```

> 위 흐름은 config 파일명(configs/exp_*.yaml)과 코드에서 확인 가능한 범위로 정리한 것이며,
> 각 실험의 정확한 수치는 experiments/ 산출물(gitignore) 참조.

관련: [skills/architecture.md], [skills/code_decisions.md]
