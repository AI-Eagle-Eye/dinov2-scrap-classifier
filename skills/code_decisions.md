# skills/code_decisions.md
# 클린코드 점검 결정 기록 (2026-06-05)

> src/ + train.py + scripts/ 정적 분석 후 수행한 수정과 그 근거.
> 기능/아키텍처 변경 없음 — 클린코드/버그 예방 목적.

---

## 수정 1: tune_threshold._build_model 버그 수정

```text
문제:
  scripts/tune_threshold.py의 _build_model이 scripts/evaluate_test.py와 달리
  - VPTConfig의 insert_from_layer 누락
  - ModelConfig의 class_aware_init_weights 누락
  이로 인해 exp_q_partial_vpt(insert_from_layer 사용) /
  exp_p_class_aware(class_aware_init_weights 사용) config로 실행 시
  잘못된 모델 구조 생성 → state_dict 로드 불일치 발생.

수정:
  evaluate_test 기준으로 모델 빌드 로직을 동기화 (수정 2의 공용화로 자동 해결).
  기능 변경 없이 evaluate_test와 동일한 ModelConfig를 생성하도록 맞춤.
```

## 수정 2: scripts/_eval_common.py 추출

```text
이유:
  evaluate_test.py와 tune_threshold.py가 5개 헬퍼를 중복 보유하고 있었고,
  한쪽만 수정되면(수정 1과 같은) 동기화 누락 버그가 재발할 수 있음.

조치:
  detect_device / load_config / build_model / collect_probs / apply_threshold
  를 scripts/_eval_common.py로 단일화. 두 스크립트는 이를 import해 사용.
  → 모델 빌드 로직이 한 곳에만 존재하므로 동기화 버그 구조적 예방.
```

## 수정 3: 미사용 import / 데드 변수·상수 제거

```text
대상: eda.py, eda_resolution_scatter.py, eda_small_objects.py,
      eda_resolution_detail.py, smoke_test_korean_font.py
내용: 미사용 import, 미참조 상수(CAMERA_MARKERS 등), 미사용 지역변수
      (blur_norm, dims_global)만 제거. 로직 변경 없음.
보류: scripts/label_cut.py의 `app` 지역변수는 ruff F841로 잡히나,
      tkinter 객체 GC 방지용 참조라 의도적으로 유지.
```

## 수정 4: DANGER_AS_SAFE_LIMIT 상수화

```text
이유:
  src/training/trainer.py에서 danger_as_safe 한계값 0.15가
  EarlyStopping.update / _update_best_model / verbose 출력 등 3곳에 하드코딩.
  세 곳이 같은 의미의 임계값이므로 분리 시 불일치 위험.

조치:
  모듈 상단에 DANGER_AS_SAFE_LIMIT = 0.15 정의 후 3곳 모두 참조하도록 통일.
```

---

## 보류한 항목 (의도적 미수정)

```text
1. EDA 스크립트 중복 로직
   - save_eda_json: eda_color / eda_resolution_detail / eda_small_objects 3곳 동일.
   - crop 계산: eda_resolution_detail.crop_size ↔ eda_resolution_scatter.compute_crop_size.
   - CLASS_COLORS / CAT_ID_TO_NAME / PADDINGS 상수 4개 스크립트 반복.
   보류 이유:
     EDA 스크립트는 각자 독립 실행되는 일회성 분석 도구라
     공용 모듈 도입 시 결합도만 높이고 실행 편의가 떨어짐.
     리팩토링 이득 < 변경 위험으로 판단해 이번 범위에서 제외.

2. DataLoader generator / worker_init_fn 미지정
   - num_workers>0 + shuffle 환경에서 완전 결정성을 원하면 보강 여지.
   - 현재 set_seed(random/numpy/torch/cuda)로 대부분 재현 가능하고,
     transforms가 numpy RNG를 쓰지 않아 실질 영향 적어 보류.

3. src/export/*, src/evaluation/threshold.py sweep/decide, calibration.py
   - 현재 scope 내 호출 스크립트가 없으나 public 라이브러리 API로 의도된 것.
   - export/calibration 단계 스크립트 구현 시 사용 예정이므로 삭제하지 않음.
```

관련: [skills/architecture.md], [skills/experiments.md]
