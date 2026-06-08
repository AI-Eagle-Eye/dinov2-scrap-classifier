"""End-to-end pipeline integration test.

trainer → evaluator → threshold sweep → calibration → ONNX export
전체 흐름이 예외 없이 동작하는지 확인한다.

- 데이터: 랜덤 이미지 30장, 위험/안전/제외 균등 분포 (각 클래스 10장)
- Backbone: torch.hub 없이 동작하는 경량 mock (Conv2d 기반)
- 학습: 2 epoch, CPU, warmup_ratio=0.1

실행:
    pytest tests/test_pipeline_e2e.py -v
"""
from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── 상수 ─────────────────────────────────────────────────────────────────────
N_TRAIN = 20
N_VAL = 10
IMAGE_SIZE = 224
EMBED_DIM = 384
NUM_CLASSES = 3
PATCH_SIZE = 14
N_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 256
BATCH_SIZE = 8
EPOCHS = 2

# sweep_threshold 기본 범위에서 나오는 결과 개수
_SWEEP_N = round((0.90 - 0.30) / 0.05) + 1  # 13


# ── Mock Backbone ─────────────────────────────────────────────────────────────
class _MockBackbone(nn.Module):
    """DINOv2Backbone 인터페이스를 구현하는 경량 mock.

    torch.hub / 인터넷 없이 동작한다. patch embed 역할을 단순 Conv2d로 대체.
    """

    embed_dim: int = EMBED_DIM
    patch_size: int = PATCH_SIZE
    num_heads: int = 6
    num_blocks: int = 12
    model_name: str = "mock_dinov2_vits14"

    def __init__(self) -> None:
        super().__init__()
        self._proj = nn.Conv2d(3, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, bias=False)
        for p in self._proj.parameters():
            p.requires_grad = False
        self._proj.eval()

    def train(self, mode: bool = True) -> _MockBackbone:
        super().train(mode)
        self._proj.eval()  # backbone은 항상 eval 유지
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            out = self._proj(x)              # [B, D, H/14, W/14]
        patches = out.flatten(2).transpose(1, 2)  # [B, N, D]
        cls = patches.mean(1)                      # [B, D]
        return cls, patches


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def _build_model() -> nn.Module:
    """mock backbone으로 HazardModel(Exp B-S) 구성."""
    from src.models.hazard_model import HazardModel, ModelConfig
    from src.models.head import MLPHead

    config = ModelConfig(backbone_name="dinov2_vits14")
    model = HazardModel.__new__(HazardModel)
    nn.Module.__init__(model)
    model.config = config
    model.backbone = _MockBackbone()
    model.vpt = None
    model.head = MLPHead(EMBED_DIM, num_classes=NUM_CLASSES, dropout=config.dropout)
    return model


def _make_loaders(seed: int = 42) -> tuple[DataLoader, DataLoader]:
    """위험/안전/제외 균등 분포 30장: train 20 / val 10."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    def _ds(n: int) -> TensorDataset:
        images = torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE)
        labels = torch.arange(n) % NUM_CLASSES  # 0,1,2,0,1,2,...
        return TensorDataset(images, labels)

    return (
        DataLoader(_ds(N_TRAIN), batch_size=BATCH_SIZE, shuffle=True, generator=gen),
        DataLoader(_ds(N_VAL), batch_size=BATCH_SIZE, shuffle=False),
    )


def _collect_logits(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor]:
    """val set의 logits / labels 수집.

    torch.no_grad() 사용: TemperatureScaler의 LBFGS backward를 위해
    inference_mode 대신 일반 텐서(requires_grad=False)를 반환한다.
    """
    model.eval()
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            logits_list.append(model(images))
            labels_list.append(labels)
    return torch.cat(logits_list), torch.cat(labels_list)


# ── 공유 Fixture ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def e2e(tmp_path_factory: pytest.TempPathFactory):
    """전체 파이프라인을 한 번 실행하고 결과를 모든 테스트 클래스가 공유.

    yield 방식: 모듈 내 모든 테스트가 완료되면 tmp 디렉토리를 명시적으로 삭제.
    pytest 자체도 tmp_path_factory 디렉토리를 세션 종료 후 정리하지만,
    teardown을 명시해 CI 환경에서 디스크 낭비를 방지한다.
    """
    tmp = tmp_path_factory.mktemp("e2e_pipeline")
    try:
        model = _build_model()
        train_loader, val_loader = _make_loaders()

        # ── 1. Trainer ────────────────────────────────────────────────────────
        from src.training.checkpoint import CheckpointManager
        from src.training.trainer import Trainer

        ckpt_manager = CheckpointManager(tmp / "checkpoints")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            ckpt_manager=ckpt_manager,
            log_dir=tmp / "logs",
            config_dict={"experiment": "e2e_test", "epochs": EPOCHS},
            device="cpu",
            lr=1e-3,
            weight_decay=1e-4,
            epochs=EPOCHS,
            warmup_ratio=0.1,
            early_stopping_patience=EPOCHS + 1,
            label_smoothing=0.1,
            seed=42,
        )
        best_metrics = trainer.fit()

        # ── 2. Evaluator ──────────────────────────────────────────────────────
        logits, labels = _collect_logits(model, val_loader)
        probs_np: np.ndarray = torch.softmax(logits, dim=-1).numpy()
        labels_np: np.ndarray = labels.numpy().astype(int)
        preds: list[int] = logits.argmax(1).tolist()

        from src.evaluation.evaluator import compute_metrics
        eval_metrics = compute_metrics(labels.tolist(), preds)

        # ── 3. Threshold sweep ────────────────────────────────────────────────
        from src.evaluation.threshold import select_best_threshold, sweep_threshold
        sweep_results = sweep_threshold(probs_np, labels_np)
        best_thr = select_best_threshold(sweep_results)

        # ── 4. Calibration ────────────────────────────────────────────────────
        from src.evaluation.calibration import TemperatureScaler
        scaler = TemperatureScaler()
        temperature = scaler.fit(logits.clone(), labels)
        cal_probs = scaler.calibrate(logits)

        # ── 5. ONNX export ────────────────────────────────────────────────────
        onnx_path = tmp / "model.onnx"
        from src.export.onnx_export import export_onnx
        export_onnx(model, onnx_path, image_size=IMAGE_SIZE)

        yield {
            "model": model,
            "best_metrics": best_metrics,
            "eval_metrics": eval_metrics,
            "logits": logits,
            "labels": labels,
            "probs_np": probs_np,
            "labels_np": labels_np,
            "sweep_results": sweep_results,
            "best_thr": best_thr,
            "temperature": temperature,
            "cal_probs": cal_probs,
            "onnx_path": onnx_path,
            "log_dir": tmp / "logs",
            "ckpt_dir": tmp / "checkpoints",
        }
    finally:
        # 모듈 내 모든 테스트 완료 후 임시 파일 삭제
        shutil.rmtree(tmp, ignore_errors=True)


# ── 1. Trainer 검증 ──────────────────────────────────────────────────────────
class TestTrainerE2E:
    _REQUIRED_KEYS = frozenset({
        "loss", "f1_macro", "f2_macro",
        "safe_precision", "danger_as_safe_rate", "accuracy",
    })

    def test_fit_returns_dict(self, e2e: dict) -> None:
        assert isinstance(e2e["best_metrics"], dict)

    def test_metric_keys_complete(self, e2e: dict) -> None:
        missing = self._REQUIRED_KEYS - e2e["best_metrics"].keys()
        assert not missing, f"누락된 metric 키: {missing}"

    def test_loss_is_positive_finite(self, e2e: dict) -> None:
        loss = e2e["best_metrics"]["loss"]
        assert loss > 0 and math.isfinite(loss)

    def test_log_csv_row_count(self, e2e: dict) -> None:
        """CSV: header 1행 + EPOCHS행."""
        csv_path = e2e["log_dir"] / "train_log.csv"
        assert csv_path.exists(), "train_log.csv가 생성되지 않음"
        with csv_path.open() as f:
            rows = list(csv.reader(f))
        assert len(rows) == EPOCHS + 1, f"CSV 행 수 오류: expected {EPOCHS + 1}, got {len(rows)}"

    def test_log_csv_header(self, e2e: dict) -> None:
        csv_path = e2e["log_dir"] / "train_log.csv"
        with csv_path.open() as f:
            header = next(csv.reader(f))
        assert "epoch" in header
        assert "val_loss" in header
        assert "danger_as_safe" in header

    def test_checkpoint_last_exists(self, e2e: dict) -> None:
        files = list(e2e["ckpt_dir"].glob("last_ep*.ckpt"))
        assert len(files) >= 1, "last 체크포인트 없음"

    def test_checkpoint_best_val_loss_exists(self, e2e: dict) -> None:
        files = list(e2e["ckpt_dir"].glob("best_val_loss_ep*.ckpt"))
        assert len(files) >= 1, "best_val_loss 체크포인트 없음"

    def test_checkpoint_is_loadable(self, e2e: dict) -> None:
        from src.training.checkpoint import CheckpointManager
        files = list(e2e["ckpt_dir"].glob("last_ep*.ckpt"))
        state = CheckpointManager.load(files[0])
        for key in ("epoch", "model_state_dict", "optimizer_state_dict", "metrics"):
            assert key in state, f"체크포인트에 {key!r} 없음"

    def test_checkpoint_epoch_matches(self, e2e: dict) -> None:
        from src.training.checkpoint import CheckpointManager
        files = sorted(e2e["ckpt_dir"].glob("last_ep*.ckpt"))
        state = CheckpointManager.load(files[-1])
        assert state["epoch"] == EPOCHS

    def test_head_params_have_grad_after_training(self, e2e: dict) -> None:
        """head backward가 정상 동작함을 확인.

        fit() 후 zero_grad()로 grad가 None이 되므로 fresh forward+backward로 재검증.
        """
        model = e2e["model"]
        model.train()
        dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        logits = model(dummy)
        logits.mean().backward()
        for name, param in model.head.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"head.{name}에 gradient 없음"
        model.eval()

    def test_backbone_params_have_no_grad(self, e2e: dict) -> None:
        """frozen backbone은 grad가 없어야 한다."""
        for name, param in e2e["model"].backbone._proj.named_parameters():
            assert param.grad is None, f"backbone.{name}에 gradient가 쌓임"


# ── 2. Evaluator 검증 ────────────────────────────────────────────────────────
class TestEvaluatorE2E:
    _METRIC_KEYS = frozenset({
        "f1_danger", "f1_cut", "f1_excluded",
        "f1_macro", "f2_macro",
        "safe_precision", "danger_as_safe_rate", "accuracy",
    })

    def test_all_metric_keys_present(self, e2e: dict) -> None:
        missing = self._METRIC_KEYS - e2e["eval_metrics"].keys()
        assert not missing, f"누락된 evaluator 키: {missing}"

    def test_accuracy_in_unit_interval(self, e2e: dict) -> None:
        assert 0.0 <= e2e["eval_metrics"]["accuracy"] <= 1.0

    def test_f1_macro_in_unit_interval(self, e2e: dict) -> None:
        assert 0.0 <= e2e["eval_metrics"]["f1_macro"] <= 1.0

    def test_f2_macro_in_unit_interval(self, e2e: dict) -> None:
        assert 0.0 <= e2e["eval_metrics"]["f2_macro"] <= 1.0

    def test_danger_as_safe_rate_in_unit_interval(self, e2e: dict) -> None:
        assert 0.0 <= e2e["eval_metrics"]["danger_as_safe_rate"] <= 1.0

    def test_safe_precision_in_unit_interval(self, e2e: dict) -> None:
        assert 0.0 <= e2e["eval_metrics"]["safe_precision"] <= 1.0

    def test_per_class_f1_in_unit_interval(self, e2e: dict) -> None:
        for key in ("f1_danger", "f1_cut", "f1_excluded"):
            assert 0.0 <= e2e["eval_metrics"][key] <= 1.0, f"{key} 범위 오류"


# ── 3. Threshold Sweep 검증 ──────────────────────────────────────────────────
class TestThresholdSweepE2E:
    _RESULT_KEYS = frozenset({"safe_thr", "danger_as_safe_rate", "safe_precision",
                               "f1_macro", "coverage"})

    def test_sweep_returns_correct_count(self, e2e: dict) -> None:
        assert len(e2e["sweep_results"]) == _SWEEP_N, \
            f"sweep 결과 개수 오류: expected {_SWEEP_N}, got {len(e2e['sweep_results'])}"

    def test_sweep_result_has_required_keys(self, e2e: dict) -> None:
        for row in e2e["sweep_results"]:
            missing = self._RESULT_KEYS - row.keys()
            assert not missing, f"sweep 결과에 키 누락: {missing}"

    def test_sweep_thresholds_in_range(self, e2e: dict) -> None:
        for row in e2e["sweep_results"]:
            assert 0.30 - 1e-6 <= row["safe_thr"] <= 0.90 + 1e-6

    def test_sweep_thresholds_monotonically_increasing(self, e2e: dict) -> None:
        thresholds = [r["safe_thr"] for r in e2e["sweep_results"]]
        assert all(a < b for a, b in zip(thresholds, thresholds[1:]))

    def test_sweep_coverage_in_unit_interval(self, e2e: dict) -> None:
        for row in e2e["sweep_results"]:
            assert 0.0 <= row["coverage"] <= 1.0

    def test_best_threshold_has_required_keys(self, e2e: dict) -> None:
        missing = self._RESULT_KEYS - e2e["best_thr"].keys()
        assert not missing

    def test_best_threshold_in_sweep_range(self, e2e: dict) -> None:
        thr = e2e["best_thr"]["safe_thr"]
        assert 0.30 - 1e-6 <= thr <= 0.90 + 1e-6

    def test_best_threshold_is_one_of_sweep_results(self, e2e: dict) -> None:
        sweep_thrs = {round(r["safe_thr"], 6) for r in e2e["sweep_results"]}
        best = round(e2e["best_thr"]["safe_thr"], 6)
        assert best in sweep_thrs, "best_thr가 sweep 결과 목록에 없음"


# ── 4. Calibration 검증 ──────────────────────────────────────────────────────
class TestCalibrationE2E:
    def test_temperature_positive(self, e2e: dict) -> None:
        assert e2e["temperature"] > 0.0

    def test_temperature_finite(self, e2e: dict) -> None:
        assert math.isfinite(e2e["temperature"])

    def test_calibrated_probs_shape(self, e2e: dict) -> None:
        assert e2e["cal_probs"].shape == (N_VAL, NUM_CLASSES)

    def test_calibrated_probs_sum_to_one(self, e2e: dict) -> None:
        # calibrate()는 softmax를 적용하므로 각 행의 합이 1이어야 한다
        row_sums = e2e["cal_probs"].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(N_VAL), atol=1e-5), \
            f"calibrated probs row sum 최대 오차: {(row_sums - 1).abs().max().item():.2e}"

    def test_calibrated_probs_nonnegative(self, e2e: dict) -> None:
        assert (e2e["cal_probs"] >= 0).all()

    def test_calibrated_probs_at_most_one(self, e2e: dict) -> None:
        assert (e2e["cal_probs"] <= 1.0 + 1e-6).all()

    def test_calibrated_argmax_same_as_raw(self, e2e: dict) -> None:
        """temperature scaling은 클래스 순서를 바꾸지 않는다 (단조 변환)."""
        raw_pred = e2e["logits"].argmax(1)
        cal_pred = e2e["cal_probs"].argmax(1)
        assert torch.equal(raw_pred, cal_pred), \
            "calibration 후 argmax가 달라짐 (temperature가 음수인 경우 발생)"


# ── 5. ONNX Export 검증 ──────────────────────────────────────────────────────
class TestONNXExportE2E:
    def test_onnx_file_created(self, e2e: dict) -> None:
        assert e2e["onnx_path"].exists()

    def test_onnx_file_nonempty(self, e2e: dict) -> None:
        size = e2e["onnx_path"].stat().st_size
        assert size > 0, f"ONNX 파일 크기 0 bytes"

    def test_onnx_output_matches_pytorch(self, e2e: dict) -> None:
        pytest.importorskip("onnxruntime", reason="onnxruntime 미설치 — 건너뜀")
        from src.export.onnx_export import verify_onnx

        torch.manual_seed(77)
        test_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        # FP16 export 기준 atol=1e-3 (workflow.md §4 참고)
        is_match = verify_onnx(e2e["onnx_path"], e2e["model"], test_input, atol=1e-3)
        assert is_match, "ONNX Runtime 출력이 PyTorch 출력과 불일치"

    def test_onnx_output_batch_shape(self, e2e: dict) -> None:
        ort = pytest.importorskip("onnxruntime", reason="onnxruntime 미설치 — 건너뜀")
        import onnxruntime as _ort

        torch.manual_seed(88)
        sess = _ort.InferenceSession(
            str(e2e["onnx_path"]), providers=["CPUExecutionProvider"]
        )
        # ONNX 모델의 입력 dtype을 자동 감지해 맞춤 (FP16 export 시 float16 필요)
        input_type = sess.get_inputs()[0].type
        np_dtype = np.float16 if "float16" in input_type else np.float32
        test_input = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE).numpy().astype(np_dtype)
        out = sess.run(None, {"image": test_input})[0]
        assert out.shape == (3, NUM_CLASSES), f"ONNX 출력 shape 오류: {out.shape}"

    def test_onnx_dynamic_batch_size(self, e2e: dict) -> None:
        """dynamic_axes 설정으로 배치 크기 1과 4 모두 처리 가능해야 한다."""
        ort = pytest.importorskip("onnxruntime", reason="onnxruntime 미설치 — 건너뜀")
        import onnxruntime as _ort

        sess = _ort.InferenceSession(
            str(e2e["onnx_path"]), providers=["CPUExecutionProvider"]
        )
        input_type = sess.get_inputs()[0].type
        np_dtype = np.float16 if "float16" in input_type else np.float32
        for batch in (1, 4):
            x = torch.randn(batch, 3, IMAGE_SIZE, IMAGE_SIZE).numpy().astype(np_dtype)
            out = sess.run(None, {"image": x})[0]
            assert out.shape == (batch, NUM_CLASSES), \
                f"batch={batch}일 때 ONNX 출력 shape 오류: {out.shape}"
