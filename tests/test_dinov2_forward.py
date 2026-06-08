"""DINOv2-S/14 forward pass 및 신규 head/model API 검증 테스트.

실행:
    pytest tests/test_dinov2_forward.py -v
    pytest tests/test_dinov2_forward.py -v -m "not slow"  # 모델 로드 제외
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ── 상수 ────────────────────────────────────────────────────────────────────
IMAGE_SIZE = 224
PATCH_SIZE = 14
BATCH_SIZE = 2
DIM_S = 384
DIM_B = 768
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 16×16 = 256
NUM_CLASSES = 3


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def backbone_s():
    """DINOv2-S/14 백본을 모듈 단위로 한 번만 로드."""
    try:
        from src.models.backbone import DINOv2Backbone
        return DINOv2Backbone("dinov2_vits14")
    except Exception as e:
        pytest.skip(f"DINOv2-S/14 로드 실패 (네트워크/캐시 확인): {e}")


@pytest.fixture(scope="module")
def dummy_batch() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture(scope="module")
def model_mlp(backbone_s):
    """backbone_s + MLPHead 조합의 HazardModel."""
    from src.models.hazard_model import HazardModel, ModelConfig, VPTConfig
    from src.models.head import MLPHead

    config = ModelConfig(
        backbone_name="dinov2_vits14",
        head_type="mlp",
        vpt=VPTConfig(enabled=False),
    )
    model = HazardModel.__new__(HazardModel)
    nn.Module.__init__(model)
    model.config = config
    model.backbone = backbone_s
    model.vpt = None
    model.head = MLPHead(DIM_S, num_classes=NUM_CLASSES)
    return model.eval()


# ── DINOv2Backbone 테스트 ─────────────────────────────────────────────────────

class TestDINOv2BackboneShapes:
    def test_cls_token_shape(self, backbone_s: object, dummy_batch: torch.Tensor) -> None:
        cls, _ = backbone_s(dummy_batch)
        assert cls.shape == (BATCH_SIZE, DIM_S), f"CLS shape 오류: {cls.shape}"

    def test_patch_tokens_shape(self, backbone_s: object, dummy_batch: torch.Tensor) -> None:
        _, patches = backbone_s(dummy_batch)
        assert patches.shape == (BATCH_SIZE, NUM_PATCHES, DIM_S), \
            f"patch shape 오류: {patches.shape}"

    def test_num_patches_matches_formula(self, backbone_s: object, dummy_batch: torch.Tensor) -> None:
        _, patches = backbone_s(dummy_batch)
        expected = (IMAGE_SIZE // PATCH_SIZE) ** 2
        assert patches.shape[1] == expected

    def test_embed_dim_property(self, backbone_s: object) -> None:
        assert backbone_s.embed_dim == DIM_S

    def test_patch_size_property(self, backbone_s: object) -> None:
        assert backbone_s.patch_size == PATCH_SIZE


class TestDINOv2BackboneFrozen:
    def test_all_params_frozen(self, backbone_s: object) -> None:
        for name, param in backbone_s._dino.named_parameters():
            assert not param.requires_grad, f"파라미터 {name!r}가 frozen 되지 않음"

    def test_dino_stays_eval_in_train_mode(self, backbone_s: object) -> None:
        """HazardModel.train() 호출 후에도 DINOv2는 eval 모드를 유지해야 한다."""
        backbone_s.train()
        assert not backbone_s._dino.training, "DINOv2는 학습 중에도 eval 모드를 유지해야 함"
        backbone_s.eval()

    def test_no_grad_accumulation_on_backbone(self, backbone_s: object, dummy_batch: torch.Tensor) -> None:
        """backbone forward 후 파라미터 grad가 None이어야 한다."""
        backbone_s(dummy_batch)
        for param in backbone_s._dino.parameters():
            assert param.grad is None, "frozen backbone에 grad가 쌓임"


class TestDINOv2BackboneDeterminism:
    def test_deterministic_output(self, backbone_s: object, dummy_batch: torch.Tensor) -> None:
        cls1, patch1 = backbone_s(dummy_batch)
        cls2, patch2 = backbone_s(dummy_batch)
        assert torch.allclose(cls1, cls2), "동일 입력에서 CLS 출력이 달라짐"
        assert torch.allclose(patch1, patch2), "동일 입력에서 patch 출력이 달라짐"

    def test_different_inputs_give_different_outputs(self, backbone_s: object) -> None:
        torch.manual_seed(1)
        x1 = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        x2 = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        cls1, _ = backbone_s(x1)
        cls2, _ = backbone_s(x2)
        assert not torch.allclose(cls1, cls2), "다른 입력이 동일한 CLS를 출력함"

    def test_batch_consistency(self, backbone_s: object) -> None:
        """배치 처리 결과가 개별 처리 결과와 일치해야 한다.

        atol=1e-4: 12블록 transformer에서 BLAS 병렬 처리 시 부동소수점 누적 오차가
        발생할 수 있다.
        """
        torch.manual_seed(2)
        x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        cls_batch, patch_batch = backbone_s(x)
        cls_single, patch_single = backbone_s(x[:1])
        assert torch.allclose(cls_batch[:1], cls_single, atol=1e-4), "배치/단일 처리 CLS 불일치"
        assert torch.allclose(patch_batch[:1], patch_single, atol=1e-4), "배치/단일 처리 patch 불일치"


class TestDINOv2BackboneDevice:
    def test_output_on_cpu(self, backbone_s: object, dummy_batch: torch.Tensor) -> None:
        cls, patches = backbone_s(dummy_batch)
        assert cls.device.type == "cpu"
        assert patches.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 없음")
    def test_output_on_cuda(self, backbone_s: object) -> None:
        backbone_s.cuda()
        dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device="cuda")
        cls, patches = backbone_s(dummy)
        assert cls.device.type == "cuda"
        assert patches.device.type == "cuda"
        backbone_s.cpu()


# ── ModelConfig 테스트 ────────────────────────────────────────────────────────

class TestModelConfig:
    def test_embed_dim_s(self) -> None:
        from src.models.hazard_model import ModelConfig
        assert ModelConfig(backbone_name="dinov2_vits14").embed_dim == DIM_S

    def test_embed_dim_b(self) -> None:
        from src.models.hazard_model import ModelConfig
        assert ModelConfig(backbone_name="dinov2_vitb14").embed_dim == DIM_B

    def test_default_head_type(self) -> None:
        from src.models.hazard_model import ModelConfig
        assert ModelConfig().head_type == "mlp"

    def test_default_vpt_disabled(self) -> None:
        from src.models.hazard_model import ModelConfig
        assert ModelConfig().vpt.enabled is False

    def test_vpt_config_explicit(self) -> None:
        from src.models.hazard_model import ModelConfig, VPTConfig
        cfg = ModelConfig(vpt=VPTConfig(enabled=True, num_tokens=5))
        assert cfg.vpt.enabled is True
        assert cfg.vpt.num_tokens == 5

    def test_invalid_backbone_raises(self) -> None:
        from src.models.backbone import DINOv2Backbone
        with pytest.raises(ValueError, match="Unknown backbone"):
            DINOv2Backbone("invalid_backbone")

    def test_invalid_head_type_raises(self) -> None:
        from src.models.hazard_model import HazardModel, ModelConfig
        with pytest.raises(ValueError, match="Unknown head_type"):
            HazardModel(ModelConfig(head_type="bad_type"))  # type: ignore[arg-type]


# ── Head 분류기 단위 테스트 ────────────────────────────────────────────────────

class TestHeadClassifiers:
    def test_mlp_head_output_shape(self) -> None:
        from src.models.head import MLPHead
        head = MLPHead(embed_dim=DIM_S, num_classes=NUM_CLASSES)
        cls = torch.randn(BATCH_SIZE, DIM_S)
        patches = torch.randn(BATCH_SIZE, NUM_PATCHES, DIM_S)
        assert head(cls, patches).shape == (BATCH_SIZE, NUM_CLASSES)

    def test_mlp_head_gradient_flows(self) -> None:
        from src.models.head import MLPHead
        head = MLPHead(embed_dim=DIM_S, num_classes=NUM_CLASSES)
        cls = torch.randn(BATCH_SIZE, DIM_S)
        patches = torch.randn(BATCH_SIZE, NUM_PATCHES, DIM_S)
        head(cls, patches).sum().backward()
        for name, param in head.named_parameters():
            assert param.grad is not None, f"MLPHead.{name}에 grad 없음"

    def test_attention_head_output_shape(self) -> None:
        from src.models.head import AttentionHead
        head = AttentionHead(embed_dim=DIM_S, num_classes=NUM_CLASSES)
        cls = torch.randn(BATCH_SIZE, DIM_S)
        patches = torch.randn(BATCH_SIZE, NUM_PATCHES, DIM_S)
        assert head(cls, patches).shape == (BATCH_SIZE, NUM_CLASSES)

    def test_attention_head_gradient_flows(self) -> None:
        from src.models.head import AttentionHead
        head = AttentionHead(embed_dim=DIM_S, num_classes=NUM_CLASSES)
        cls = torch.randn(BATCH_SIZE, DIM_S)
        patches = torch.randn(BATCH_SIZE, NUM_PATCHES, DIM_S)
        head(cls, patches).sum().backward()
        for name, param in head.named_parameters():
            assert param.grad is not None, f"AttentionHead.{name}에 grad 없음"

    def test_both_heads_same_output_rank(self) -> None:
        """모든 head가 동일한 [B, num_classes] shape를 반환해야 한다."""
        from src.models.head import AttentionHead, MLPHead
        cls = torch.randn(BATCH_SIZE, DIM_S)
        patches = torch.randn(BATCH_SIZE, NUM_PATCHES, DIM_S)
        expected = (BATCH_SIZE, NUM_CLASSES)
        for head in [
            MLPHead(DIM_S, num_classes=NUM_CLASSES),
            AttentionHead(DIM_S, num_classes=NUM_CLASSES),
        ]:
            logits = head(cls, patches)
            assert logits.shape == expected, \
                f"{type(head).__name__} output shape {logits.shape} ≠ {expected}"


# ── HazardModel 통합 테스트 ──────────────────────────────────────────────────

class TestHazardModelMlp:
    def test_forward_output_shape(
        self, model_mlp: object, dummy_batch: torch.Tensor
    ) -> None:
        with torch.inference_mode():
            logits = model_mlp(dummy_batch)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_trainable_params_exclude_backbone(self, model_mlp: object) -> None:
        """backbone 파라미터는 requires_grad=False여야 한다."""
        backbone_ids = {id(p) for p in model_mlp.backbone.parameters()}
        for p in model_mlp.parameters():
            if p.requires_grad:
                assert id(p) not in backbone_ids, "backbone 파라미터가 학습 파라미터에 포함됨"

    def test_trainable_params_nonempty(self, model_mlp: object) -> None:
        trainable = [p for p in model_mlp.parameters() if p.requires_grad]
        assert len(trainable) > 0, "학습 가능한 파라미터가 없음"
