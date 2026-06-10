from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

BACKBONE_REGISTRY: dict[str, dict[str, int]] = {
    "dinov2_vits14":           {"embed_dim": 384, "patch_size": 14, "num_heads": 6},
    "dinov2_vitb14":           {"embed_dim": 768, "patch_size": 14, "num_heads": 12},
    "eva02_base_patch14_448":  {"embed_dim": 768, "patch_size": 14, "num_heads": 12},
}

_HUB_REMOTE = "facebookresearch/dinov2"
_HUB_LOCAL = str(Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main")
_EVA02_NAMES: frozenset[str] = frozenset({"eva02_base_patch14_448"})


class DINOv2Backbone(nn.Module):
    """DINOv2 backbone returning CLS token and patch tokens."""

    def __init__(self, model_name: str = "dinov2_vits14", frozen: bool = True) -> None:
        super().__init__()
        if model_name not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone: {model_name}. Choose from {list(BACKBONE_REGISTRY)}")
        cfg = BACKBONE_REGISTRY[model_name]
        self.model_name = model_name
        self.frozen = frozen
        self.embed_dim: int = cfg["embed_dim"]
        self.patch_size: int = cfg["patch_size"]
        self.num_heads: int = cfg["num_heads"]

        # 로컬 캐시 우선; 없으면 GitHub에서 다운로드
        hub_src, hub_source_kw = (
            (_HUB_LOCAL, "local") if Path(_HUB_LOCAL).exists() else (_HUB_REMOTE, "github")
        )
        self._dino = torch.hub.load(hub_src, model_name, source=hub_source_kw)
        if frozen:
            self._freeze()

    def _freeze(self) -> None:
        for param in self._dino.parameters():
            param.requires_grad = False
        self._dino.eval()

    def train(self, mode: bool = True) -> DINOv2Backbone:
        super().train(mode)
        if self.frozen:
            self._dino.eval()  # frozen backbone은 학습 중에도 eval 유지
        return self

    @property
    def num_blocks(self) -> int:
        return len(self._dino.blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            cls_token: [B, embed_dim]
            patch_tokens: [B, N, embed_dim]  N = (H / patch_size)^2
        """
        # torch.no_grad(): inference_mode와 동등하게 grad를 차단하지만
        # ONNX dynamo exporter 및 torch.export.export와 호환된다.
        if self.frozen:
            with torch.no_grad():
                features = self._dino.forward_features(x)
        else:
            features = self._dino.forward_features(x)
        cls_token: torch.Tensor = features["x_norm_clstoken"]
        patch_tokens: torch.Tensor = features["x_norm_patchtokens"]
        return cls_token, patch_tokens


class EVA02Backbone(nn.Module):
    """EVA-02 backbone (via timm) returning CLS token and patch tokens.

    timm lazy import: DINOv2 전용 환경에서 timm 미설치 시에도 모듈을 import할 수 있다.
    forward_features → [B, N+1, D] where index 0 = CLS, 1: = patch tokens.
    """

    def __init__(self, model_name: str = "eva02_base_patch14_448", frozen: bool = True) -> None:
        super().__init__()
        if model_name not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone: {model_name}. Choose from {list(BACKBONE_REGISTRY)}")
        import timm  # noqa: PLC0415 — EVA-02 사용 시에만 필요
        cfg = BACKBONE_REGISTRY[model_name]
        self.model_name = model_name
        self.frozen = frozen
        self.embed_dim: int = cfg["embed_dim"]
        self.patch_size: int = cfg["patch_size"]
        self.num_heads: int = cfg["num_heads"]
        self._model = timm.create_model(model_name, pretrained=True, num_classes=0)
        if frozen:
            self._freeze()

    def _freeze(self) -> None:
        for param in self._model.parameters():
            param.requires_grad = False
        self._model.eval()

    def train(self, mode: bool = True) -> EVA02Backbone:
        super().train(mode)
        if self.frozen:
            self._model.eval()  # frozen backbone은 학습 중에도 eval 유지
        return self

    @property
    def num_blocks(self) -> int:
        return len(self._model.blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cls_token:    [B, embed_dim]
            patch_tokens: [B, N, embed_dim]  N = (H / patch_size)^2
        """
        if self.frozen:
            with torch.no_grad():
                tokens = self._model.forward_features(x)  # [B, N+1, D]
        else:
            tokens = self._model.forward_features(x)  # [B, N+1, D]
        return tokens[:, 0], tokens[:, 1:]


def build_backbone(model_name: str, frozen: bool = True) -> DINOv2Backbone | EVA02Backbone:
    """Backbone 팩토리: eva02_* → EVA02Backbone, 나머지 → DINOv2Backbone."""
    if model_name in _EVA02_NAMES:
        return EVA02Backbone(model_name, frozen=frozen)
    return DINOv2Backbone(model_name, frozen=frozen)
