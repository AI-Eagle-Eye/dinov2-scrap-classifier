from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

from .backbone import BACKBONE_REGISTRY, DINOv2Backbone, EVA02Backbone, build_backbone
from .head import AttentionHead, ClassAwareHead, LinearHead, MLPHead
from .vpt import VPTBackbone

_NUM_CLASSES = 3

HeadType = Literal["mlp", "attention", "class_aware", "linear"]


@dataclass(slots=True)
class VPTConfig:
    enabled: bool = False
    num_tokens: int = 10
    insert_from_layer: int = 0


@dataclass(slots=True)
class ModelConfig:
    backbone_name: str = "dinov2_vitb14"
    head_type: HeadType = "mlp"
    vpt: VPTConfig = field(default_factory=VPTConfig)
    dropout: float = 0.3
    num_classes: int = _NUM_CLASSES
    use_grad_checkpoint: bool = True
    class_aware_init_weights: list[float] | None = None
    backbone_frozen: bool = True
    unfreeze_last_n: int = 0
    head_use_cls: bool = False

    @property
    def embed_dim(self) -> int:
        return BACKBONE_REGISTRY[self.backbone_name]["embed_dim"]


def _build_head(
    head_type: str,
    embed_dim: int,
    num_classes: int,
    dropout: float,
    class_aware_init_weights: list[float] | None = None,
    head_use_cls: bool = False,
) -> nn.Module:
    if head_type == "mlp":
        return MLPHead(embed_dim, num_classes=num_classes, dropout=dropout)
    if head_type == "attention":
        return AttentionHead(embed_dim, num_classes=num_classes, dropout=dropout, use_cls=head_use_cls)
    if head_type == "class_aware":
        return ClassAwareHead(
            embed_dim,
            num_classes=num_classes,
            dropout=dropout,
            init_weights=class_aware_init_weights,
        )
    if head_type == "linear":
        return LinearHead(embed_dim, num_classes=num_classes)
    raise ValueError(f"Unknown head_type: {head_type!r}. Choose from ('mlp', 'attention', 'class_aware', 'linear')")


class HazardModel(nn.Module):
    """DINOv2 backbone + optional VPT + classification head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.backbone: DINOv2Backbone | EVA02Backbone = build_backbone(
            config.backbone_name, frozen=config.backbone_frozen, unfreeze_last_n=config.unfreeze_last_n
        )

        self.vpt: VPTBackbone | None = None
        if config.vpt.enabled:
            self.vpt = VPTBackbone(
                self.backbone,
                num_tokens=config.vpt.num_tokens,
                dropout=config.dropout,
                use_grad_checkpoint=config.use_grad_checkpoint,
                insert_from_layer=config.vpt.insert_from_layer,
            )

        self.head = _build_head(
            config.head_type,
            config.embed_dim,
            config.num_classes,
            config.dropout,
            config.class_aware_init_weights,
            config.head_use_cls,
        )

    def _extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.vpt is not None:
            return self.vpt(x)
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token, patch_tokens = self._extract_features(x)
        return self.head(cls_token, patch_tokens)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cls_token, logits) for SupCon training."""
        cls_token, patch_tokens = self._extract_features(x)
        logits = self.head(cls_token, patch_tokens)
        return cls_token, logits

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


def build_model(config: ModelConfig) -> HazardModel:
    return HazardModel(config)
