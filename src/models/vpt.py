from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as grad_ckpt

from .backbone import DINOv2Backbone

_PROMPT_INIT_STD = 0.02


def _apply_block_with_prompts(
    block: nn.Module,
    tokens: torch.Tensor,
    prompts: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Insert prompts, run block, remove prompt outputs."""
    cls = tokens[:, :1]
    patches = tokens[:, 1:]
    tokens = torch.cat([cls, prompts, patches], dim=1)
    tokens = block(tokens)
    # 프롬프트 출력은 버리고 CLS + patch만 유지
    return torch.cat([tokens[:, :1], tokens[:, 1 + num_tokens :]], dim=1)


class VPTBackbone(nn.Module):
    """DINOv2 backbone with VPT-Deep prompt tokens.

    insert_from_layer=0 (default): prompts injected at every block (original behavior).
    insert_from_layer=k: blocks 0..k-1 run as frozen (no prompts), prompts start at block k.
    Only blocks from insert_from_layer onward have learnable prompt parameters.
    """

    def __init__(
        self,
        backbone: DINOv2Backbone,
        num_tokens: int = 10,
        dropout: float = 0.1,
        use_grad_checkpoint: bool = True,
        insert_from_layer: int = 0,
    ) -> None:
        super().__init__()
        num_blocks = backbone.num_blocks
        if not (0 <= insert_from_layer < num_blocks):
            raise ValueError(
                f"insert_from_layer={insert_from_layer} must be in [0, {num_blocks - 1}]"
            )
        self.backbone = backbone
        self.num_tokens = num_tokens
        self.use_grad_checkpoint = use_grad_checkpoint
        self.insert_from_layer = insert_from_layer

        # only allocate prompts for layers that actually use them
        active_blocks = num_blocks - insert_from_layer
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, num_tokens, backbone.embed_dim) * _PROMPT_INIT_STD)
            for _ in range(active_blocks)
        ])
        self.prompt_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            cls_token: [B, embed_dim]
            patch_tokens: [B, N, embed_dim]
        """
        B = x.shape[0]
        dino = self.backbone._dino

        # backbone의 patch embed + CLS prepend + positional embed
        tokens: torch.Tensor = dino.prepare_tokens_with_masks(x)  # [B, 1+N, D]

        for i, block in enumerate(dino.blocks):
            if i < self.insert_from_layer:
                # frozen layers: no trainable params → no_grad saves activation memory
                with torch.no_grad():
                    tokens = block(tokens)
            else:
                prompt_idx = i - self.insert_from_layer
                prompts = self.prompt_dropout(
                    self.prompts[prompt_idx].expand(B, -1, -1)
                )  # [B, T, D]

                if self.use_grad_checkpoint and torch.is_grad_enabled():
                    # VRAM 절약: 각 블록 forward를 checkpoint로 래핑
                    def _step(
                        t: torch.Tensor, b=block, p=prompts, n=self.num_tokens
                    ) -> torch.Tensor:
                        return _apply_block_with_prompts(b, t, p, n)

                    tokens = grad_ckpt.checkpoint(_step, tokens, use_reentrant=False)
                else:
                    tokens = _apply_block_with_prompts(block, tokens, prompts, self.num_tokens)

        tokens = dino.norm(tokens)  # [B, 1+N, D]
        return tokens[:, 0], tokens[:, 1:]
