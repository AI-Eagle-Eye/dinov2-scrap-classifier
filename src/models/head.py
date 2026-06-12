from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASSES = 3
_MLP_HIDDEN = 256
_MLP_DROPOUT = 0.3
_ATT_HEADS = 8
_CLASS_AWARE_HEADS = 4


class MLPHead(nn.Module):
    """CLS token → LayerNorm → Linear→GELU→Dropout → Linear."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = _MLP_HIDDEN,
        num_classes: int = NUM_CLASSES,
        dropout: float = _MLP_DROPOUT,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, cls_token: torch.Tensor, _patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_token:     [B, D]
            _patch_tokens: [B, N, D]  (unused)
        Returns:
            logits: [B, num_classes]
        """
        return self.net(cls_token)


class AttentionHead(nn.Module):
    """Patch tokens → MultiHeadAttention (self-attn) → mean pool → Linear.

    use_cls=True이면 CLS token을 patch token 앞에 prepend하여 self-attention과
    mean pool 대상에 포함시킨다.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = _ATT_HEADS,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
        use_cls: bool = False,
    ) -> None:
        super().__init__()
        self.use_cls = use_cls
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_token:    [B, D]  (use_cls=True일 때만 사용)
            patch_tokens: [B, N, D]
        Returns:
            logits: [B, num_classes]
        """
        if self.use_cls:
            tokens = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        else:
            tokens = patch_tokens
        attended, _ = self.attn(tokens, tokens, tokens)
        pooled = self.norm(attended.mean(dim=1))
        return self.classifier(pooled)


class LinearHead(nn.Module):
    """CLS token → Linear (pure linear probe)."""

    def __init__(self, embed_dim: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, cls_token: torch.Tensor, _patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.fc(cls_token)


class ClassAwareHead(nn.Module):
    """CLS token (query) attends to class-aware learnable tokens (key/value) → logits.

    Rare classes receive a proportionally larger initial scale via inverse-frequency
    init_weights, giving the head a head start on imbalanced distributions.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = NUM_CLASSES,
        num_heads: int = _CLASS_AWARE_HEADS,
        dropout: float = 0.1,
        init_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.class_tokens = nn.Parameter(torch.empty(num_classes, embed_dim))
        self._init_class_tokens(init_weights)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def _init_class_tokens(self, init_weights: list[float] | None) -> None:
        nn.init.normal_(self.class_tokens, std=0.02)
        if init_weights is None:
            return
        w = torch.tensor(init_weights, dtype=torch.float32)
        w = w / w.mean()  # normalize so overall scale stays near 0.02
        with torch.no_grad():
            self.class_tokens.mul_(w.unsqueeze(1))

    def forward(self, cls_token: torch.Tensor, _patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_token:     [B, D]
            _patch_tokens: [B, N, D]  (unused)
        Returns:
            logits: [B, num_classes]
        """
        B = cls_token.shape[0]
        query = cls_token.unsqueeze(1)                          # [B, 1, D]
        kv = self.class_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, num_classes, D]
        attended, _ = self.attn(query, kv, kv)                 # [B, 1, D]
        out = self.norm(cls_token + attended.squeeze(1))        # residual + norm: [B, D]
        return self.classifier(out)                             # [B, num_classes]
