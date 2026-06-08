from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss with optional class weights and label smoothing."""

    weight: torch.Tensor | None

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # detach for focal weight — gradients flow through ce only
        pt = F.softmax(logits.detach(), dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt) ** self.gamma
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        return (focal_weight * ce).mean()


class LearnableClassWeight(nn.Module):
    """Focal loss with per-class weights learned via gradient descent.

    Weights are reparameterized through softmax so their sum stays == num_classes.
    """

    def __init__(self, num_classes: int, init_weights: list[float]) -> None:
        super().__init__()
        init = torch.tensor(init_weights, dtype=torch.float32)
        self.w = nn.Parameter(init)
        self.num_classes = num_classes

    def get_weights(self) -> torch.Tensor:
        """softmax → sum=1 → ×num_classes → sum=num_classes."""
        return F.softmax(self.w, dim=0) * self.num_classes

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ) -> torch.Tensor:
        w = self.get_weights()                                        # (C,) requires_grad
        ce = F.cross_entropy(logits, labels, reduction="none",        # (B,) no weight arg
                             label_smoothing=label_smoothing)
        sample_w = w[labels]                                          # (B,) gradient flows here
        pt = F.softmax(logits.detach(), dim=1).gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_w = (1.0 - pt) ** gamma
        return (ce * sample_w * focal_w).mean()
