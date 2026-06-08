from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal_loss import FocalLoss


class DangerRecallLoss(nn.Module):
    """CE + danger recall 직접 최대화.

    Loss_total = CE(logits, labels) + lambda_dr * (1 - danger_recall)
    danger_recall = danger 샘플의 danger 예측 확률 평균 (class index 1)
    lambda_dr=0 이면 순수 CE와 동일.
    """

    def __init__(self, lambda_dr: float = 1.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.lambda_dr = lambda_dr
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_ce = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        if self.lambda_dr == 0.0:
            return loss_ce

        danger_mask = (labels == 1).float()
        danger_prob = F.softmax(logits, dim=1)[:, 1]
        danger_recall = (danger_prob * danger_mask).sum() / (danger_mask.sum() + 1e-8)
        loss_dr = 1.0 - danger_recall

        return loss_ce + self.lambda_dr * loss_dr


class FocalDangerRecallLoss(nn.Module):
    """Focal Loss + danger recall 직접 최대화.

    Loss_total = FocalLoss(logits, labels) + lambda_dr * (1 - danger_recall)
    danger class index: 1
    """

    def __init__(
        self,
        lambda_dr: float = 1.0,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lambda_dr = lambda_dr
        self.focal = FocalLoss(gamma=gamma, weight=weight, label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_focal = self.focal(logits, labels)

        if self.lambda_dr == 0.0:
            return loss_focal

        danger_mask = (labels == 1).float()
        danger_prob = F.softmax(logits, dim=1)[:, 1]
        danger_recall = (danger_prob * danger_mask).sum() / (danger_mask.sum() + 1e-8)
        loss_dr = 1.0 - danger_recall

        return loss_focal + self.lambda_dr * loss_dr
