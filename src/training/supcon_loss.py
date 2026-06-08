from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020).

    Pulls same-class features together and pushes different-class features apart.
    Expects L2-normalized features; normalizes internally as a safeguard.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, dim) — pre-classifier feature vectors
            labels:   (B,)
        Returns:
            scalar loss
        """
        device = features.device
        B = features.shape[0]

        features = F.normalize(features, dim=1)

        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # numerical stability: subtract per-row max
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~self_mask  # (B, B)

        exp_sim = torch.exp(sim) * (~self_mask).float()
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)  # (B, B)

        pos_count = pos_mask.float().sum(dim=1)  # (B,)
        valid = pos_count > 0

        if not valid.any():
            return features.new_zeros(()).requires_grad_(features.requires_grad)

        mean_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
        mean_log_prob = mean_log_prob[valid] / pos_count[valid]
        return -mean_log_prob.mean()
