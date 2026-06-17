from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

_DEFAULT_ECE_BINS: int = 15


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = _DEFAULT_ECE_BINS) -> float:
    """Expected Calibration Error (max-confidence, equal-width binning)."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == labels
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_acc = float(accuracies[in_bin].mean())
        bin_conf = float(confidences[in_bin].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


class TemperatureScaler(nn.Module):
    """Post-hoc calibration via temperature scaling.

    val set의 NLL을 최소화하도록 T만 학습. test set으로 T를 선택하면 안 된다.
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """val set logits/labels로 temperature를 최적화. 최종 T 값을 반환."""
        self.train()
        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        self.eval()
        return float(self.temperature.item())

    @torch.inference_mode()
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """학습 완료 후 logits에 temperature 적용, softmax 확률 반환."""
        self.eval()
        scaled = self.forward(logits)
        return torch.softmax(scaled, dim=-1)
