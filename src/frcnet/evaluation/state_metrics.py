from __future__ import annotations

from dataclasses import dataclass

import torch

from frcnet.utils import content_entropy, resolution_entropy


@dataclass(slots=True)
class StateMetrics:
    state_content_entropy: torch.Tensor
    state_weighted_content_entropy: torch.Tensor
    state_entropy: torch.Tensor


def compute_state_metrics(
    resolution_ratio: torch.Tensor,
    content_distribution: torch.Tensor,
) -> StateMetrics:
    state_content_entropy = content_entropy(content_distribution)
    state_weighted_content_entropy = resolution_ratio * state_content_entropy
    state_entropy = resolution_entropy(resolution_ratio) + state_weighted_content_entropy
    return StateMetrics(
        state_content_entropy=state_content_entropy,
        state_weighted_content_entropy=state_weighted_content_entropy,
        state_entropy=state_entropy,
    )
