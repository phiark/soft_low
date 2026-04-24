from __future__ import annotations

import pytest
import torch

from frcnet.evaluation import compute_state_metrics
from frcnet.utils import content_entropy, resolution_entropy


def test_state_entropy_decomposition():
    resolution_ratio = torch.tensor([0.25, 0.8], dtype=torch.float32)
    content_distribution = torch.tensor(
        [[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]],
        dtype=torch.float32,
    )

    metrics = compute_state_metrics(resolution_ratio, content_distribution)
    expected_content_entropy = content_entropy(content_distribution)
    expected_weighted_entropy = resolution_ratio * expected_content_entropy
    expected_state_entropy = resolution_entropy(resolution_ratio) + expected_weighted_entropy

    assert metrics.state_content_entropy.tolist() == pytest.approx(expected_content_entropy.tolist())
    assert metrics.state_weighted_content_entropy.tolist() == pytest.approx(expected_weighted_entropy.tolist())
    assert metrics.state_entropy.tolist() == pytest.approx(expected_state_entropy.tolist())
