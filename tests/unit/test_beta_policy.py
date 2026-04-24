from __future__ import annotations

import pytest
import torch

from frcnet.evaluation import (
    binary_pignistic_beta,
    candidate_symmetric_beta,
    completion_from_masses,
    top1_symmetric_beta,
)


def test_beta_policy_defaults():
    assert top1_symmetric_beta(10) == pytest.approx(0.1)
    assert candidate_symmetric_beta(2, 10) == pytest.approx(0.2)
    assert binary_pignistic_beta() == pytest.approx(0.5)


def test_completion_from_masses():
    truth_mass = torch.tensor([0.3, 0.7], dtype=torch.float32)
    unknown_mass = torch.tensor([0.4, 0.2], dtype=torch.float32)

    completion = completion_from_masses(truth_mass, unknown_mass, beta=0.25)

    assert completion.tolist() == pytest.approx([0.4, 0.75])
