from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class BetaPolicy:
    policy_name: str
    beta: float


def top1_symmetric_beta(num_classes: int) -> float:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    return 1.0 / float(num_classes)


def candidate_symmetric_beta(candidate_count: int, num_classes: int) -> float:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if not 0 < candidate_count <= num_classes:
        raise ValueError("candidate_count must be within [1, num_classes].")
    return float(candidate_count) / float(num_classes)


def binary_pignistic_beta() -> float:
    return 0.5


def completion_from_masses(
    truth_mass: torch.Tensor,
    unknown_mass: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be within [0, 1].")
    return truth_mass + (float(beta) * unknown_mass)
