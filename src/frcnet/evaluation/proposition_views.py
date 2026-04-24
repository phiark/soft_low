from __future__ import annotations

from dataclasses import dataclass

import torch

from frcnet.utils import ternary_entropy_from_masses


@dataclass(frozen=True, slots=True)
class PropositionViewSpec:
    view_name: str
    label_aware: bool


@dataclass(slots=True)
class PropositionView:
    spec: PropositionViewSpec
    truth_mass: torch.Tensor
    false_mass: torch.Tensor
    unknown_mass: torch.Tensor
    truth_ratio: torch.Tensor
    ternary_entropy: torch.Tensor


TOP1_VIEW_SPEC = PropositionViewSpec(view_name="top1_view", label_aware=False)
TARGET_VIEW_SPEC = PropositionViewSpec(view_name="target_view", label_aware=True)
CANDIDATE_VIEW_SPEC = PropositionViewSpec(view_name="candidate_view", label_aware=True)
EMPTY_VIEW_SPEC = PropositionViewSpec(view_name="empty_view", label_aware=False)


def _truth_ratio(truth_mass: torch.Tensor, false_mass: torch.Tensor) -> torch.Tensor:
    resolved_mass = truth_mass + false_mass
    truth_ratio = torch.zeros_like(truth_mass)
    mask = resolved_mass > torch.finfo(resolved_mass.dtype).eps
    truth_ratio[mask] = truth_mass[mask] / resolved_mass[mask]
    return truth_ratio


def _build_view(
    *,
    spec: PropositionViewSpec,
    truth_mass: torch.Tensor,
    class_mass: torch.Tensor,
    unknown_mass: torch.Tensor,
) -> PropositionView:
    false_mass = (class_mass.sum(dim=1) - truth_mass).clamp_min(0.0)
    return PropositionView(
        spec=spec,
        truth_mass=truth_mass,
        false_mass=false_mass,
        unknown_mass=unknown_mass,
        truth_ratio=_truth_ratio(truth_mass, false_mass),
        ternary_entropy=ternary_entropy_from_masses(truth_mass, false_mass, unknown_mass),
    )


def build_top1_view(class_mass: torch.Tensor, unknown_mass: torch.Tensor) -> PropositionView:
    truth_mass = class_mass.max(dim=1).values
    return _build_view(
        spec=TOP1_VIEW_SPEC,
        truth_mass=truth_mass,
        class_mass=class_mass,
        unknown_mass=unknown_mass,
    )


def build_target_view(
    class_mass: torch.Tensor,
    unknown_mass: torch.Tensor,
    target_class_index: torch.Tensor,
) -> PropositionView:
    if target_class_index.ndim != 1 or int(target_class_index.shape[0]) != int(class_mass.shape[0]):
        raise ValueError("target_class_index must be a 1D tensor aligned with class_mass.")
    if torch.any(target_class_index < 0) or torch.any(target_class_index >= int(class_mass.shape[1])):
        raise ValueError("target_class_index entries must be valid class indices.")
    truth_mass = class_mass.gather(1, target_class_index.long().unsqueeze(1)).squeeze(1)
    return _build_view(
        spec=TARGET_VIEW_SPEC,
        truth_mass=truth_mass,
        class_mass=class_mass,
        unknown_mass=unknown_mass,
    )


def build_candidate_view(
    class_mass: torch.Tensor,
    unknown_mass: torch.Tensor,
    candidate_class_mask: torch.Tensor,
) -> PropositionView:
    if candidate_class_mask.ndim != 2 or tuple(candidate_class_mask.shape) != tuple(class_mass.shape):
        raise ValueError("candidate_class_mask must be a 2D tensor aligned with class_mass.")
    if candidate_class_mask.dtype is not torch.bool:
        raise ValueError("candidate_class_mask must use torch.bool dtype.")
    if torch.any(candidate_class_mask.sum(dim=1) == 0):
        raise ValueError("candidate_class_mask must expose at least one class per row.")
    truth_mass = (class_mass * candidate_class_mask.to(dtype=class_mass.dtype)).sum(dim=1)
    return _build_view(
        spec=CANDIDATE_VIEW_SPEC,
        truth_mass=truth_mass,
        class_mass=class_mass,
        unknown_mass=unknown_mass,
    )


def build_empty_view(class_mass: torch.Tensor, unknown_mass: torch.Tensor) -> PropositionView:
    truth_mass = torch.zeros_like(unknown_mass)
    return _build_view(
        spec=EMPTY_VIEW_SPEC,
        truth_mass=truth_mass,
        class_mass=class_mass,
        unknown_mass=unknown_mass,
    )
