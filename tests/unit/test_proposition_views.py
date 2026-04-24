from __future__ import annotations

import inspect

import pytest
import torch

from frcnet.evaluation import build_candidate_view, build_target_view, build_top1_view


def test_top1_view_is_label_free_and_conserves_mass():
    signature = inspect.signature(build_top1_view)
    assert "target" not in "".join(signature.parameters)
    assert "candidate" not in "".join(signature.parameters)

    class_mass = torch.tensor([[0.6, 0.2, 0.1], [0.1, 0.5, 0.2]], dtype=torch.float32)
    unknown_mass = torch.tensor([0.1, 0.2], dtype=torch.float32)

    view = build_top1_view(class_mass, unknown_mass)

    assert view.spec.label_aware is False
    total_mass = view.truth_mass + view.false_mass + view.unknown_mass
    assert total_mass.tolist() == pytest.approx([1.0, 1.0])
    assert view.truth_ratio.tolist() == pytest.approx([0.6 / 0.9, 0.5 / 0.8])


def test_target_and_candidate_views_are_label_aware():
    class_mass = torch.tensor([[0.6, 0.2, 0.1], [0.1, 0.5, 0.2]], dtype=torch.float32)
    unknown_mass = torch.tensor([0.1, 0.2], dtype=torch.float32)
    target_view = build_target_view(class_mass, unknown_mass, torch.tensor([0, 1]))
    candidate_mask = torch.tensor([[True, True, False], [False, True, True]])
    candidate_view = build_candidate_view(class_mass, unknown_mass, candidate_mask)

    assert target_view.spec.label_aware is True
    assert candidate_view.spec.label_aware is True
    assert (target_view.truth_mass + target_view.false_mass + target_view.unknown_mass).tolist() == pytest.approx(
        [1.0, 1.0]
    )
    assert (candidate_view.truth_mass + candidate_view.false_mass + candidate_view.unknown_mass).tolist() == pytest.approx(
        [1.0, 1.0]
    )
