from __future__ import annotations

import pytest
import torch

from frcnet.evaluation import reference_scores_from_logits


def test_reference_scores_from_logits_are_external_score_records():
    logits = torch.tensor([[4.0, 1.0, 0.0], [0.1, 0.2, 0.3]], dtype=torch.float32)

    entropy_records = reference_scores_from_logits(logits, ["a", "b"], score_name="softmax_entropy")
    max_probability_records = reference_scores_from_logits(
        logits,
        ["a", "b"],
        score_name="softmax_max_probability",
    )

    assert entropy_records[0].reference_score_name == "softmax_entropy"
    assert entropy_records[0].reference_score_value < entropy_records[1].reference_score_value
    assert max_probability_records[0].reference_score_value > max_probability_records[1].reference_score_value


def test_reference_scores_reject_unknown_score_name():
    with pytest.raises(ValueError, match="Unsupported score_name"):
        reference_scores_from_logits(torch.randn(2, 3), ["a", "b"], score_name="bad")
