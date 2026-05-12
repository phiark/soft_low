from __future__ import annotations

from pathlib import Path

import pytest
import torch

from frcnet.evaluation.reference_baselines import (
    ENERGY_SCORE_NAME,
    MAX_SOFTMAX_PROBABILITY_SCORE_NAME,
    PREDICTIVE_ENTROPY_SCORE_NAME,
    SOFTMAX_ENTROPY_SCORE_NAME,
    SOFTMAX_CE_REFERENCE_FAMILY,
    build_reference_score_records,
    compute_softmax_reference_scores,
    read_reference_score_records,
    resolve_softmax_reference_score_names,
    softmax_entropy_reference_scores,
    write_reference_score_records,
)


def test_compute_softmax_reference_scores_returns_expected_math() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    score_names = (
        MAX_SOFTMAX_PROBABILITY_SCORE_NAME,
        PREDICTIVE_ENTROPY_SCORE_NAME,
        SOFTMAX_ENTROPY_SCORE_NAME,
        ENERGY_SCORE_NAME,
    )

    scores = compute_softmax_reference_scores(logits, score_names)

    probabilities = torch.softmax(logits, dim=-1)
    safe_probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).eps)
    expected_entropy = -(safe_probabilities * torch.log(safe_probabilities)).sum(dim=-1)
    assert tuple(scores) == score_names
    torch.testing.assert_close(
        scores[MAX_SOFTMAX_PROBABILITY_SCORE_NAME],
        probabilities.max(dim=-1).values,
    )
    torch.testing.assert_close(scores[PREDICTIVE_ENTROPY_SCORE_NAME], expected_entropy)
    torch.testing.assert_close(scores[SOFTMAX_ENTROPY_SCORE_NAME], expected_entropy)
    torch.testing.assert_close(softmax_entropy_reference_scores(logits), expected_entropy)
    torch.testing.assert_close(scores[ENERGY_SCORE_NAME], -torch.logsumexp(logits, dim=-1))


def test_resolve_softmax_reference_score_names_keeps_legacy_default() -> None:
    assert resolve_softmax_reference_score_names({}) == (SOFTMAX_ENTROPY_SCORE_NAME,)
    assert resolve_softmax_reference_score_names({"score_name": ENERGY_SCORE_NAME}) == (
        ENERGY_SCORE_NAME,
    )
    assert resolve_softmax_reference_score_names(
        {
            "score_name": SOFTMAX_ENTROPY_SCORE_NAME,
            "score_names": [MAX_SOFTMAX_PROBABILITY_SCORE_NAME, PREDICTIVE_ENTROPY_SCORE_NAME],
        }
    ) == (MAX_SOFTMAX_PROBABILITY_SCORE_NAME, PREDICTIVE_ENTROPY_SCORE_NAME)

    with pytest.raises(ValueError, match="Unsupported softmax CE reference score"):
        resolve_softmax_reference_score_names({"score_names": ["odin_score"]})


def test_reference_score_records_roundtrip_multiple_scores(tmp_path: Path) -> None:
    score_tensors = {
        MAX_SOFTMAX_PROBABILITY_SCORE_NAME: torch.tensor([0.8, 0.6], dtype=torch.float32),
        PREDICTIVE_ENTROPY_SCORE_NAME: torch.tensor([0.4, 0.7], dtype=torch.float32),
        ENERGY_SCORE_NAME: torch.tensor([-2.0, -1.5], dtype=torch.float32),
    }
    records = build_reference_score_records(
        score_tensors=score_tensors,
        sample_ids=["sample-a", "sample-b"],
        split_names=["test", "test"],
        cohort_names=["easy_id", "ood"],
        source_dataset_names=["cifar10", "cifar100"],
        reference_model_family=SOFTMAX_CE_REFERENCE_FAMILY,
        reference_run_id="softmax-reference-test",
    )

    output_path = write_reference_score_records(records, tmp_path / "reference_score_records.jsonl")
    loaded_records = read_reference_score_records(output_path)

    assert loaded_records == records
    assert len(loaded_records) == 6
    assert [record.sample_id for record in loaded_records[:3]] == [
        "sample-a",
        "sample-a",
        "sample-a",
    ]
    assert [record.reference_score_name for record in loaded_records[:3]] == list(score_tensors)
    assert [record.sample_id for record in loaded_records[3:]] == [
        "sample-b",
        "sample-b",
        "sample-b",
    ]
