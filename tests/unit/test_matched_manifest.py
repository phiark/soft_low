from __future__ import annotations

from dataclasses import replace

import pytest

from frcnet.evaluation import build_frozen_matched_manifest, summarize_matched_manifest
from frcnet.models import FRCNetModel
from frcnet.evaluation.inference import build_sample_analysis_records
from tests.conftest import build_synthetic_batch


def _records_for_matching():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    records = build_sample_analysis_records(model(batch_input.image), batch_input, run_id="RUN-1", protocol_id="p")
    duplicated = []
    for repeat in range(4):
        for record in records:
            duplicated_record = replace(record)
            duplicated_record.sample_id = f"{record.sample_id}-{repeat}"
            duplicated.append(duplicated_record)
    return duplicated


def test_frozen_matched_manifest_hash_is_stable():
    records = _records_for_matching()
    reference_scores = {record.sample_id: float(index % 6) for index, record in enumerate(records)}

    manifest_a = build_frozen_matched_manifest(
        records,
        reference_scores=reference_scores,
        reference_score_name="softmax_entropy",
        num_bins=3,
    )
    manifest_b = build_frozen_matched_manifest(
        records,
        reference_scores=reference_scores,
        reference_score_name="softmax_entropy",
        num_bins=3,
    )
    summary = summarize_matched_manifest(manifest_a)

    assert [record.to_dict() for record in manifest_a] == [record.to_dict() for record in manifest_b]
    assert manifest_a[0].manifest_hash
    assert summary["manifest_hash"] == manifest_a[0].manifest_hash
    assert summary["reference_score_name"] == "softmax_entropy"
    assert summary["cohort_counts"]["ambiguous_id"] == summary["cohort_counts"]["ood"]


def test_frozen_matched_manifest_requires_both_cohorts():
    records = [record for record in _records_for_matching() if record.cohort_name != "ood"]
    reference_scores = {record.sample_id: 1.0 for record in records}

    with pytest.raises(ValueError, match="both cohorts"):
        build_frozen_matched_manifest(
            records,
            reference_scores=reference_scores,
            reference_score_name="softmax_entropy",
        )
