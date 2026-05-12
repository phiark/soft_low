from __future__ import annotations

from frcnet.evaluation import (
    MatchedManifestRecord,
    SampleAnalysisRecord,
    summarize_decision_regret,
)


def _sample_record(
    sample_id: str,
    cohort_name: str,
    *,
    resolution_ratio: float,
    state_content_entropy: float,
    top1_completion_beta_0_1: float = 0.5,
) -> SampleAnalysisRecord:
    return SampleAnalysisRecord(
        model_family="frcnet_explicit_unknown",
        run_id="RUN-DECISION",
        protocol_id="plan_a_next_v1_0_decision_regret",
        sample_id=sample_id,
        split_name="test",
        cohort_name=cohort_name,
        source_dataset_name="synthetic",
        source_class_label=None,
        class_label=0 if cohort_name in {"easy_id", "hard_id"} else -1,
        predicted_class_index=0,
        resolution_ratio=resolution_ratio,
        unknown_mass=1.0 - resolution_ratio,
        state_content_entropy=state_content_entropy,
        state_weighted_content_entropy=resolution_ratio * state_content_entropy,
        state_entropy=state_content_entropy,
        resolution_entropy=0.0,
        top1_class_mass=top1_completion_beta_0_1,
        top1_view_truth_mass=top1_completion_beta_0_1,
        top1_view_false_mass=max(0.0, 1.0 - top1_completion_beta_0_1),
        top1_view_unknown_mass=0.0,
        top1_view_tau=top1_completion_beta_0_1,
        proposition_truth_mass=top1_completion_beta_0_1,
        proposition_false_mass=max(0.0, 1.0 - top1_completion_beta_0_1),
        proposition_unknown_mass=0.0,
        proposition_truth_ratio=top1_completion_beta_0_1,
        ternary_entropy=0.0,
        auxiliary_top1_content_probability=top1_completion_beta_0_1,
        top1_completion_beta_0_1=top1_completion_beta_0_1,
        top1_completion_beta_0_25=top1_completion_beta_0_1,
        top1_completion_beta_0_5=top1_completion_beta_0_1,
        top1_completion_beta_0_75=top1_completion_beta_0_1,
        candidate_class_indices=(0, 1) if cohort_name == "ambiguous_id" else (),
    )


def _decision_records(records_per_cohort: int = 12) -> list[SampleAnalysisRecord]:
    records: list[SampleAnalysisRecord] = []
    for index in range(records_per_cohort):
        offset = index * 0.001
        records.append(
            _sample_record(
                f"easy-{index:02d}",
                "easy_id",
                resolution_ratio=0.82 + offset,
                state_content_entropy=0.10 + offset,
            )
        )
        records.append(
            _sample_record(
                f"ambiguous-{index:02d}",
                "ambiguous_id",
                resolution_ratio=0.82 + offset,
                state_content_entropy=0.90 + offset,
            )
        )
        records.append(
            _sample_record(
                f"ood-{index:02d}",
                "ood",
                resolution_ratio=0.18 + offset,
                state_content_entropy=0.20 + offset,
            )
        )
    return records


def _manifest_record(
    sample_record: SampleAnalysisRecord,
    *,
    manifest_role: str,
    paired_group_id: str,
) -> MatchedManifestRecord:
    return MatchedManifestRecord(
        sample_id=sample_record.sample_id,
        cohort_name=sample_record.cohort_name,
        reference_score_name="synthetic_reference",
        reference_score_value=0.0,
        match_bin_id="bin-00",
        manifest_role=manifest_role,
        paired_group_id=paired_group_id,
        manifest_hash="",
        construction_config_hash="synthetic",
        source_dataset_name=sample_record.source_dataset_name,
        source_dataset_split=sample_record.source_dataset_split,
        source_role=sample_record.source_role,
        source_partition_name=sample_record.source_partition_name,
    )


def test_pair_policy_has_lower_regret_than_resolution_scalar_on_synthetic_records():
    summary = summarize_decision_regret(
        _decision_records(),
        test_size=0.5,
        random_state=7,
    )

    summaries = {policy_summary.policy_name: policy_summary for policy_summary in summary.policy_summaries}

    assert summaries["pair_resolution_ratio_state_content_entropy"].mean_regret < summaries[
        "resolution_ratio"
    ].mean_regret
    assert summaries["pair_resolution_ratio_state_content_entropy"].mean_regret == 0.0


def test_oracle_state_policy_has_zero_regret():
    summary = summarize_decision_regret(
        _decision_records(records_per_cohort=4),
        policy_names=("oracle_state",),
        test_size=0.5,
        random_state=7,
    )

    assert len(summary.policy_summaries) == 1
    assert summary.policy_summaries[0].policy_name == "oracle_state"
    assert summary.policy_summaries[0].mean_regret == 0.0
    assert all(row.regret == 0.0 for row in summary.rows)


def test_q_beta_scalar_completion_can_be_decision_blind():
    summary = summarize_decision_regret(
        _decision_records(),
        policy_names=("q_beta_top1_completion_beta_0_1",),
        test_size=0.5,
        random_state=7,
    )

    selected_actions = {row.selected_action for row in summary.rows}

    assert len(selected_actions) == 1
    assert summary.policy_summaries[0].mean_regret > 0.0


def test_frozen_manifest_uses_manifest_test_roles_for_evaluation():
    records = _decision_records(records_per_cohort=3)
    sample_not_in_manifest = _sample_record(
        "easy-extra-not-in-manifest",
        "easy_id",
        resolution_ratio=0.0,
        state_content_entropy=9.0,
    )
    manifest_missing_sample = _sample_record(
        "ood-manifest-only",
        "ood",
        resolution_ratio=1.0,
        state_content_entropy=9.0,
    )
    manifest_records = []
    expected_test_ids = set()
    for index, record in enumerate(records):
        manifest_role = "test" if index % 4 == 0 else "train"
        if index % 7 == 0:
            manifest_role = "eval_only"
        if manifest_role in {"test", "eval_only"}:
            expected_test_ids.add(record.sample_id)
        manifest_records.append(
            _manifest_record(record, manifest_role=manifest_role, paired_group_id=f"pair-{index:02d}")
        )
    manifest_records.append(
        _manifest_record(manifest_missing_sample, manifest_role="test", paired_group_id="pair-extra-missing")
    )

    summary = summarize_decision_regret(
        records + [sample_not_in_manifest],
        policy_names=("oracle_state",),
        matched_manifest_records=manifest_records,
        test_size=0.5,
        random_state=7,
    )

    assert {row.sample_id for row in summary.rows} == expected_test_ids
    assert {row.manifest_role for row in summary.rows} == {"test", "eval_only"}
    assert summary.num_test_records == len(expected_test_ids)
    assert summary.num_train_records == len(records) - len(expected_test_ids)
