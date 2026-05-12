from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from frcnet.evaluation.matched_manifest import MatchedManifestRecord
from frcnet.evaluation.records import SampleAnalysisRecord

ACCEPT_KNOWN_ACTION = "accept_known"
DEFER_ACTION = "defer"
REJECT_UNKNOWN_ACTION = "reject_unknown"

DEFAULT_DECISION_ACTIONS = (ACCEPT_KNOWN_ACTION, DEFER_ACTION, REJECT_UNKNOWN_ACTION)
DEFAULT_COHORT_ORACLE_ACTIONS = {
    "easy_id": ACCEPT_KNOWN_ACTION,
    "hard_id": ACCEPT_KNOWN_ACTION,
    "ambiguous_id": DEFER_ACTION,
    "ood": REJECT_UNKNOWN_ACTION,
    "unknown_supervision": REJECT_UNKNOWN_ACTION,
}
DEFAULT_DECISION_ACTION_UTILITIES = {
    action_name: {
        other_name: float(action_name == other_name)
        for other_name in DEFAULT_DECISION_ACTIONS
    }
    for action_name in DEFAULT_DECISION_ACTIONS
}


@dataclass(frozen=True, slots=True)
class DecisionActionUtility:
    oracle_action: str
    selected_action: str
    utility: float


@dataclass(frozen=True, slots=True)
class DecisionPolicySpec:
    policy_name: str
    feature_names: tuple[str, ...] = ()
    oracle: bool = False


@dataclass(slots=True)
class DecisionRegretRow:
    policy_name: str
    sample_id: str
    cohort_name: str
    source_dataset_name: str
    source_role: str
    source_partition_name: str
    manifest_role: str
    oracle_action: str
    selected_action: str
    oracle_utility: float
    selected_utility: float
    regret: float

    def to_csv_row(self) -> dict[str, str | float]:
        return {
            "policy_name": self.policy_name,
            "sample_id": self.sample_id,
            "cohort_name": self.cohort_name,
            "source_dataset_name": self.source_dataset_name,
            "source_role": self.source_role,
            "source_partition_name": self.source_partition_name,
            "manifest_role": self.manifest_role,
            "oracle_action": self.oracle_action,
            "selected_action": self.selected_action,
            "oracle_utility": self.oracle_utility,
            "selected_utility": self.selected_utility,
            "regret": self.regret,
        }


@dataclass(slots=True)
class DecisionRegretPolicySummary:
    policy_name: str
    num_train_records: int
    num_test_records: int
    mean_regret: float
    median_regret: float
    action_accuracy: float

    def to_csv_row(self) -> dict[str, str | int | float]:
        return {
            "policy_name": self.policy_name,
            "num_train_records": self.num_train_records,
            "num_test_records": self.num_test_records,
            "mean_regret": self.mean_regret,
            "median_regret": self.median_regret,
            "action_accuracy": self.action_accuracy,
        }


@dataclass(slots=True)
class DecisionRegretBenchmarkSummary:
    policy_summaries: tuple[DecisionRegretPolicySummary, ...]
    rows: tuple[DecisionRegretRow, ...]
    num_train_records: int
    num_test_records: int
    test_size: float
    random_state: int


DEFAULT_DECISION_POLICY_SPECS = (
    DecisionPolicySpec("q_beta_top1_completion_beta_0_1", ("top1_completion_beta_0_1",)),
    DecisionPolicySpec("q_beta_top1_completion_beta_0_25", ("top1_completion_beta_0_25",)),
    DecisionPolicySpec("q_beta_top1_completion_beta_0_5", ("top1_completion_beta_0_5",)),
    DecisionPolicySpec("q_beta_top1_completion_beta_0_75", ("top1_completion_beta_0_75",)),
    DecisionPolicySpec("resolution_ratio", ("resolution_ratio",)),
    DecisionPolicySpec("state_content_entropy", ("state_content_entropy",)),
    DecisionPolicySpec("state_weighted_content_entropy", ("state_weighted_content_entropy",)),
    DecisionPolicySpec(
        "pair_resolution_ratio_state_content_entropy",
        ("resolution_ratio", "state_content_entropy"),
    ),
    DecisionPolicySpec(
        "pair_resolution_ratio_state_weighted_content_entropy",
        ("resolution_ratio", "state_weighted_content_entropy"),
    ),
    DecisionPolicySpec("oracle_state", oracle=True),
)

_POLICY_ALIASES = {
    "top1_completion_beta_0_1": "q_beta_top1_completion_beta_0_1",
    "top1_completion_beta_0_25": "q_beta_top1_completion_beta_0_25",
    "top1_completion_beta_0_5": "q_beta_top1_completion_beta_0_5",
    "top1_completion_beta_0_75": "q_beta_top1_completion_beta_0_75",
    "pair_resolution_entropy": "pair_resolution_ratio_state_content_entropy",
    "pair_resolution_weighted_entropy": "pair_resolution_ratio_state_weighted_content_entropy",
}


def _resolve_policy_specs(policy_names: Sequence[str] | None) -> tuple[DecisionPolicySpec, ...]:
    spec_by_name = {spec.policy_name: spec for spec in DEFAULT_DECISION_POLICY_SPECS}
    if policy_names is None:
        return DEFAULT_DECISION_POLICY_SPECS

    resolved_specs: list[DecisionPolicySpec] = []
    for policy_name in policy_names:
        resolved_name = _POLICY_ALIASES.get(str(policy_name), str(policy_name))
        if resolved_name not in spec_by_name:
            supported = sorted((*spec_by_name, *_POLICY_ALIASES))
            raise ValueError(
                f"Unsupported decision policy `{policy_name}`. Supported values: {supported}"
            )
        if spec_by_name[resolved_name] not in resolved_specs:
            resolved_specs.append(spec_by_name[resolved_name])
    return tuple(resolved_specs)


def _oracle_action(
    record: SampleAnalysisRecord,
    cohort_oracle_actions: Mapping[str, str],
) -> str:
    try:
        return str(cohort_oracle_actions[record.cohort_name])
    except KeyError as exc:
        raise ValueError(
            f"Unsupported cohort for decision-regret oracle: `{record.cohort_name}`."
        ) from exc


def _utility(
    *,
    oracle_action: str,
    selected_action: str,
    utility_matrix: Mapping[str, Mapping[str, float]],
) -> float:
    try:
        return float(utility_matrix[oracle_action][selected_action])
    except KeyError as exc:
        raise ValueError(
            f"Missing decision utility for oracle_action={oracle_action} "
            f"selected_action={selected_action}."
        ) from exc


def _record_feature(record: SampleAnalysisRecord, feature_name: str) -> float:
    return float(getattr(record, feature_name))


def _feature_matrix(
    records: Sequence[SampleAnalysisRecord],
    feature_names: Sequence[str],
) -> np.ndarray:
    return np.array(
        [
            [_record_feature(record, feature_name) for feature_name in feature_names]
            for record in records
        ],
        dtype=np.float64,
    )


def _prepare_records(
    sample_analysis_records: Sequence[SampleAnalysisRecord],
    *,
    cohort_oracle_actions: Mapping[str, str],
    test_size: float,
    random_state: int,
    matched_manifest_records: Sequence[MatchedManifestRecord] | None,
) -> tuple[list[SampleAnalysisRecord], np.ndarray, np.ndarray, tuple[str, ...]]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be within (0, 1).")

    usable_cohorts = set(cohort_oracle_actions)
    usable_records = [
        record
        for record in sample_analysis_records
        if record.cohort_name in usable_cohorts
    ]
    if len(usable_records) < 2:
        raise ValueError("Decision-regret benchmark requires at least two usable records.")
    record_by_id = {record.sample_id: record for record in usable_records}

    manifest_roles: list[str]
    if matched_manifest_records is None:
        ordered_records = sorted(usable_records, key=lambda record: record.sample_id)
        oracle_actions = np.array(
            [_oracle_action(record, cohort_oracle_actions) for record in ordered_records],
            dtype=object,
        )
        unique_actions, action_counts = np.unique(oracle_actions, return_counts=True)
        stratify = (
            oracle_actions
            if len(unique_actions) > 1 and int(action_counts.min()) >= 2
            else None
        )
        train_index, test_index = train_test_split(
            np.arange(len(ordered_records)),
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        manifest_roles = ["train"] * len(ordered_records)
        for index in test_index.tolist():
            manifest_roles[index] = "test"
        return ordered_records, np.sort(train_index), np.sort(test_index), tuple(manifest_roles)

    selected_manifest = [
        record
        for record in matched_manifest_records
        if record.sample_id in record_by_id
        and record.manifest_role in {"train", "test", "eval_only"}
    ]
    if not selected_manifest:
        raise ValueError("Frozen matched manifest did not match any decision-regret records.")
    selected_manifest = sorted(
        selected_manifest,
        key=lambda record: (record.paired_group_id, record.cohort_name, record.sample_id),
    )
    ordered_records = [record_by_id[record.sample_id] for record in selected_manifest]
    manifest_roles = [record.manifest_role for record in selected_manifest]
    train_index = np.array(
        [index for index, role in enumerate(manifest_roles) if role == "train"],
        dtype=np.int64,
    )
    test_index = np.array(
        [index for index, role in enumerate(manifest_roles) if role in {"test", "eval_only"}],
        dtype=np.int64,
    )
    if train_index.size == 0 or test_index.size == 0:
        raise ValueError(
            "Frozen decision-regret manifest must include train and test/eval_only records."
        )
    return ordered_records, train_index, test_index, tuple(manifest_roles)


def _majority_action(actions: Sequence[str]) -> str:
    counts: dict[str, int] = {}
    for action in actions:
        counts[str(action)] = counts.get(str(action), 0) + 1
    return sorted(counts, key=lambda action: (-counts[action], action))[0]


def _predict_policy_actions(
    *,
    policy_spec: DecisionPolicySpec,
    ordered_records: Sequence[SampleAnalysisRecord],
    oracle_actions: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    random_state: int,
) -> tuple[str, ...]:
    if policy_spec.oracle:
        return tuple(str(value) for value in oracle_actions[test_index])

    train_actions = oracle_actions[train_index]
    if len(set(str(value) for value in train_actions)) < 2:
        return tuple([str(train_actions[0])] * int(test_index.size))

    features = _feature_matrix(ordered_records, policy_spec.feature_names)
    train_features = features[train_index]
    test_features = features[test_index]
    if np.allclose(train_features, train_features[0]):
        majority_action = _majority_action([str(value) for value in train_actions])
        return tuple([majority_action] * int(test_index.size))

    classifier = LogisticRegression(random_state=random_state, max_iter=1000, multi_class="auto")
    classifier.fit(train_features, train_actions)
    return tuple(str(value) for value in classifier.predict(test_features))


def summarize_decision_regret(
    sample_analysis_records: Sequence[SampleAnalysisRecord],
    *,
    policy_names: Sequence[str] | None = None,
    cohort_oracle_actions: Mapping[str, str] = DEFAULT_COHORT_ORACLE_ACTIONS,
    utility_matrix: Mapping[str, Mapping[str, float]] = DEFAULT_DECISION_ACTION_UTILITIES,
    test_size: float = 0.3,
    random_state: int = 7,
    matched_manifest_records: Sequence[MatchedManifestRecord] | None = None,
) -> DecisionRegretBenchmarkSummary:
    policy_specs = _resolve_policy_specs(policy_names)
    ordered_records, train_index, test_index, manifest_roles = _prepare_records(
        sample_analysis_records,
        cohort_oracle_actions=cohort_oracle_actions,
        test_size=test_size,
        random_state=random_state,
        matched_manifest_records=matched_manifest_records,
    )
    oracle_actions = np.array(
        [_oracle_action(record, cohort_oracle_actions) for record in ordered_records],
        dtype=object,
    )

    summaries: list[DecisionRegretPolicySummary] = []
    rows: list[DecisionRegretRow] = []
    for policy_spec in policy_specs:
        selected_actions = _predict_policy_actions(
            policy_spec=policy_spec,
            ordered_records=ordered_records,
            oracle_actions=oracle_actions,
            train_index=train_index,
            test_index=test_index,
            random_state=random_state,
        )
        policy_regrets: list[float] = []
        correct_action_count = 0
        for sample_index, selected_action in zip(
            test_index.tolist(),
            selected_actions,
            strict=True,
        ):
            record = ordered_records[sample_index]
            oracle_action = str(oracle_actions[sample_index])
            oracle_utility = _utility(
                oracle_action=oracle_action,
                selected_action=oracle_action,
                utility_matrix=utility_matrix,
            )
            selected_utility = _utility(
                oracle_action=oracle_action,
                selected_action=selected_action,
                utility_matrix=utility_matrix,
            )
            regret = oracle_utility - selected_utility
            policy_regrets.append(regret)
            correct_action_count += int(selected_action == oracle_action)
            rows.append(
                DecisionRegretRow(
                    policy_name=policy_spec.policy_name,
                    sample_id=record.sample_id,
                    cohort_name=record.cohort_name,
                    source_dataset_name=record.source_dataset_name,
                    source_role=record.source_role,
                    source_partition_name=record.source_partition_name,
                    manifest_role=manifest_roles[sample_index],
                    oracle_action=oracle_action,
                    selected_action=selected_action,
                    oracle_utility=oracle_utility,
                    selected_utility=selected_utility,
                    regret=regret,
                )
            )
        summaries.append(
            DecisionRegretPolicySummary(
                policy_name=policy_spec.policy_name,
                num_train_records=int(train_index.size),
                num_test_records=int(test_index.size),
                mean_regret=mean(policy_regrets),
                median_regret=median(policy_regrets),
                action_accuracy=correct_action_count / len(policy_regrets),
            )
        )

    return DecisionRegretBenchmarkSummary(
        policy_summaries=tuple(summaries),
        rows=tuple(rows),
        num_train_records=int(train_index.size),
        num_test_records=int(test_index.size),
        test_size=test_size,
        random_state=random_state,
    )


def write_decision_regret_policy_summaries(
    summary: DecisionRegretBenchmarkSummary,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            list(summary.policy_summaries[0].to_csv_row()) if summary.policy_summaries else []
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for policy_summary in summary.policy_summaries:
                writer.writerow(policy_summary.to_csv_row())
    return output


def write_decision_regret_rows(
    rows: Iterable[DecisionRegretRow],
    output_path: str | Path,
) -> Path:
    materialized = list(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(materialized[0].to_csv_row()) if materialized else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in materialized:
                writer.writerow(row.to_csv_row())
    return output
