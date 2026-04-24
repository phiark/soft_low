from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from frcnet.evaluation.records import SampleAnalysisRecord


@dataclass(frozen=True, slots=True)
class MatchedManifestRecord:
    sample_id: str
    cohort_name: str
    source_dataset_name: str
    source_index: str
    reference_score_name: str
    reference_score_value: float
    match_bin_id: str
    paired_group_id: str
    manifest_role: str
    construction_config_hash: str
    manifest_hash: str = ""

    def to_dict(self) -> dict[str, str | float]:
        return {
            "sample_id": self.sample_id,
            "cohort_name": self.cohort_name,
            "source_dataset_name": self.source_dataset_name,
            "source_index": self.source_index,
            "reference_score_name": self.reference_score_name,
            "reference_score_value": self.reference_score_value,
            "match_bin_id": self.match_bin_id,
            "paired_group_id": self.paired_group_id,
            "manifest_role": self.manifest_role,
            "construction_config_hash": self.construction_config_hash,
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MatchedManifestRecord":
        return cls(
            sample_id=str(payload["sample_id"]),
            cohort_name=str(payload["cohort_name"]),
            source_dataset_name=str(payload["source_dataset_name"]),
            source_index=str(payload.get("source_index", "")),
            reference_score_name=str(payload["reference_score_name"]),
            reference_score_value=float(payload["reference_score_value"]),
            match_bin_id=str(payload["match_bin_id"]),
            paired_group_id=str(payload["paired_group_id"]),
            manifest_role=str(payload.get("manifest_role", "eval")),
            construction_config_hash=str(payload["construction_config_hash"]),
            manifest_hash=str(payload.get("manifest_hash", "")),
        )


def _stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compute_construction_config_hash(config: Mapping[str, object]) -> str:
    return hashlib.sha256(_stable_json(dict(config)).encode("utf-8")).hexdigest()


def compute_manifest_hash(records: Iterable[MatchedManifestRecord]) -> str:
    rows = []
    for record in records:
        payload = record.to_dict()
        payload["manifest_hash"] = ""
        rows.append(payload)
    return hashlib.sha256(_stable_json(rows).encode("utf-8")).hexdigest()


def build_frozen_matched_manifest(
    records: list[SampleAnalysisRecord],
    *,
    reference_scores: Mapping[str, float],
    reference_score_name: str,
    positive_cohort: str = "ambiguous_id",
    negative_cohort: str = "ood",
    num_bins: int = 10,
    manifest_role: str = "eval",
    construction_config_hash: str | None = None,
) -> list[MatchedManifestRecord]:
    if positive_cohort == negative_cohort:
        raise ValueError("positive_cohort and negative_cohort must be different.")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    cohort_records = [
        record
        for record in records
        if record.cohort_name in {positive_cohort, negative_cohort} and record.sample_id in reference_scores
    ]
    if len({record.cohort_name for record in cohort_records}) < 2:
        raise ValueError("Matched manifest requires scored records from both cohorts.")

    values = np.array([float(reference_scores[record.sample_id]) for record in cohort_records], dtype=np.float64)
    if float(values.max()) == float(values.min()):
        bin_edges = np.linspace(float(values.min()), float(values.max()) + 1e-6, num_bins + 1)
    else:
        bin_edges = np.linspace(float(values.min()), float(values.max()), num_bins + 1)

    binned: dict[tuple[str, int], list[SampleAnalysisRecord]] = defaultdict(list)
    for record in cohort_records:
        score = float(reference_scores[record.sample_id])
        bin_index = int(np.searchsorted(bin_edges, score, side="right") - 1)
        bin_index = min(max(bin_index, 0), num_bins - 1)
        binned[(record.cohort_name, bin_index)].append(record)

    config_hash = construction_config_hash or compute_construction_config_hash(
        {
            "positive_cohort": positive_cohort,
            "negative_cohort": negative_cohort,
            "reference_score_name": reference_score_name,
            "num_bins": num_bins,
            "manifest_role": manifest_role,
        }
    )
    matched_records: list[MatchedManifestRecord] = []
    for bin_index in range(num_bins):
        positive_records = sorted(
            binned[(positive_cohort, bin_index)],
            key=lambda record: (float(reference_scores[record.sample_id]), record.sample_id),
        )
        negative_records = sorted(
            binned[(negative_cohort, bin_index)],
            key=lambda record: (float(reference_scores[record.sample_id]), record.sample_id),
        )
        pair_count = min(len(positive_records), len(negative_records))
        for pair_index in range(pair_count):
            paired_group_id = f"bin{bin_index:02d}-pair{pair_index:05d}"
            for record in (positive_records[pair_index], negative_records[pair_index]):
                matched_records.append(
                    MatchedManifestRecord(
                        sample_id=record.sample_id,
                        cohort_name=record.cohort_name,
                        source_dataset_name=record.source_dataset_name,
                        source_index=record.sample_id,
                        reference_score_name=reference_score_name,
                        reference_score_value=float(reference_scores[record.sample_id]),
                        match_bin_id=f"bin{bin_index:02d}",
                        paired_group_id=paired_group_id,
                        manifest_role=manifest_role,
                        construction_config_hash=config_hash,
                    )
                )

    if not matched_records:
        raise ValueError("No matched records were produced. Check reference scores and binning.")
    manifest_hash = compute_manifest_hash(matched_records)
    return [
        MatchedManifestRecord(
            sample_id=record.sample_id,
            cohort_name=record.cohort_name,
            source_dataset_name=record.source_dataset_name,
            source_index=record.source_index,
            reference_score_name=record.reference_score_name,
            reference_score_value=record.reference_score_value,
            match_bin_id=record.match_bin_id,
            paired_group_id=record.paired_group_id,
            manifest_role=record.manifest_role,
            construction_config_hash=record.construction_config_hash,
            manifest_hash=manifest_hash,
        )
        for record in matched_records
    ]


def summarize_matched_manifest(records: Iterable[MatchedManifestRecord]) -> dict[str, object]:
    materialized = list(records)
    cohort_counts: dict[str, int] = defaultdict(int)
    bin_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in materialized:
        cohort_counts[record.cohort_name] += 1
        bin_counts[record.match_bin_id][record.cohort_name] += 1
    return {
        "manifest_hash": materialized[0].manifest_hash if materialized else "",
        "construction_config_hash": materialized[0].construction_config_hash if materialized else "",
        "reference_score_name": materialized[0].reference_score_name if materialized else "",
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "bin_counts": {bin_id: dict(sorted(counts.items())) for bin_id, counts in sorted(bin_counts.items())},
    }


def write_matched_manifest_jsonl(records: Iterable[MatchedManifestRecord], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
    return output


def read_matched_manifest_jsonl(input_path: str | Path) -> list[MatchedManifestRecord]:
    matched_records: list[MatchedManifestRecord] = []
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                matched_records.append(MatchedManifestRecord.from_dict(json.loads(line)))
    return matched_records


def write_matched_manifest_summary(records: Iterable[MatchedManifestRecord], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summarize_matched_manifest(records), indent=2, sort_keys=True), encoding="utf-8")
    return output
