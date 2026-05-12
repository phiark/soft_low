from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from frcnet.utils import content_entropy

SOFTMAX_CE_REFERENCE_FAMILY = "softmax_ce_reference"
MAX_SOFTMAX_PROBABILITY_SCORE_NAME = "max_softmax_probability"
PREDICTIVE_ENTROPY_SCORE_NAME = "predictive_entropy"
SOFTMAX_ENTROPY_SCORE_NAME = "softmax_entropy"
ENERGY_SCORE_NAME = "energy_score"
DEFAULT_SOFTMAX_REFERENCE_SCORE_NAMES = (SOFTMAX_ENTROPY_SCORE_NAME,)
SUPPORTED_SOFTMAX_REFERENCE_SCORE_NAMES = (
    MAX_SOFTMAX_PROBABILITY_SCORE_NAME,
    PREDICTIVE_ENTROPY_SCORE_NAME,
    SOFTMAX_ENTROPY_SCORE_NAME,
    ENERGY_SCORE_NAME,
)


@dataclass(slots=True)
class ReferenceScoreRecord:
    sample_id: str
    split_name: str
    cohort_name: str
    source_dataset_name: str
    reference_model_family: str
    reference_run_id: str
    reference_score_name: str
    reference_score_value: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "sample_id": self.sample_id,
            "split_name": self.split_name,
            "cohort_name": self.cohort_name,
            "source_dataset_name": self.source_dataset_name,
            "reference_model_family": self.reference_model_family,
            "reference_run_id": self.reference_run_id,
            "reference_score_name": self.reference_score_name,
            "reference_score_value": self.reference_score_value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReferenceScoreRecord":
        return cls(
            sample_id=str(payload["sample_id"]),
            split_name=str(payload["split_name"]),
            cohort_name=str(payload["cohort_name"]),
            source_dataset_name=str(payload["source_dataset_name"]),
            reference_model_family=str(payload["reference_model_family"]),
            reference_run_id=str(payload["reference_run_id"]),
            reference_score_name=str(payload["reference_score_name"]),
            reference_score_value=float(payload["reference_score_value"]),
        )


def softmax_entropy_reference_scores(logits: torch.Tensor) -> torch.Tensor:
    return predictive_entropy_reference_scores(logits)


def max_softmax_probability_reference_scores(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    return probabilities.max(dim=-1).values


def predictive_entropy_reference_scores(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    return content_entropy(probabilities)


def energy_score_reference_scores(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits, dim=-1)


def validate_softmax_reference_score_names(score_names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(str(score_name) for score_name in score_names)
    if not names:
        raise ValueError("At least one softmax CE reference score name is required.")
    seen: set[str] = set()
    for score_name in names:
        if score_name not in SUPPORTED_SOFTMAX_REFERENCE_SCORE_NAMES:
            supported = ", ".join(SUPPORTED_SOFTMAX_REFERENCE_SCORE_NAMES)
            raise ValueError(
                f"Unsupported softmax CE reference score '{score_name}'. "
                f"Supported scores: {supported}."
            )
        if score_name in seen:
            raise ValueError(
                f"Duplicate softmax CE reference score '{score_name}' is not allowed."
            )
        seen.add(score_name)
    return names


def resolve_softmax_reference_score_names(
    reference_config: Mapping[str, object],
) -> tuple[str, ...]:
    configured_names = reference_config.get("score_names")
    if configured_names is None:
        return validate_softmax_reference_score_names(
            [str(reference_config.get("score_name", SOFTMAX_ENTROPY_SCORE_NAME))]
        )
    if isinstance(configured_names, str):
        return validate_softmax_reference_score_names([configured_names])
    try:
        return validate_softmax_reference_score_names(
            str(score_name) for score_name in configured_names
        )
    except TypeError as exc:
        raise ValueError(
            "reference_train.score_names must be a string or iterable of strings."
        ) from exc


def compute_softmax_reference_scores(
    logits: torch.Tensor,
    score_names: Iterable[str] = DEFAULT_SOFTMAX_REFERENCE_SCORE_NAMES,
) -> dict[str, torch.Tensor]:
    resolved_score_names = validate_softmax_reference_score_names(score_names)
    probabilities: torch.Tensor | None = None
    entropy_scores: torch.Tensor | None = None
    scores: dict[str, torch.Tensor] = {}
    for score_name in resolved_score_names:
        if score_name == MAX_SOFTMAX_PROBABILITY_SCORE_NAME:
            if probabilities is None:
                probabilities = torch.softmax(logits, dim=-1)
            scores[score_name] = probabilities.max(dim=-1).values
        elif score_name in {PREDICTIVE_ENTROPY_SCORE_NAME, SOFTMAX_ENTROPY_SCORE_NAME}:
            if entropy_scores is None:
                if probabilities is None:
                    probabilities = torch.softmax(logits, dim=-1)
                entropy_scores = content_entropy(probabilities)
            scores[score_name] = entropy_scores
        elif score_name == ENERGY_SCORE_NAME:
            scores[score_name] = energy_score_reference_scores(logits)
    return scores


def build_reference_score_records(
    *,
    score_tensors: Mapping[str, torch.Tensor],
    sample_ids: Sequence[str],
    split_names: Sequence[str],
    cohort_names: Sequence[str],
    source_dataset_names: Sequence[str],
    reference_model_family: str,
    reference_run_id: str,
) -> list[ReferenceScoreRecord]:
    score_names = validate_softmax_reference_score_names(score_tensors.keys())
    sample_count = len(sample_ids)
    if not (
        len(split_names) == sample_count
        and len(cohort_names) == sample_count
        and len(source_dataset_names) == sample_count
    ):
        raise ValueError(
            "Reference score metadata sequences must have the same length as sample_ids."
        )
    score_values_by_name: dict[str, list[float]] = {}
    for score_name in score_names:
        score_tensor = score_tensors[score_name]
        if score_tensor.ndim != 1:
            raise ValueError(f"Reference score tensor '{score_name}' must be one-dimensional.")
        if int(score_tensor.shape[0]) != sample_count:
            raise ValueError(
                f"Reference score tensor '{score_name}' must contain one score per sample."
            )
        score_values_by_name[score_name] = [
            float(value) for value in score_tensor.detach().cpu().tolist()
        ]

    records: list[ReferenceScoreRecord] = []
    for index, sample_id in enumerate(sample_ids):
        for score_name in score_names:
            records.append(
                ReferenceScoreRecord(
                    sample_id=str(sample_id),
                    split_name=str(split_names[index]),
                    cohort_name=str(cohort_names[index]),
                    source_dataset_name=str(source_dataset_names[index]),
                    reference_model_family=reference_model_family,
                    reference_run_id=reference_run_id,
                    reference_score_name=score_name,
                    reference_score_value=score_values_by_name[score_name][index],
                )
            )
    return records


def write_reference_score_records(
    records: Iterable[ReferenceScoreRecord],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
    return output


def read_reference_score_records(input_path: str | Path) -> list[ReferenceScoreRecord]:
    records: list[ReferenceScoreRecord] = []
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(ReferenceScoreRecord.from_dict(json.loads(line)))
    return records
