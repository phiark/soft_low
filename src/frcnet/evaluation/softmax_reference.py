from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from frcnet.data import BatchInput
from frcnet.models.backbones import build_backbone
from frcnet.utils import RuntimeSpec, move_batch_to_device

SOFTMAX_REFERENCE_FAMILY = "softmax_ce_reference"


class SoftmaxReferenceModel(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str = "resnet18") -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.num_classes = int(num_classes)
        self.backbone_name = backbone_name
        self.backbone, feature_dim = build_backbone(backbone_name)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(image_batch))


@dataclass(frozen=True, slots=True)
class ReferenceScoreRecord:
    sample_id: str
    reference_score_name: str
    reference_score_value: float
    model_family: str = SOFTMAX_REFERENCE_FAMILY

    def to_csv_row(self) -> dict[str, str | float]:
        return {
            "sample_id": self.sample_id,
            "reference_score_name": self.reference_score_name,
            "reference_score_value": self.reference_score_value,
            "model_family": self.model_family,
        }


def _softmax_entropy(probability: torch.Tensor) -> torch.Tensor:
    safe_probability = probability.clamp_min(torch.finfo(probability.dtype).eps)
    return -(safe_probability * torch.log(safe_probability)).sum(dim=1)


def reference_scores_from_logits(
    logits: torch.Tensor,
    sample_ids: list[str],
    *,
    score_name: str = "softmax_entropy",
) -> list[ReferenceScoreRecord]:
    if logits.ndim != 2:
        raise ValueError("logits must be a 2D tensor.")
    if len(sample_ids) != int(logits.shape[0]):
        raise ValueError("sample_ids must be aligned with logits.")
    probability = torch.softmax(logits, dim=1)
    if score_name == "softmax_entropy":
        values = _softmax_entropy(probability)
    elif score_name == "softmax_max_probability":
        values = probability.max(dim=1).values
    else:
        raise ValueError("Unsupported score_name. Use softmax_entropy or softmax_max_probability.")
    return [
        ReferenceScoreRecord(
            sample_id=sample_id,
            reference_score_name=score_name,
            reference_score_value=float(value.item()),
        )
        for sample_id, value in zip(sample_ids, values, strict=True)
    ]


def run_softmax_reference_train_epoch(
    model: SoftmaxReferenceModel,
    dataloader: DataLoader,
    optimizer: Optimizer,
    runtime_spec: RuntimeSpec,
) -> dict[str, float | int]:
    model.to(runtime_spec.device)
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_input in dataloader:
        batch_on_device = move_batch_to_device(batch_input, runtime_spec)
        id_mask = batch_on_device.class_label.ge(0)
        if not bool(id_mask.any()):
            continue
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_on_device.image[id_mask])
        loss = torch.nn.functional.cross_entropy(logits, batch_on_device.class_label[id_mask].long())
        loss.backward()
        optimizer.step()
        batch_count = int(id_mask.sum().item())
        total_loss += float(loss.item()) * batch_count
        total_samples += batch_count
    mean_loss = 0.0 if total_samples == 0 else total_loss / total_samples
    return {"mean_loss": mean_loss, "num_trainable_samples": total_samples}


def run_softmax_reference_export(
    model: SoftmaxReferenceModel,
    dataloader: DataLoader,
    runtime_spec: RuntimeSpec,
    *,
    score_name: str = "softmax_entropy",
) -> list[ReferenceScoreRecord]:
    model.to(runtime_spec.device)
    model.eval()
    records: list[ReferenceScoreRecord] = []
    with torch.no_grad():
        for batch_input in dataloader:
            batch_on_device = move_batch_to_device(batch_input, runtime_spec)
            logits = model(batch_on_device.image)
            records.extend(reference_scores_from_logits(logits, batch_on_device.sample_id, score_name=score_name))
    return records


def write_reference_score_records(records: list[ReferenceScoreRecord], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(records[0].to_csv_row().keys()) if records else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_csv_row())
    return output


def read_reference_score_records(input_path: str | Path) -> list[ReferenceScoreRecord]:
    records: list[ReferenceScoreRecord] = []
    with Path(input_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                ReferenceScoreRecord(
                    sample_id=row["sample_id"],
                    reference_score_name=row["reference_score_name"],
                    reference_score_value=float(row["reference_score_value"]),
                    model_family=row.get("model_family", SOFTMAX_REFERENCE_FAMILY),
                )
            )
    return records
