#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

from _bootstrap import bootstrap_repo

REPO_ROOT = bootstrap_repo(configure_runtime_cache=True)

from frcnet.data import ManifestBackedVisionDataset, collate_manifest_samples, load_plan_a_source_datasets
from frcnet.data import read_manifest_jsonl
from frcnet.evaluation.reference_baselines import (
    ReferenceScoreRecord,
    SOFTMAX_CE_REFERENCE_FAMILY,
    build_reference_score_records,
    compute_softmax_reference_scores,
    resolve_softmax_reference_score_names,
    write_reference_score_records,
)
from frcnet.models import SoftmaxReferenceModel
from frcnet.utils import resolve_pin_memory, resolve_runtime


def _load_yaml_section(path: str | Path, section_name: str) -> dict[str, Any]:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8"))[section_name])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export softmax CE reference scores for a Plan A manifest.")
    parser.add_argument("--protocol-config", default="configs/protocol/plan_a_next_v0_2_test.yaml")
    parser.add_argument("--model-config", default="configs/model/frcnet_resnet18_base.yaml")
    parser.add_argument("--reference-config", default="configs/reference/plan_a_next_v0_4_softmax_ce_reference.yaml")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="softmax_ce_reference-final-test")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_config = _load_yaml_section(args.protocol_config, "protocol")
    model_config = _load_yaml_section(args.model_config, "model")
    reference_config = _load_yaml_section(args.reference_config, "reference_train")
    score_names = resolve_softmax_reference_score_names(reference_config)
    runtime_config = dict(reference_config.get("runtime", {}))
    runtime_spec = resolve_runtime(
        requested_backend=runtime_config.get("backend", "auto"),
        dtype=runtime_config.get("dtype", "float32"),
        amp_enabled=bool(runtime_config.get("amp_enabled", False)),
    )
    dataloader_config = dict(reference_config.get("inference_dataloader", reference_config.get("dataloader", {})))
    source_datasets = load_plan_a_source_datasets(protocol_config)
    manifest_records = read_manifest_jsonl(args.manifest_path)
    dataset = ManifestBackedVisionDataset(manifest_records, source_datasets, int(protocol_config["num_classes"]))
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or dataloader_config.get("batch_size", 128)),
        shuffle=False,
        drop_last=False,
        num_workers=int(dataloader_config.get("num_workers", 0)),
        collate_fn=collate_manifest_samples,
        pin_memory=resolve_pin_memory(dataloader_config.get("pin_memory", "auto"), runtime_spec),
    )

    model = SoftmaxReferenceModel(
        num_classes=int(model_config["num_classes"]),
        backbone_name=str(model_config.get("backbone", "resnet18")),
    )
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(runtime_spec.device)
    model.eval()

    records: list[ReferenceScoreRecord] = []
    with torch.no_grad():
        for batch_input in dataloader:
            image_batch = batch_input.image.to(runtime_spec.device, dtype=runtime_spec.dtype)
            logits = model(image_batch)
            score_tensors = compute_softmax_reference_scores(logits, score_names)
            records.extend(
                build_reference_score_records(
                    score_tensors=score_tensors,
                    sample_ids=batch_input.sample_id,
                    split_names=batch_input.split_name,
                    cohort_names=batch_input.cohort_name,
                    source_dataset_names=batch_input.source_dataset_name,
                    reference_model_family=SOFTMAX_CE_REFERENCE_FAMILY,
                    reference_run_id=args.run_id,
                )
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = write_reference_score_records(
        records,
        output_dir / "reference_score_records.jsonl",
    )
    summary_path = output_dir / "reference_score_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "model_family": SOFTMAX_CE_REFERENCE_FAMILY,
                "reference_score_name": score_names[0],
                "reference_score_names": list(score_names),
                "reference_score_path": str(score_path),
                "record_count": len(records),
                "checkpoint_path": str(args.checkpoint_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(score_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
