# FRCNet next-v0.1

FRCNet is a document-driven research codebase for an explicit-unknown model with a native resolution/content factorization:

```text
p_k = r * c_k
u = 1 - r
```

This branch is intentionally narrowed to the current `next-v0.1` work: semantic layer separation plus a fair matched benchmark skeleton. Historical `v0.3` study protocols, multi-seed aggregation, and debug checkpoint policy branches are not active in this branch.

## Active Scope

- `state layer`: `resolution_ratio`, `content_distribution`, `unknown_mass`, `state_content_entropy`, `state_weighted_content_entropy`, `state_entropy`
- `proposition layer`: proposition-specific `pT / pF / pU / tau_A`
- `completion layer`: `q_beta` readouts bound to a declared proposition view
- `matched benchmark`: label-free feature whitelist plus optional frozen matched manifest
- `softmax reference`: minimal same-backbone CE reference used only to build external reference scores

## Not In This Branch

- full baseline matrix
- multi-seed study workflow
- decision benchmark
- Transformer or teacher-distillation variants
- generated experiment artifacts or checkpoints

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Run tests:

```bash
python -m pytest -q
```

## Main Commands

Prepare CIFAR-10 / SVHN data:

```bash
python scripts/prepare_plan_a_data.py
```

Train the native FRCNet model:

```bash
python scripts/train_plan_a.py \
  --protocol-config configs/protocol/plan_a_next_v0_1_train.yaml \
  --model-config configs/model/frcnet_resnet18_base.yaml \
  --train-config configs/train/plan_a_train_base.yaml
```

Run a single end-to-end bundle:

```bash
python scripts/run_plan_a_experiment.py
```

Build an analysis manifest only:

```bash
python scripts/build_plan_a_manifest.py \
  --protocol-config configs/protocol/plan_a_next_v0_1_analysis.yaml
```

Generate report artifacts from an existing analysis export:

```bash
python scripts/generate_plan_a_artifacts.py \
  --analysis-path artifacts/reports/generated/RUN-LOCAL/sample_analysis_records.csv \
  --analysis-summary-path artifacts/reports/generated/RUN-LOCAL/analysis_summary.json \
  --protocol-config configs/protocol/plan_a_next_v0_1_analysis.yaml \
  --analysis-config configs/analysis/plan_a_next_v0_1_artifacts.yaml \
  --eval-config configs/eval/plan_a_next_v0_1_matched_manifest.yaml \
  --output-dir artifacts/reports/generated/RUN-LOCAL/report
```

## Current Documents

- [Document Index](docs/index.md)
- [Architecture Description](docs/architecture/architecture_description.md)
- [Plan A next-v0.1 Protocol](docs/architecture/plan_a_next_v0_1_protocol.md)
- [Plan A Paper Linkage](docs/architecture/plan_a_paper_linkage.md)
- [System Requirements](docs/requirements/system_requirements_specification.md)
- [Verification Plan](docs/verification/verification_and_validation_plan.md)
- [ADR-0006 next-v0.1 Semantic Repair](records/decisions/adr_0006_next_v0_1_semantic_repair.md)
