# Scripts

This directory contains only active `next-v0.1` entrypoints.

- `prepare_plan_a_data.py`: verify or download CIFAR-10 / SVHN assets.
- `build_plan_a_manifest.py`: build an analysis manifest snapshot.
- `train_plan_a.py`: train FRCNet from a manifest-backed protocol.
- `run_plan_a_inference.py`: export sample analysis records and proposition records.
- `generate_plan_a_artifacts.py`: generate figures, tables, and an experiment record.
- `run_plan_a_experiment.py`: run the single-run train -> inference -> report bundle.

The multi-seed study and aggregate scripts are intentionally absent from this branch.
