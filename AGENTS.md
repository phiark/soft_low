# Repository Guidelines

## Project Structure & Module Organization

FRCNet is document-driven. Treat `docs/` as the normative baseline, `records/` as evidence, `configs/` as runnable experiment contracts, `src/frcnet/` as implementation, `tests/` as verification, `scripts/` as repeatable entrypoints, and `artifacts/` as generated output. Do not add temporary roots such as `misc/`, `tmp_work/`, `new_files/`, `artifact/`, or misspelled study folders.

Apply a measure-twice, change-once policy: read the baseline document and matching config before editing implementation or records.

## Build, Test, and Development Commands

Use the validated local interpreter when available:

```bash
.venv313/bin/python -m pytest -q
.venv313/bin/python scripts/run_plan_a_study.py --study-config configs/study/plan_a_next_v0_6c_near_ood_cifar100_class_holdout.yaml
.venv313/bin/python scripts/cleanup_checkpoints.py
```

Do not introduce Docker or container workflows unless the user explicitly requests them. New training or scientific claims require a new version plan before running experiments. Avoid redundant workflow prose; link to the existing script, config, or protocol instead.

## Coding Style & Naming Conventions

Python uses 4-space indentation, `snake_case` for modules/functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Use canonical variables from `docs/governance/naming_and_identifier_standard.md`: `resolution_ratio`, `content_distribution`, `unknown_mass`, `state_content_entropy`, and `proposition_truth_ratio`. Avoid long-lived paper-symbol variables such as `r`, `u`, `c`, `tau`, or `S` outside formulas and short local expressions.

## Artifact & Study Naming

Generated study roots belong under `artifacts/studies/`. New roots must use `YYYY-MM-DD_vX.Y_slug`, for example `artifacts/studies/2026-05-11_v0.7_near_ood_repair`. Keep legacy roots only for traceability; do not create new `RUN-*`, `KYUC-*`, `artifact/`, or `stuidie/` paths.

## Testing Guidelines

Add or update focused tests when documentation, config, or workflow contracts change. Prefer contract tests for schemas and provenance, integration tests for workflow assembly, and unit tests for math invariants. A normal maintenance check should not stage generated `artifacts/` files.

## Commit & Pull Request Guidelines

Keep commits scoped by document, config, code, or evidence category. PRs should state the baseline document, changed config paths, validation command, and whether the change is maintenance-only or starts a new version plan.
