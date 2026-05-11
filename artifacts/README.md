# Artifacts

Generated outputs live here.

- `figures/`: plots for analysis and paper figures
- `tables/`: CSV or markdown tables
- `reports/`: generated report bundles
- `checkpoints/`: model checkpoints
- `studies/`: generated multi-run study bundles
- `logs/`: training and evaluation logs

Artifacts are derived outputs, not the source of truth.

New study roots should use `YYYY-MM-DD_vX.Y_slug`, for example `studies/2026-05-11_v0.7_near_ood_repair/`. Existing legacy roots are retained only for traceability and should not be copied as naming templates.
