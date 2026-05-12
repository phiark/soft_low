# ADR-0011: Plan A V1.0 Decision-Regret Paper Benchmark

- status: accepted
- date: 2026-05-11
- owner: frcnet_project
- related_protocol: `2026-05-11_v1.0_decision_regret_seen_unseen`

## Context

V0.6C is archived as clean partial evidence. It provides strict class-holdout
near-OOD evidence, source overlap checks, source-slice AUROC, and checkpoint
policy comparison. It still does not close the paper's strongest empirical gap:
the manuscript theorem is about decision blindness after scalar completion, but
the experiment primarily reports AUROC.

The V4 manuscript also needs journal-facing reproducibility and baseline
discipline: frozen manifests, seen/unseen source slices, per-source reporting,
and primary tables that exclude label-aware `proposition_truth_ratio` audit
numbers.

## Decision

Adopt V1.0 as a paper-facing decision-regret benchmark over the existing FRCNet
state contract. V1.0 reuses the V0.6C strict seen/unseen data split and adds a
new evaluation layer:

- decision regret is the primary theorem-aligned metric.
- AUROC remains secondary diagnostic context.
- pair-state policies must be compared against scalar completion policies and
  one-feature scalar policies.
- `oracle_state` is reported as a label-aware upper bound.
- `proposition_truth_ratio` remains appendix-only audit evidence.

## Scope

In scope:

1. Decision-regret evaluator over `SampleAnalysisRecord`.
2. Report artifacts for primary and source-slice regret.
3. Study aggregate tables for regret mean/std/min/max.
4. Softmax-reference baseline score export for multiple scalar scores on the
   same frozen manifest.
5. V1.0 protocol, study, eval, and analysis configs.

Out of scope:

- new FRCNet architecture.
- 4-way gate head.
- CLIP/DINO backbone.
- ODIN/Mahalanobis/ensemble training results unless separately implemented and
  run under the same frozen manifest.
- using `proposition_truth_ratio` as primary evidence.

## Consequences

V1.0 can support the manuscript only if the decision-regret benchmark is reported
even when the pair-state policy does not win. A negative or small result narrows
the empirical claim to diagnostic transparency rather than superiority.

