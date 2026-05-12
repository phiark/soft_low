# Plan A V1.0 Decision-Regret Paper Protocol

- document_id: arch_plan_a_v1_0_decision_regret_protocol
- status: draft
- owner: frcnet_project
- last_updated: 2026-05-11

## Purpose

V1.0 is the paper-facing experiment line after the V0.6C clean partial archive.
It does not introduce a new FRCNet architecture. It upgrades the evidence target
from AUROC-only diagnostics to a decision benchmark aligned with the manuscript
claim that a one-completion scalar can be decision-blind.

Primary question:

Can the native FRCNet state variables reduce downstream decision regret relative
to single scalar completion policies on the same strict frozen seen/unseen
manifest?

## Study

- study config: `configs/study/plan_a_v1_0_decision_regret_seen_unseen.yaml`
- base train protocol: `configs/protocol/plan_a_next_v0_6c_train.yaml`
- base validation protocol: `configs/protocol/plan_a_next_v0_6c_validation.yaml`
- final protocol: `configs/protocol/plan_a_next_v0_6c_test_cifar100_class_holdout.yaml`
- model config: `configs/model/frcnet_resnet18_source_invariant.yaml`
- train config: `configs/train/plan_a_next_v0_6c_near_ood_weighted_curriculum.yaml`
- eval config: `configs/eval/plan_a_v1_0_decision_regret.yaml`
- analysis config: `configs/analysis/plan_a_v1_0_decision_regret_artifacts.yaml`
- reference config: `configs/reference/plan_a_v1_0_softmax_ce_reference.yaml`
- seeds: `[7, 17, 27]`

The reuse of V0.6C data protocols is deliberate: V1.0 changes the paper-facing
evaluation contract, not the archived model/data split. Any architecture change
belongs to a later version plan.

## Decision Benchmark

The benchmark is offline and uses exported `SampleAnalysisRecord` rows. It
compares policies on a held-out split from the frozen matched manifest.

Default actions:

- `accept_known`: act as if the input is resolved known content.
- `defer`: request more information for ambiguous known content.
- `reject_unknown`: treat the input as outside the known scope.

Default oracle action by cohort:

- `easy_id`, `hard_id`: `accept_known`
- `ambiguous_id`: `defer`
- `ood`, `unknown_supervision`: `reject_unknown`

Default utility gives `1.0` to the oracle action and `0.0` otherwise. The
primary regret is:

```text
regret = U(a_star | sigma) - U(a_policy | sigma)
```

Required policy families:

- `q_beta`: `top1_completion_beta_0_1`, `top1_completion_beta_0_25`,
  `top1_completion_beta_0_5`, `top1_completion_beta_0_75`
- `resolution_ratio`
- `state_content_entropy`
- `state_weighted_content_entropy`
- pair: `(resolution_ratio, state_content_entropy)`
- weighted pair: `(resolution_ratio, state_weighted_content_entropy)`
- `oracle_state`

## Evidence Contract

V1.0 must report:

- `mean_regret`
- `median_regret`
- `policy_action_accuracy`
- `worst_source_mean_regret`
- `seen_unseen_regret_gap`
- AUROC as secondary context only

Required source slices:

- `ambiguous_vs_seen_ood_svhn`
- `ambiguous_vs_seen_ood_tiny_imagenet`
- `ambiguous_vs_seen_ood_cifar100_seen_classes`
- `ambiguous_vs_unseen_ood_cifar100_heldout_classes`
- `ambiguous_vs_all_ood`

The proposition `tau` diagnostic is appendix-only evidence. It must not appear
in the primary matched table or decision-regret comparison because its semantics
are label-aware and too close to the cohort construction.

## Release Gate

V1.0 is manuscript-facing only if:

- all required frozen matched manifests are present.
- source overlap audit remains zero.
- decision-regret tables are emitted for the primary benchmark and source slices.
- `oracle_state` has zero mean regret.
- at least one pair-state policy is compared against every scalar policy.
- negative results are preserved rather than hidden.
