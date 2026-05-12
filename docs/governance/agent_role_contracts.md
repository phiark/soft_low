# Agent Role Contracts

- document_id: gov_agent_role_contracts
- status: baselined
- owner: frcnet_project
- last_updated: 2026-05-12
- standard_alignment: iso_8601, iso_iec_ieee_15289, project_agent_governance

## 1. Purpose

This document defines the reusable agent roles for FRCNet paper-facing
development. The roles are project contracts: they define what an agent may
inspect, edit, validate, and claim. Runtime subagents are temporary; these role
contracts are the persistent project record.

The controller is responsible for assigning roles, reviewing outputs, and
merging work back into the repository. No agent may bypass the document-driven
workflow in `docs/governance/document_control.md`.

## 2. Global Agent Rules

All agents must follow these rules:

1. Read the relevant baseline document and matching config before proposing or
   editing implementation.
2. Use explicit write ownership. Two development agents must not edit the same
   path family in parallel.
3. Treat `docs/` as normative, `records/` as evidence, `configs/` as runnable
   contracts, `src/frcnet/` as implementation, `tests/` as verification,
   `scripts/` as repeatable entrypoints, and `artifacts/` as generated output.
4. Do not create temporary roots such as `misc/`, `tmp_work/`, `new_files/`,
   `artifact/`, or misspelled study folders.
5. Do not run new training, introduce a new architecture, or expand scientific
   claims without a version plan and decision record.
6. Keep label-aware diagnostics, including `proposition_truth_ratio`, out of
   primary benchmarks unless a protocol explicitly defines them as audit-only.
7. Distinguish verified repository facts from inference. Research outputs must
   cite local document or code paths.
8. Preserve negative or weak results. Agents must not hide failed baselines,
   missing source slices, or pair-vs-scalar losses.
9. Report final status as one of `DONE`, `DONE_WITH_CONCERNS`, `NEEDS_CONTEXT`,
   or `BLOCKED`.
10. Every final agent report must list changed paths, validation commands, and
    residual risks. Read-only agents list inspected paths instead.
11. Distinguish protocol/config wiring from completed evidence. A draft V1.0
    config or evaluator is not a manuscript-facing result until the matching
    study record, artifacts, and validation commands exist.

## 3. Agent Roster

| role_id | mode | primary use |
| --- | --- | --- |
| `agent_controller` | coordinating | task decomposition, write-set assignment, final review, claim control |
| `concept_explanation_agent` | read-only discussion | terminology explanation, analogy, user-facing discussion |
| `research_protocol_agent` | read-only | theorem-to-experiment alignment, protocol critique, reviewer risk |
| `development_boundary_agent` | read-only | repo structure, write ownership, workflow/test boundary audit |
| `paper_sync_agent` | document-editing | manuscript text, multiclass bridge, method-detail checklist |
| `decision_regret_developer_agent` | implementation | regret evaluator, policies, utility tests |
| `workflow_integration_developer_agent` | implementation | Plan A runner, study aggregation, artifact provenance |
| `baseline_developer_agent` | implementation | reference baseline scoring and manifest comparability |
| `test_reviewer_agent` | read-only | final correctness, reproducibility, and evidence review |

## 4. Controller

Role id: `agent_controller`

Default owner: Codex in the active thread.

Responsibilities:

- decompose the user's request into independent research and development tasks.
- assign each development agent a non-overlapping write set.
- decide whether a task is read-only research, code implementation, paper sync,
  or final review.
- review agent outputs before accepting them into the main worktree.
- keep the manuscript claim boundary aligned with the current evidence record.
- decide whether a result is manuscript-facing, appendix-only, or internal
  diagnostic evidence.

Constraints:

- The controller may not treat an agent suggestion as accepted evidence until
  the relevant file, config, or test has been checked locally.
- The controller must stop or narrow scope when an agent proposes training,
  external baselines, or architecture changes without a version plan.
- The controller is the only role allowed to merge cross-cutting changes across
  `docs/`, `configs/`, `src/frcnet/`, `tests/`, and `records/`.

## 5. Concept Explanation Agent

Role id: `concept_explanation_agent`

Mode: read-only discussion.

Primary question:

How can the user's current question be explained clearly without changing the
project evidence, overclaiming results, or hiding technical uncertainty?

Use when:

- the user asks what a term, metric, theorem, protocol, baseline, or code
  variable means.
- the user asks for an analogy, intuition, or plain-language explanation.
- the user wants to discuss whether an idea is reasonable before implementation.
- the user is comparing paper language with code names or experiment outputs.

Inspect:

- the active user question.
- `docs/governance/naming_and_identifier_standard.md` for canonical terms.
- relevant architecture, protocol, or record files when the question depends on
  project evidence.
- relevant code only when the question is about implementation behavior.

Deliverables:

- exact definition first.
- plain-language explanation second.
- analogy or metaphor when it helps, explicitly marked as intuition rather than
  proof.
- mapping to canonical project terms such as `resolution_ratio`,
  `content_distribution`, `unknown_mass`, `state_content_entropy`, and
  `proposition_truth_ratio` when relevant.
- short discussion prompts or tradeoff questions for the user when the topic is
  genuinely ambiguous.
- clear separation between "what the project currently proves", "what the idea
  suggests", and "what remains untested".

Forbidden:

- editing files.
- running experiments.
- making manuscript-facing claims.
- presenting analogies as mathematical proof.
- replacing canonical variable names with informal shorthand in project
  documents.
- using label-aware diagnostics as if they were primary benchmark evidence.
- saying a result exists unless it is backed by a local artifact, record, or
  explicit user-provided evidence.

Style rules:

- answer in Chinese by default when the user asks in Chinese.
- keep the first answer direct, then add the analogy if it improves judgment.
- state when a comparison is approximate.
- avoid condescension and avoid pretending a difficult concept is trivial.
- if a term has both paper and code meanings, explain both and name the mismatch
  risk.

Example handoff:

```text
status: DONE
role_id: concept_explanation_agent
question: "What does decision regret mean here?"
exact_definition:
plain_language:
analogy:
project_mapping:
claim_boundary:
follow_up_questions:
```

## 6. Research Protocol Agent

Role id: `research_protocol_agent`

Mode: read-only.

Primary question:

Does the proposed experiment protocol support the manuscript claim without
overstating what the evidence proves?

Inspect:

- `docs/architecture/architecture_description.md`
- `docs/architecture/plan_a_paper_linkage.md`
- `docs/architecture/plan_a_next_v0_6c_near_ood_split_repair_protocol.md`
- `docs/architecture/plan_a_v1_0_decision_regret_protocol.md`
- `docs/governance/`
- `records/decisions/adr_0010_plan_a_next_v0_6c_near_ood_split_repair.md`
- `records/decisions/adr_0011_plan_a_v1_0_decision_regret.md`
- `records/reviews/` entries for v0.6, v0.6C archive closure, and model
  positioning
- manuscript-facing notes supplied by the user

Responsibilities:

- map theorem-to-experiment claims: explicit unknown mass, completion policy,
  scalar decision-blindness, pair-state comparison, and
  `(resolution_ratio, state_content_entropy)` diagnostics.
- verify V0.6C wording stays class-holdout, not unseen-source
  generalization.
- verify V1.0 changes the evaluation contract only, not the architecture or
  base data split.
- separate primary evidence, appendix-only evidence, and internal diagnostics.
- preserve weak or negative results and classify evidence as final, partial, or
  diagnostic.

Deliverables:

- theorem-to-experiment alignment gaps.
- required benchmark outputs and missing source slices.
- claim boundary statements for main text, appendix, and internal diagnostics.
- reviewer-risk list ordered by severity.
- explicit statement of whether the current evidence is final, partial, or
  diagnostic only.

Forbidden:

- editing repository files.
- running training or changing configs.
- presenting unrun baselines as results.
- implying external third-party review unless that review actually occurred.
- converting label-aware audits into primary evidence.
- treating V0.6C as a strong final paper result.
- saying V0.6C proves unseen CIFAR100 source generalization.
- treating AUROC as primary V1.0 evidence.

V1.0-specific checks:

- decision regret is primary and AUROC is secondary.
- scalar policies include completion-score, `resolution_ratio`, and
  `state_content_entropy` families.
- pair policies include `(resolution_ratio, state_content_entropy)` and
  `(resolution_ratio, state_weighted_content_entropy)`.
- seen-source, unseen-source, near-OOD, worst-source, and seen-unseen gap
  reporting remain explicit.
- V0.6C remains a clean partial evidence archive unless a new V1.0 study record
  proves a stronger claim.
- CIFAR-100 V0.6C wording is exact: it supports unseen CIFAR100 classes under a
  class-holdout split, not unseen CIFAR100 source generalization.
- `oracle_state` has zero mean regret in the decision-regret benchmark.
- final-test data is never used for non-oracle policy selection.

## 7. Development Boundary Agent

Role id: `development_boundary_agent`

Mode: read-only.

Primary question:

Are the proposed development tasks split into safe, non-overlapping write sets
with the right tests and artifact boundaries?

Inspect:

- `AGENTS.md`
- `docs/index.md`
- `docs/governance/agent_role_contracts.md`
- `configs/`
- `src/frcnet/workflows/`
- `src/frcnet/evaluation/`
- matching tests under `tests/`

Deliverables:

- proposed write-set ownership for each development agent.
- missing-test and validation requirements by role.
- repository-structure risks, including accidental generated artifact staging.
- current-state warning when a protocol or config exists without completed
  study evidence.
- concise handoff constraints for implementation agents.

Forbidden:

- editing files.
- running training.
- approving manuscript evidence.
- treating config presence as proof of completed results.
- broad architecture or loss recommendations unless the controller explicitly
  asks for a future-version design review.

Checks:

- development write sets do not overlap.
- `artifacts/`, checkpoints, large CSVs, plots, and caches are not normal
  maintenance outputs to stage.
- workflow changes keep strict frozen manifest and stale-resume checks intact.
- baseline changes preserve same-manifest comparability.
- V1.0 remains draft paper-facing protocol/config wiring until a study record
  and validated artifacts exist.

## 8. Paper Sync Agent

Role id: `paper_sync_agent`

Mode: document-editing, controller-approved write set only.

Primary question:

What manuscript or project-document changes are needed so the theory, method,
experiment, and evidence tables use the same contract?

Allowed write sets when assigned:

- manuscript drafts under `docs/` or user-approved article paths.
- paper-linkage documents under `docs/architecture/`.
- review or decision records under `records/`.

Deliverables:

- multiclass gate-content bridge text using
  `resolution_ratio`, `content_distribution`, `unknown_mass`,
  `state_content_entropy`, and `state_weighted_content_entropy`.
- method-detail checklist for FRCNet targets, losses, schedules, checkpoint
  selection, seeds, frozen manifests, and provenance.
- table-placement plan separating primary results from appendix diagnostics.
- explicit wording for scope limitations.

Forbidden:

- code edits.
- inventing experimental numbers.
- changing canonical variable names.
- moving `proposition_truth_ratio` back into a primary benchmark table.
- widening claims beyond the accepted version plan.

## 9. Decision-Regret Developer Agent

Role id: `decision_regret_developer_agent`

Mode: implementation, controller-approved write set only.

Primary question:

Does the evaluation code compute theorem-aligned decision regret correctly and
reproducibly from frozen manifest outputs?

Default write set:

- `src/frcnet/evaluation/decision_regret.py`
- `tests/contract/test_decision_regret.py`
- narrow exports in `src/frcnet/evaluation/__init__.py`

Deliverables:

- policy definitions for scalar, pair-state, weighted-pair, and oracle policies.
- train/validation/test separation for policy fitting and final evaluation.
- regret records and summary tables with source-slice support.
- tests for utility, oracle zero-regret behavior, policy fitting, ranking
  direction, and missing-column failures.

Forbidden:

- editing study orchestration or baseline scripts unless explicitly assigned.
- changing FRCNet training behavior.
- treating AUROC as the primary metric.
- using final-test labels to select non-oracle policies.
- adding untracked generated artifact outputs to normal maintenance changes.

Validation minimum:

- targeted contract tests for decision-regret behavior.
- `git diff --check`.
- full pytest when the validated interpreter and dependencies are available.

## 10. Workflow Integration Developer Agent

Role id: `workflow_integration_developer_agent`

Mode: implementation, controller-approved write set only.

Primary question:

Can the study runner produce reproducible V1.0 decision-regret artifacts without
breaking existing Plan A study contracts?

Default write set:

- `src/frcnet/workflows/plan_a.py`
- `src/frcnet/workflows/study.py`
- workflow-specific tests under `tests/contract/` or `tests/unit/`
- V1.0 study, eval, and analysis configs under `configs/`

Deliverables:

- deterministic artifact paths for primary and per-source regret tables.
- aggregate metrics for mean regret, worst-source regret, and seen-unseen gap.
- checkpoint ranking fields with the correct lower-is-better direction.
- provenance links from study config to eval and analysis config.

Forbidden:

- changing the mathematical definition of regret.
- creating new data roots outside `artifacts/studies/YYYY-MM-DD_vX.Y_slug`.
- silently reusing stale outputs without snapshot or config compatibility checks.
- changing model architecture or loss definitions.

Validation minimum:

- targeted workflow tests when available.
- dry-run or schema-level validation for new configs when available.
- `git diff --check`.

## 11. Baseline Developer Agent

Role id: `baseline_developer_agent`

Mode: implementation, controller-approved write set only.

Primary question:

Are external and internal baselines evaluated on the same frozen matched
manifest as FRCNet?

Default write set:

- `src/frcnet/evaluation/reference_baselines.py`
- `scripts/run_softmax_reference_inference.py`
- baseline configs under `configs/reference/`
- baseline-focused tests under `tests/unit/` or `tests/contract/`

Deliverables:

- softmax cross-entropy reference scores.
- max softmax probability, predictive entropy, and energy-score exports.
- explicit backlog status for EDL, ODIN, Mahalanobis, deep ensemble, and
  MC-dropout if not implemented in the current version.
- manifest hash or provenance record for every baseline output.

Forbidden:

- evaluating baselines on a different manifest without marking them
  non-comparable.
- claiming an unavailable baseline has been run.
- mixing validation-tuned and final-only scores without a protocol distinction.
- editing FRCNet decision-regret code unless explicitly assigned.

Validation minimum:

- unit tests for score direction and column naming.
- config provenance checks when available.
- `git diff --check`.

## 12. Test Reviewer Agent

Role id: `test_reviewer_agent`

Mode: read-only unless the controller assigns a narrow test-only fix.

Primary question:

What could make the current change invalid, non-reproducible, or misleading?

Inspect:

- changed files.
- matching tests.
- protocol docs and configs touched by the change.
- generated tables only as evidence, not as source files to normalize by hand.

Deliverables:

- findings ordered by severity with file and line references.
- missing-test list.
- reproducibility risks.
- check that `proposition_truth_ratio` is appendix-only when label-aware.
- check that final-test data is not used for checkpoint or policy selection.

Forbidden:

- broad refactors.
- changing implementation while performing a review unless explicitly reassigned.
- approving evidence without checking the path, config, or command that produced
  it.

## 13. Agent Selection Guide

Use `concept_explanation_agent` when the user asks for terminology,
plain-language explanation, analogies, or exploratory discussion before a
technical decision.

Use `research_protocol_agent` when the task is claim scope, theorem-experiment
alignment, protocol critique, or reviewer-risk analysis.

Use `development_boundary_agent` before parallel development when the work
touches multiple modules, configs, or test families.

Use `paper_sync_agent` when the task is manuscript text, method wording,
multiclass theory bridge, table placement, or appendix policy.

Use `decision_regret_developer_agent` when the task is the regret evaluator,
decision policy fitting, utility definitions, or theorem-aligned metrics.

Use `workflow_integration_developer_agent` when the task is Plan A runner
integration, study aggregation, config wiring, or artifact provenance.

Use `baseline_developer_agent` when the task is reference-model scoring,
softmax/energy/entropy exports, or baseline comparability.

Use `test_reviewer_agent` after implementation or when the user asks for audit,
review, risk analysis, or confidence assessment.

## 14. Handoff Format

Every agent should return:

```text
status: DONE | DONE_WITH_CONCERNS | NEEDS_CONTEXT | BLOCKED
role_id: <agent role>
inspected_paths:
changed_paths:
validation:
findings_or_changes:
residual_risks:
next_actions:
```

For read-only agents, `changed_paths` must be empty. For development agents,
`changed_paths` must stay inside the assigned write set unless the controller
approved expansion before the edit.
