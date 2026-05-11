# Review: Model Positioning And Future Direction

- review_id: review_model_positioning_and_future
- date: 2026-05-02
- owner: frcnet_project
- status: analysis
- scope: `FRCNet 0.6.0 / V0.6C`, `V3_reset.pdf`, current OOD/open-set research context

## 1. Executive Position

FRCNet should not currently be positioned as a state-of-the-art OOD detector.

Its defensible position is narrower and more interesting:

> FRCNet is a dedicated explicit-unknown state model. It turns the paper's completion-projection idea into an inspectable neural architecture by separating resolution (`r`) from resolved content (`c`) and exposing unknown mass (`u = 1 - r`) before any scalar confidence score is chosen.

The current result is a clean partial evidence point, not a strong final paper result. The system is valuable because it makes the failure visible: far/source-style OOD is learnable, but CIFAR-like near-OOD semantic unknown remains weak.

## 2. What The Paper Actually Says

The reset manuscript is strongest as a projection/diagnostic paper, not as an entropy paper and not as a performance paper.

The core paper claim is:

- explicit unknown mass is not just another confidence penalty;
- a single confidence score appears only after a completion policy `beta`;
- one completion scalar can be decision-blind;
- two distinct completion projections can recover the original ternary state;
- the useful chart is `(r, tau)`, where `r` is resolution and `tau` is truth ratio inside the resolved subset.

The empirical part of `V3_reset.pdf` is deliberately modest. It reports a corrected EDL audit where the pair `(r, rH_cont)` consistently but only slightly beats the best scalar baseline, and selective rejection is near-null. The manuscript itself says the content axis is underpopulated because many samples collapse toward `tau ~= 1`.

This matches the code project's outcome: the architecture implements the paper's split, but current evidence does not support a strong "natural three-group geometry" or broad decision-level advantage claim.

## 3. What The Code Has Actually Built

The implementation is structurally coherent.

The core model in `src/frcnet/models/frcnet_model.py` is:

- backbone feature extractor;
- resolution head;
- content head;
- optional source adversary head;
- `class_mass = resolution_ratio * content_distribution`;
- `unknown_mass = 1 - resolution_ratio`.

The loss implementation in `src/frcnet/training/losses.py` covers the intended cohorts:

- easy/hard ID: optimize class mass on the target class;
- unknown supervision: optimize unknown mass;
- ambiguous ID: optimize candidate-set content distribution and an `ambiguous_resolution_target`;
- V0.6B/V0.6C repairs: source adversarial loss, OOD supervised contrastive loss, and source-balanced calibration.

The evaluation layer is also aligned with the paper. `src/frcnet/evaluation/matched_benchmark.py` tests pair features against scalar readouts, and the analysis contract exports `resolution_ratio`, `unknown_mass`, `state_content_entropy`, proposition fields, and completion beta scans.

Engineering health is good. Current validation passed:

```text
.venv313/bin/python -m pytest -q
108 passed in 41.35s
```

## 4. Current Evidence State

The archived V0.6C result is:

- unseen CIFAR100 held-out classes pair AUROC: `0.6512 +/- 0.0112`
- all-OOD pair AUROC: `0.8019 +/- 0.0076`
- worst-source pair AUROC: `0.6435 +/- 0.0214`
- seen-unseen gap: `0.2178 +/- 0.0020`
- hard ID top-1: `0.6333 +/- 0.0120`
- ambiguous candidate hit: `0.7167 +/- 0.0127`
- pair-scalar delta: `0.0030 +/- 0.0032`

Interpretation:

- Far/source-style OOD separation is real.
- Multi-source and source-invariant repair improved all-OOD aggregate behavior.
- CIFAR-like near-OOD remains the hard case.
- The pair-vs-scalar margin is too small for a strong FRCNet-specific advantage claim.
- The project should not claim unseen CIFAR100 source generalization; V0.6C only supports unseen CIFAR100 class-holdout evidence.

## 5. Position Against Current Research

The field around FRCNet has several strong families:

- Softmax confidence baselines started with maximum softmax probability for misclassification/OOD detection.
- ODIN improved post-hoc softmax separation with temperature scaling and input perturbation.
- Energy-based OOD uses energy scores instead of softmax confidence and can be used as a score or training objective.
- ReAct reduces OOD overconfidence by clipping/rectifying internal activations.
- ViM combines feature-space residuals and logits through a virtual OOD logit.
- Outlier Exposure trains with broad auxiliary outliers and often helps unseen anomaly detection.
- OpenOOD and OpenOOD v1.5 show that evaluation must be standardized, large-scale, and now include foundation models and full-spectrum shifts.
- Semantically Coherent OOD argues that dataset-as-OOD benchmarks can reward low-level source differences instead of semantic unknown recognition.
- VLM-era methods such as GL-MCM, OLE, and EOE use CLIP or language-generated outlier labels to improve zero-shot and hard OOD detection.

Against that landscape, FRCNet is currently behind as a detector, especially for near-OOD and real-world semantic OOD. It does not yet use strong pretrained visual-language semantics, DINOv2/CLIP-scale representations, feature residual scoring, or language-generated outlier label space.

But FRCNet has a different asset: it exposes a state decomposition before scalar scoring. Most strong OOD detectors optimize one rejection score. FRCNet can become useful if it proves that the state decomposition improves diagnosis or decisions where "ambiguous known" and "semantic unknown" require different actions.

## 6. What Is Better And What Is Worse

Better than ordinary confidence/OOD baselines:

- It separates "how resolved is this sample" from "which known class is it".
- It makes completion policy visible instead of hiding unknown mass inside one score.
- It can export proposition-level `truth / false / unknown` views.
- It has unusually strong experiment traceability: docs, configs, manifests, sidecars, matched benchmarks, records.
- It explains failure modes rather than just reporting AUROC.

Worse than current strong methods:

- Raw near-OOD performance is weak.
- Pair-scalar delta is tiny.
- The content axis is not reliably populated.
- Source/style shortcuts dominate far-OOD learning.
- The backbone and supervision are much weaker than modern VLM/foundation-model approaches.
- It has not shown decision-level gains.
- It has not been benchmarked against Energy, ReAct, ViM, MSP/ODIN, CLIP MCM/OLE/EOE under the same final protocol.

## 7. Architecture Diagnosis

There is no obvious fatal implementation mismatch in the current architecture. The architecture is doing what the paper asked for.

The problem is that the paper-level factorization is necessary but not sufficient. Main issues:

1. The single resolution gate is overloaded. It must represent far OOD, near OOD, ambiguity, hard ID, and source shift with one scalar `r`.
2. Unknown supervision teaches "this source is outside" more easily than "this semantic neighborhood is outside".
3. Ambiguous supervision is mostly candidate-set/MixUp-style ambiguity, not natural semantic ambiguity.
4. Source-adversarial repair reduces explicit source leakage but does not create semantic unknown structure by itself.
5. The content entropy coordinate can collapse if the content head becomes too confident inside the resolved subspace.
6. Completion scores are analyzed, but the model is not yet trained against a downstream decision-regret objective where preserving multiple completions matters.

So the design is conceptually sound as a diagnostic container, but too small as a final semantic unknown detector.

## 8. What We Did Wrong

The main wrong assumption was treating the `r/c/u` factorization as if it would naturally produce the desired three groups once implemented.

In practice:

- far OOD gives the gate an easy low-level shortcut;
- near-OOD shares CIFAR-like image statistics and can look resolved;
- content entropy is not forced to represent semantic alternatives;
- artificial ambiguity does not automatically teach natural ambiguity;
- better data diversity alone does not create semantic unknown recognition;
- pair geometry was expected to become the result, but it remained mostly a diagnostic.

This was not direct scientific fraud and not just a coding bug. It was an over-strong modeling assumption.

## 9. What We Can Still Do

There are three viable paths.

### Path A: Keep The Paper Narrow And Defensible

Position the paper as:

- explicit unknown states need a completion layer;
- one scalar confidence can be decision-blind;
- `(r, tau)` is a compact diagnostic chart;
- current EDL/FRCNet evidence shows real but modest residual signal;
- stronger model supervision remains future work.

This is the safest paper path.

### Path B: Build FRCNet V0.7 As A Semantic Unknown Model

The next model should not merely add more OOD sources. It should add semantic pressure.

Recommended V0.7 changes:

- add modern baselines first: MSP, ODIN, Energy, ReAct, ViM, softmax CE, and CLIP/MCM-style scores;
- add a strong frozen semantic teacher, likely CLIP or DINOv2, for semantic-neighborhood constraints;
- split unknown into multiple unresolved states, e.g. far-source unknown, near-semantic unknown, and ambiguous-known;
- train with class-neighborhood or prototype margins so CIFAR100-like classes are not treated as arbitrary source style;
- add a decision-regret benchmark where ambiguous-known and semantic-unknown have different optimal actions;
- make the claim "state decomposition improves action diagnosis" rather than "AUROC SOTA".

### Path C: Pivot To A Hybrid Diagnostic Layer

Instead of competing with CLIP/VLM OOD methods, put FRCNet on top of a strong representation:

- use CLIP/DINOv2 features as the backbone;
- let FRCNet learn `r`, `content_distribution`, and proposition states over semantic prototypes;
- keep completion beta scans and proposition outputs as the value-add;
- compare against CLIP energy/MCM/OLE/EOE baselines.

This is likely the highest-upside direction because it keeps the paper's originality while not fighting modern representation learning with a small ResNet alone.

## 10. Direct Answers

### Can we still do something?

Yes. The project can still produce a meaningful paper and a better V0.7 model, but only if the claim is narrowed from "better OOD detector" to "explicit state decomposition and decision-aware uncertainty diagnostics".

### What is the next step?

Create a V0.7 plan before any new training. The plan should define modern baselines, semantic-teacher integration, multi-unresolved-state design, and decision-regret evaluation.

### What should the model do?

It should distinguish:

- known and resolved;
- known but ambiguous;
- semantically near but outside;
- far/source-style outside.

The current single `unknown_mass` should become either multi-state unknown or be supported by explicit semantic-neighborhood diagnostics.

### What is better or worse than others?

Better: interpretability of unknown mass and completion sensitivity.

Worse: raw OOD performance and modern benchmark competitiveness.

### Is the architecture wrong?

No. It is a valid first architecture, but it is underpowered for the original claim.

### Can we do something others cannot?

Not in the absolute sense. Others could implement similar state decomposition. The unique contribution is the combination of completion theory, explicit neural state contract, proposition diagnostics, and traceable evidence workflow. That can be publishable if framed honestly.

### Is the paper meaningful?

Yes, if framed as a projection/diagnostic paper. No, if framed as a new entropy theory or SOTA OOD detector paper.

### Why is the model worse?

Because modern methods either use stronger representations, stronger OOD objectives, feature-space geometry, large auxiliary outliers, or language semantics. Our current model uses a small explicit factorization and expects the right geometry to emerge from weak supervision.

### What did we do wrong?

We overestimated what the factorization alone would learn. The next version must directly supervise or constrain the semantic distinction the paper wants.

## 11. Sources Checked

Local:

- `V3_reset.pdf` / `V3_reset.tex`
- `README.md`
- `docs/architecture/architecture_description.md`
- `docs/requirements/system_requirements_specification.md`
- `docs/architecture/plan_a_next_v0_6c_near_ood_split_repair_protocol.md`
- `records/reviews/2026-04-26_review_v0_6_evidence_report.md`
- `records/reviews/2026-04-27_review_v0_6c_archive_closure.md`
- `src/frcnet/models/frcnet_model.py`
- `src/frcnet/training/losses.py`
- `src/frcnet/evaluation/matched_benchmark.py`

External:

- Hendrycks and Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks", ICLR 2017, https://arxiv.org/abs/1610.02136
- Liang, Li, and Srikant, "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks", ICLR 2018, https://arxiv.org/abs/1706.02690
- Hendrycks, Mazeika, and Dietterich, "Deep Anomaly Detection with Outlier Exposure", ICLR 2019, https://arxiv.org/abs/1812.04606
- Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020, https://papers.nips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html
- Sun, Guo, and Li, "ReAct: Out-of-distribution Detection With Rectified Activations", NeurIPS 2021, https://papers.nips.cc/paper/2021/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html
- Wang et al., "ViM: Out-of-Distribution With Virtual-Logit Matching", CVPR 2022, https://openaccess.thecvf.com/content/CVPR2022/html/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.html
- Yang et al., "Semantically Coherent Out-of-Distribution Detection", ICCV 2021, https://arxiv.org/abs/2108.11941
- Yang et al., "OpenOOD: Benchmarking Generalized Out-of-Distribution Detection", NeurIPS Datasets and Benchmarks 2022, https://arxiv.org/abs/2210.07242
- Zhang et al., "OpenOOD v1.5", DMLR, https://arxiv.org/abs/2306.09301
- Miyai et al., "GL-MCM: Global and Local Maximum Concept Matching for Zero-Shot Out-of-Distribution Detection", IJCV 2025, https://arxiv.org/abs/2304.04521
- Cao et al., "Envisioning Outlier Exposure by Large Language Models for Out-of-Distribution Detection", ICML 2024, https://arxiv.org/abs/2406.00806
- Ding and Pang, "Zero-Shot Out-of-Distribution Detection with Outlier Label Exposure", IJCNN 2024, https://arxiv.org/abs/2406.01170
- Miyai et al., "Generalized Out-of-Distribution Detection and Beyond in Vision Language Model Era: A Survey", TMLR 2025, https://arxiv.org/abs/2407.21794
- Noda et al., "A Benchmark and Evaluation for Real-World Out-of-Distribution Detection Using Vision-Language Models", ICIP 2025, https://arxiv.org/abs/2501.18463
