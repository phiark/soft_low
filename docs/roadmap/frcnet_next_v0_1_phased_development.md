# FRCNet next-v0.1 分阶段开发计划

- document_id: roadmap_frcnet_next_v0_1_phased_development
- status: active
- owner: frcnet_project
- last_updated: 2026-04-24

## 1. 目标

`next-v0.1` 只做两件事:

- 语义拆层: state / proposition / completion
- 公平评估骨架: external-reference frozen matched manifest + scalar fairness 表

本分支不是 `v0.3debug_r2` 的继续调参分支，也不是完整 baseline 仓库。

## 2. 当前必须保留

- FRCNet 原生双头模型
- Plan A manifest 和单 run workflow
- sample analysis record / proposition record
- state metrics / proposition views / beta policy
- matched benchmark / matched manifest / scalar baseline
- minimal Softmax CE reference score pipeline
- legacy analysis record reader

## 3. 当前必须移除或暂停

- `v0.3*` protocol/config/doc 入口
- multi-seed study runner
- aggregate report builder
- decision benchmark
- full external baseline matrix
- new model/loss tuning experiments

## 4. Phase 1: Semantic Layer Contract

交付:

- `src/frcnet/evaluation/state_metrics.py`
- `src/frcnet/evaluation/proposition_views.py`
- `src/frcnet/evaluation/beta_policy.py`
- analysis record 导出 `state_*` 和 `top1_*` 字段
- label-aware audit 字段与 label-free 主字段分离

验收:

- `top1_view` 不读取 label。
- `target_view` / `candidate_view` 显式 `label_aware = true`。
- `state_entropy = h(r) + r * H_K(c)`。
- proposition mass 满足 `pT + pF + pU = 1`。

## 5. Phase 2: Fair Matched Benchmark

交付:

- `src/frcnet/evaluation/matched_manifest.py`
- `src/frcnet/evaluation/scalar_baselines.py`
- `configs/eval/plan_a_next_v0_1_matched_manifest.yaml`
- matched manifest hash 和 bin summary

验收:

- 主 benchmark feature whitelist 只允许 label-free 字段。
- label-aware feature 进入主 benchmark 时直接失败。
- scalar 同时输出 raw / oriented / one-feature-logistic AUC。
- matched manifest 在同配置下 hash 稳定。

## 6. Phase 3: Minimal Reference Pipeline

交付:

- `src/frcnet/evaluation/softmax_reference.py`
- 独立 Softmax CE reference score export

验收:

- reference score 不来自被评估 FRCNet run。
- Softmax reference 只用于 matching，不作为完整 baseline 结论。

## 7. 完成定义

`next-v0.1` 完成时必须具备:

- 干净的 active config 集合
- 单 run workflow 可执行
- 主 benchmark 不含 `proposition_truth_ratio`
- frozen matched manifest 可复现
- 旧 analysis record 可读
- 测试套件通过

## 8. 延后项目

以下内容只作为后续版本候选，不进入本分支验收:

- decision utility / regret benchmark
- held-out ambiguity recipe 大实验
- multi-seed study 聚合
- Softmax / EDL / SelectiveNet baseline 矩阵
