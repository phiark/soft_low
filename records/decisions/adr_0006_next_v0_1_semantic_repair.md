# ADR-0006: next-v0.1 语义拆层与公平 matched benchmark

- status: accepted
- date: 2026-04-24
- owner: frcnet_project

## Context

审计报告指出, 当前项目的主要风险不是 FRCNet 双头结构错误, 而是 K 类 state 坐标、二元 proposition 坐标和 label-aware audit 坐标容易混用。若不拆开, `tau`, `content_entropy`, `completion_score` 和 matched benchmark 都可能被误读。

## Decision

`next-v0.1` 保留 FRCNet 主体结构, 只做语义拆层和公平评估骨架:

- 新增 state metrics: `state_content_entropy`, `state_weighted_content_entropy`, `state_entropy`
- 新增 proposition views: `top1_view` label-free, target/candidate views label-aware
- 新增 beta policy: completion 必须绑定 view
- 主 benchmark 只允许 label-free features
- scalar baseline 同时报 raw / oriented / one-feature logistic AUROC
- frozen matched manifest 必须由外部 Softmax CE reference score 构造
- Softmax reference 只服务 matching, 不作为完整 baseline 矩阵

## Consequences

- `proposition_truth_ratio` 不再允许作为主 benchmark scalar
- `content_entropy` 与 `completion_score_beta_*` 保留为 legacy alias
- `v0.3*` study/debug 协议不再属于本分支 active surface
- v0.1 的成功标准是评估语义可复核, 不是重新提升 FRCNet 训练指标
- 后续 baseline 和 decision benchmark 必须建立在此语义层上
