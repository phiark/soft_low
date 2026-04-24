# 方案 A next-v0.1 语义修复协议

- document_id: arch_plan_a_next_v0_1_protocol
- status: draft
- owner: frcnet_project
- last_updated: 2026-04-24

## 1. 目标

`next-v0.1` 在 `v0.3debug_r2` 之后修复实验语义和公平评估骨架。它不改变 FRCNet 主体结构, 只把 state / proposition / completion / matched benchmark 的边界拆清楚。

## 2. 拆层口径

state layer:

- `state_content_entropy = H_K(c)`
- `state_weighted_content_entropy = r * H_K(c)`
- `state_entropy = h(r) + r * H_K(c)`

proposition layer:

- `top1_view_*` 是 label-free, 可进入主 benchmark
- `target_view_*` 和 `candidate_view_*` 是 label-aware, 只允许进入 audit
- 旧 `proposition_truth_ratio` 保留为 legacy / diagnostic 字段, 不允许作为主 benchmark scalar

completion layer:

- `top1_completion_beta_*` 绑定 top-1 proposition view
- 旧 `completion_score_beta_*` 保留为兼容 alias
- beta policy 必须声明 view 语义, 不能被解释成全局 confidence

## 3. 公平评估口径

主 matched benchmark 默认只允许 label-free feature:

- `resolution_ratio`
- `unknown_mass`
- `state_content_entropy`
- `state_weighted_content_entropy`
- `top1_class_mass`
- `top1_view_tau`
- `top1_completion_beta_*`

默认 pair:

- raw state pair: `resolution_ratio__state_content_entropy`
- weighted state pair: `resolution_ratio__state_weighted_content_entropy`

默认 scalar:

- `top1_completion_beta_0_1`

每个 scalar 必须输出:

- `raw_auc`
- `oriented_auc`
- `one_feature_logistic_auc`

## 4. Frozen Matched Manifest

`next-v0.1` 引入 frozen matched manifest, 由外部 Softmax CE reference score 生成, 不使用被评估 FRCNet run 的输出自匹配。

默认 reference:

- model_family: `softmax_ce_reference`
- score_name: `softmax_entropy`
- matching bins: `10`

manifest 必须记录:

- `reference_score_name`
- `reference_score_value`
- `match_bin_id`
- `paired_group_id`
- `construction_config_hash`
- `manifest_hash`

## 5. Softmax Reference

Softmax CE reference 是最小辅助 pipeline:

- 使用同 backbone
- 只训练 in-domain class label 样本
- 只导出 `softmax_entropy` 或 `softmax_max_probability`
- 不作为完整 baseline study 行

## 6. 兼容性

- `content_entropy` 继续可读, 但新报告优先使用 `state_content_entropy`
- `resolution_weighted_content_entropy` 继续可读, 但新报告优先使用 `state_weighted_content_entropy`
- `completion_score_beta_*` 继续可读, 但新报告优先使用 `top1_completion_beta_*`
- proposition diagnostics 仍可使用 `proposition_truth_ratio`, 但主 benchmark 会拒绝它
