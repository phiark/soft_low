# FRCNet Next v0.1 分阶段开发计划

- document_id: roadmap_frcnet_next_v0_1_phased_development
- status: draft
- owner: frcnet_project
- last_updated: 2026-04-24
- source_review: `/Users/zero_lab/Desktop/FRCNet_项目审计报告_执行意见_内部版.docx`
- baseline_branch: `codex/v0-3debug-r2-clean`

## 1. 目的

本文件定义 FRCNet 下一条开发线的 `next-v0.1` 分阶段计划。这里的 `v0.1` 不是回退当前 `v0.3debug_r2`，而是把审计报告提出的语义修复路线重新开成一个干净的小版本序列。

核心目标是保留 FRCNet 原生双头结构，同时把实验语义和评估骨架修清楚:

- 模型原生层: `resolution_ratio`, `content_distribution`, `class_mass`, `unknown_mass`
- state 诊断层: `state_content_entropy`, `state_weighted_content_entropy`, `state_entropy`
- proposition 视图层: `pT`, `pF`, `pU`, `tau_A`
- completion / decision 层: `q_beta`, interval, utility, regret

## 2. 审计结论转化

审计报告的主要判断是: FRCNet 的模型方向是对的，真正需要重做的是实验语义和评估骨架。当前风险不是 `u = 1-r` 或 `p_k = r*c_k` 这条主结构错了，而是二元 proposition 坐标、K 类 state 坐标、带标签 audit 坐标仍然容易混在同一套字段和图表里。

本版本必须优先解决以下问题:

- `content_entropy` 被写得像论文二元 `H_cont`，但实际是 K 类 `H_K(c)`
- `proposition_truth_ratio` 是 label-aware audit 量，不应进入主 benchmark feature pool
- 现有 matched benchmark 主要是 cohort 数量平衡，不是外部 reference-score matched manifest
- scalar baseline 只报 raw AUROC，方向和 probe 能力不公平
- ambiguous supervision 可能把目标几何直接教出来，需要 held-out recipe 和 ablation
- seen unknown 与 unseen OOD 没有足够强的命名隔离
- 缺少直接对应 decision blindness 的动作层实验

## 3. 范围

`next-v0.1` 做语义修复、评估修复和最小反证实验，不重写模型主体。

包含范围:

- 保留 ResNet-18 backbone、resolution head、content head
- 新增 state / proposition / beta policy / decision benchmark 模块
- 重命名或并行导出容易混淆的 analysis 字段
- 建立 label-free 主 benchmark feature whitelist
- 建立 frozen matched manifest 合同
- 修复 scalar baseline 的公平比较口径
- 增加最小 ablation 和 held-out recipe 计划

不包含范围:

- 不引入 Transformer
- 不引入 teacher distillation 作为默认方案
- 不把 `unknown` 改成自由的第 K+1 softmax 类
- 不把 FRCNet 写成论文理论本身的证明
- 不把完整外部 baseline 矩阵塞进第一阶段

## 4. 版本验收原则

`next-v0.1` 的通过标准不是单个 AUROC 数字，而是语义链条干净。

必须满足:

- 主 benchmark 的输入特征全部 label-free
- target / candidate proposition 只进入 audit 和附表
- state entropy 与 proposition entropy 命名分离
- scalar baseline 同时报 `raw_auc`, `oriented_auc`, `one_feature_logistic_auc`
- matched manifest 有 hash、bin stats、reference score distribution summary
- completion beta 与 proposition view 绑定，不再全局复用一个 `completion_score`
- 至少一个 held-out ambiguity recipe 或 held-out class-pair 实验通过 smoke
- 至少一个 decision benchmark 输出 utility / regret 表

## 5. 分阶段计划

### Phase 0: 仓库卫生与开发基线

目标:

- 防止实验产物、checkpoint、缓存再次进入 Git 历史
- 把下一版开发从干净分支开始，而不是污染本地 `main`

主要任务:

- 更新 `.gitignore`，忽略 `artifacts/experiments/`, `artifacts/studies/`, `records/experiments/`, `.cache/`, `tmp/pdfs/`
- 明确实验产物只能作为本地或外部存储对象，不作为普通 Git 代码提交
- 新增本文件并挂入 `docs/index.md`

验收:

- `git check-ignore` 能命中 study checkpoint 和 experiment report 产物
- 新分支只包含源码、配置、文档和测试
- 任何 PR 不包含 `.pt`, `.zip`, `.cache`, generated report image

### Phase 1: State / Proposition / Completion 拆层

目标:

- 把 K 类 state 坐标和二元 proposition 坐标拆成两个明确模块
- 让 `tau` 只有在指定 proposition view 之后才有定义

新增模块:

- `src/frcnet/evaluation/state_metrics.py`
- `src/frcnet/evaluation/proposition_views.py`
- `src/frcnet/evaluation/beta_policy.py`

字段合同:

- `state_content_entropy`: K 类 resolved distribution 的 `H_K(c)`
- `state_weighted_content_entropy`: `resolution_ratio * H_K(c)`
- `state_entropy`: `h(r) + r * H_K(c)`
- `top1_view_tau`: label-free top-1 proposition truth ratio
- `target_view_tau`: label-aware target proposition truth ratio
- `candidate_view_tau`: label-aware candidate-set proposition truth ratio
- `auxiliary_top1_content_probability`: legacy auxiliary field

验收:

- `content_entropy` 继续兼容旧记录，但新报告优先使用 `state_content_entropy`
- `proposition_truth_ratio` 不再作为单一万能 `tau` 暴露给主 benchmark
- top-1 view 不读取 `class_label` 或 `candidate_class_indices`
- target / candidate view 明确标记 `label_aware = true`

### Phase 2: Fair Matched Benchmark

目标:

- 把当前数量平衡 matched benchmark 升级为可复核的 frozen matched manifest
- 让 pair-vs-scalar 比较不再被方向和 probe 能力质疑

新增模块:

- `src/frcnet/evaluation/matched_manifest.py`
- `src/frcnet/evaluation/scalar_baselines.py`

manifest 字段:

- `sample_id`
- `cohort_name`
- `source_dataset_name`
- `source_index`
- `reference_score_name`
- `reference_score_value`
- `match_bin_id`
- `paired_group_id`
- `manifest_role`
- `manifest_hash`
- `construction_config_hash`

benchmark 输出:

- `pair_auc`
- `weighted_pair_auc`
- `scalar_raw_auc`
- `scalar_oriented_auc`
- `scalar_one_feature_logistic_auc`
- `reference_match_bin_summary.csv`
- `matched_manifest_summary.json`

主特征 whitelist:

- `resolution_ratio`
- `unknown_mass`
- `state_content_entropy`
- `state_weighted_content_entropy`
- `top1_class_mass`
- `top1_view_tau`
- `top1_completion_beta_*`

禁止进入主 benchmark 的字段:

- `class_label`
- `candidate_class_indices`
- `target_view_*`
- `candidate_view_*`
- `cohort_name` as feature

验收:

- 主 benchmark 如果配置了 label-aware 字段，必须直接失败
- scalar 与 pair 使用同一 train/test split
- scalar 结果至少包含 raw、oriented、一维 logistic 三列
- frozen manifest 在同一配置下 hash 稳定

### Phase 3: Ambiguity 与 OOD 反证实验

目标:

- 区分“模型学到 ambiguity 结构”和“模型只学到 recipe 伪迹”
- 区分 seen unknown 与 unseen OOD

最小实验:

- `heldout_ambiguity_recipe`: train MixUp, test overlay / occlusion / double-object 中至少一种
- `heldout_class_pair`: train 部分类对, test 未见类对
- `r_target_sweep`: `0.6`, `0.7`, `0.8`, `0.9`
- `no_r_target_ablation`: ambiguous 只用 candidate CE
- `unknown_source_split`: 明确 `unknown_supervision_train`, `seen_unknown_test`, `unseen_ood_test`

验收:

- 每个 ablation 产物都能回链到 config、manifest、checkpoint、report
- `seen_unknown_test` 和 `unseen_ood_test` 在文件名和报告字段中分离
- 主结论不得只依赖训练中见过的 ambiguous recipe
- 如果 no-r-target 后几何消失，报告必须把 ambiguous 结构写成 supervised mechanism，而不是 natural discovery

### Phase 4: Decision Benchmark

目标:

- 让实验直接对应论文的 decision blindness，而不是只停在 AUROC 诊断

新增模块:

- `src/frcnet/evaluation/decision_benchmark.py`

最小 action set:

- `accept_prediction`
- `request_more_evidence`
- `ask_finer_label_or_candidate_resolution`
- `reject_unknown`

policy 对照:

- `policy_q_beta_only`
- `policy_best_scalar`
- `policy_pair_state`
- `policy_oracle`

输出:

- `decision_expected_utility_table.csv`
- `decision_regret_table.csv`
- `decision_action_confusion_matrix.csv`
- `decision_policy_summary.json`

验收:

- pair policy 至少和 q-only policy 在同一 utility table 中比较
- regret 能按 cohort 拆开
- 报告中明确区分 diagnostic AUROC 和 decision utility

### Phase 5: 最小 Baseline 接入

目标:

- 为论文主结果补最代表性的对照，但不让 baseline 矩阵拖垮语义修复

第一批 baseline:

- `softmax_ce_same_backbone`
- `edl_same_backbone`

第一批 FRCNet ablation:

- `frcnet_no_ambiguous_supervision`
- `frcnet_no_unknown_supervision`
- `frcnet_no_unknown_content_neutrality`
- `frcnet_no_r_target_for_ambiguous`

验收:

- baseline 与 FRCNet 使用同一 frozen eval manifest
- baseline report 不导出 FRCNet-only 字段作为主状态字段
- aggregate 表支持 `model_family`
- 主文只比较同一任务、同一数据协议、同一 reference matched manifest 下的结果

## 6. 文件级开发清单

优先新增:

- `src/frcnet/evaluation/state_metrics.py`
- `src/frcnet/evaluation/proposition_views.py`
- `src/frcnet/evaluation/beta_policy.py`
- `src/frcnet/evaluation/matched_manifest.py`
- `src/frcnet/evaluation/scalar_baselines.py`
- `src/frcnet/evaluation/decision_benchmark.py`
- `configs/eval/plan_a_next_v0_1_matched_manifest.yaml`
- `configs/study/plan_a_next_v0_1_study.yaml`

优先修改:

- `src/frcnet/evaluation/inference.py`
- `src/frcnet/evaluation/records.py`
- `src/frcnet/evaluation/matched_benchmark.py`
- `src/frcnet/analysis/reporting.py`
- `src/frcnet/workflows/study.py`
- `docs/architecture/plan_a_paper_linkage.md`
- `docs/verification/verification_and_validation_plan.md`

## 7. 测试清单

必须新增:

- `test_mass_conservation`
- `test_state_entropy_decomposition`
- `test_proposition_mass_conservation`
- `test_top1_view_label_free`
- `test_target_candidate_audit_only`
- `test_beta_policy`
- `test_two_completion_recovery`
- `test_scalar_orientation`
- `test_frozen_manifest_hash`
- `test_no_seen_ood_confusion`

回归测试:

- 旧 `v0.3debug_r2` analysis record 仍可读取
- 旧 `content_entropy` 字段可作为 legacy alias 读取
- 旧 report 不会把 target / candidate audit 字段放入主 benchmark
- study resume 和 sidecar integrity 不回归

## 8. 论文口径约束

允许写:

- FRCNet 是论文 explicit-unknown chart 的 K 类原生参数化
- `state_content_entropy` 是 `H_K(c)`
- 二元 `tau_A` 必须先声明 proposition view `A`
- pair 优势是 diagnostic evidence
- decision blindness 需要 decision benchmark 支撑

禁止写:

- FRCNet 直接证明了论文理论
- `content_entropy` 就是论文二元 `H_cont`
- `target_view_tau` 或 `candidate_view_tau` 是公平主 benchmark scalar
- 数量平衡 cohort 就等于 scalar-matched benchmark
- ambiguous geometry 是自然发现，除非 held-out 和 ablation 通过

## 9. v0.1 完成定义

`next-v0.1` 完成时必须交付:

- 一套拆层后的 analysis record schema
- 一套 label-free 主 benchmark feature whitelist
- 一套 frozen matched manifest 生成和验证链
- 一套公平 scalar baseline 表
- 至少一个 held-out ambiguity 反证实验
- 至少一个 decision utility / regret 小实验
- 更新后的 paper linkage 和 V&V 文档
- 不含生成产物和 checkpoint 的 clean branch

## 10. 当前优先级

第一优先级:

- Phase 1 和 Phase 2

第二优先级:

- Phase 3 的最小 held-out recipe 与 no-r-target ablation

第三优先级:

- Phase 4 decision benchmark

第四优先级:

- Phase 5 baseline 接入

执行顺序不能反过来。先补 baseline 但不修语义，会让对照结果继续站在混乱字段上；先修语义和 matched manifest，后续 baseline 才有解释价值。
