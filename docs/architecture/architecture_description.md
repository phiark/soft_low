# 架构说明

- document_id: arch_architecture_description
- status: baselined
- owner: frcnet_project
- last_updated: 2026-05-11
- standard_alignment: iso_ieee_42010_informed, iso_iec_ieee_15289

## 1. 架构目标

架构必须把论文中的 gate/content 分解变成原生工程结构, 而不是后处理附加层。模型和分析链路都围绕以下显式对象构建:

- `resolution_ratio`
- `content_distribution`
- `class_mass`
- `unknown_mass`
- `state_content_entropy`
- `state_weighted_content_entropy`
- `top1_completion_beta_*`

## 2. 上下文视图

```text
input sample
  -> cohort-aware data builder
  -> backbone
  -> resolution head
  -> content head
  -> structured output contract
  -> loss computation / inference metrics
  -> evaluation datasets
  -> analysis and paper artifacts
```

## 3. 模块分解

### 3.1 `src/frcnet/data`

职责:

- 数据源注册
- cohort 构造
- 歧义样本 candidate set 编码
- batch 合约标准化

建议子模块:

- `dataset_registry.py`
- `cohort_builders.py`
- `sample_contracts.py`
- `transforms.py`

### 3.2 `src/frcnet/models`

职责:

- backbone 封装
- resolution head
- content head
- 输出归一化与结构化打包

建议子模块:

- `backbones.py`
- `resolution_head.py`
- `content_head.py`
- `frcnet_model.py`
- `output_contracts.py`

### 3.3 `src/frcnet/training`

职责:

- loss 实现
- 训练步骤
- 日志与 checkpoint 逻辑

建议子模块:

- `losses.py`
- `trainer.py`
- `optimizers.py`
- `schedulers.py`

### 3.4 `src/frcnet/evaluation`

职责:

- 样本级推理导出
- scalar 指标
- pair probe 指标
- selective / matched benchmark

建议子模块:

- `inference.py`
- `scalar_metrics.py`
- `pair_metrics.py`
- `matched_benchmark.py`

### 3.5 `src/frcnet/analysis`

职责:

- cohort occupancy
- scatter / hexbin 可视化
- completion sensitivity
- 论文表格聚合

建议子模块:

- `geometry_reports.py`
- `plotting.py`
- `completion_analysis.py`
- `report_tables.py`

### 3.6 `src/frcnet/utils`

职责:

- 熵计算
- 标识符生成
- 序列化
- 可追溯日志工具

## 4. 数据合约

### 4.1 Input Batch Contract

每个 batch 至少应包含:

- `image`
- `class_label`
- `sample_id`
- `split_name`
- `cohort_name`
- `source_dataset_name`
- `source_class_label` 可选
- `candidate_class_set` 可选

### 4.2 Model Output Contract

模型前向输出至少应包含:

- `backbone_feature`
- `resolution_logit`
- `resolution_ratio`
- `content_logits`
- `content_distribution`
- `class_mass`
- `unknown_mass`

### 4.3 Analysis Record Contract

样本级分析表至少应包含:

- `run_id`
- `protocol_id`
- `sample_id`
- `split_name`
- `cohort_name`
- `source_dataset_name`
- `source_class_label`
- `predicted_class_index`
- `class_label`
- `resolution_ratio`
- `unknown_mass`
- `state_content_entropy`
- `state_weighted_content_entropy`
- `state_entropy`
- `resolution_entropy`
- `top1_class_mass`
- `top1_view_truth_mass`
- `top1_view_false_mass`
- `top1_view_unknown_mass`
- `top1_view_tau`
- `proposition_truth_mass`
- `proposition_false_mass`
- `proposition_unknown_mass`
- `proposition_truth_ratio`
- `ternary_entropy`
- `auxiliary_top1_content_probability`
- `top1_completion_beta_0_1`
- `top1_completion_beta_0_25`
- `top1_completion_beta_0_5`
- `top1_completion_beta_0_75`

`content_entropy`, `resolution_weighted_content_entropy`, and `completion_score_beta_*` are legacy aliases for reading historical records, not canonical v0.2 output fields.

### 4.4 Analysis Export Summary Contract

analysis 阶段必须额外生成 `analysis_summary.json`, 作为 report 阶段的规范 sidecar。

该 sidecar 至少应包含:

- `run_id`
- `protocol_id`
- `analysis_path`
- `checkpoint_path`
- `checkpoint_selection_summary_path`
- `manifest_snapshot_path`
- `model_config_snapshot_path`
- `proposition_path`
- `model_family`
- `integrity_overrides`
- `sidecar_resolution_mode`

## 5. 关键架构约束

- `ARCH-001`: `unknown_mass` 只能由 `resolution_ratio` 导出
- `ARCH-002`: `content_distribution` 仅描述 resolved 子空间内部类别分布
- `ARCH-003`: `completion_score` 属于读出层, 不属于模型主状态
- `ARCH-004`: 分析层必须支持 pair 与 scalar 并行输出
- `ARCH-005`: 文档、配置、代码、实验记录之间必须有可追溯链接
- `ARCH-006`: report 阶段必须先完成 bundle integrity 校验, 再写 `experiment_record.md`
- `ARCH-007`: analysis/report 间的 sidecar 解析必须优先使用规范 `analysis_summary.json`, 不得默认依赖 sibling 猜测

## 6. 当前技术基线

- 默认 backbone: `ResNet-18`
- 默认主包路径: `src/frcnet/`
- 默认配置组织: `configs/model`, `configs/data`, `configs/protocol`, `configs/train`, `configs/eval`, `configs/analysis`, `configs/study`
- 默认记录组织: `records/decisions`, `records/reviews`
- 默认工作流入口: `src/frcnet/workflows/plan_a.py` 与 `src/frcnet/workflows/study.py`

## 7. 已收口架构决策

- 主 paper-facing pair 使用 `(resolution_ratio, state_content_entropy)`, weighted pair 保留为 secondary/ablation 输出
- `top1 correctness proposition` 属于评估与诊断层对象, 不进入训练主监督
- analysis sidecar 使用 `analysis_summary.json`; CSV 是当前样本级记录交换格式
