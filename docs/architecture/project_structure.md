# 项目结构说明

- document_id: arch_project_structure
- status: baselined
- owner: frcnet_project
- last_updated: 2026-05-11

## 1. 设计原则

项目结构以“文档先行、证据留痕、实现解耦”为原则组织:

- `docs/` 放规范
- `records/` 放事实
- `src/` 放实现
- `artifacts/` 放生成物

## 2. 标准目录树

```text
HardMin/
├── README.md
├── pyproject.toml
├── docs/
│   ├── index.md
│   ├── governance/
│   ├── requirements/
│   ├── architecture/
│   ├── verification/
│   ├── records/
│   └── templates/
├── records/
│   ├── decisions/
│   └── reviews/
├── src/
│   └── frcnet/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── analysis/
│       ├── maintenance/
│       ├── workflows/
│       └── utils/
├── configs/
│   ├── model/
│   ├── data/
│   ├── protocol/
│   ├── train/
│   ├── eval/
│   ├── analysis/
│   └── study/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
├── scripts/
├── artifacts/
│   ├── figures/
│   ├── tables/
│   ├── reports/
│   ├── checkpoints/
│   ├── studies/
│   └── logs/
└── notebooks/
```

## 3. 分层语义

### 3.1 `docs/`

项目的规范性源头。这里定义术语、接口、需求、架构和验证要求。

### 3.2 `records/`

存放 ADR、评审记录和 compact evidence record。这里不改写规范, 只记录决策和结果。

`records/experiments/` 属于历史/生成性实验记录路径, 当前不作为常规文档分类新增目录。需要实验记录时, 优先由 study/report 工作流生成到 artifact bundle, 再把 compact review 或 archive record 放入 `records/reviews/`。

### 3.3 `src/frcnet/`

实现层。任何新增模块都应能说明它对应哪个 requirement 或 ADR。

`src/frcnet/maintenance/` 只放封存、artifact hygiene 和可持续性维护工具。该目录不得引入新的训练或评估科学口径。

### 3.4 `configs/`

配置与代码分离。训练、评估、分析参数都应通过配置文件落盘并被实验记录引用。

### 3.5 `tests/`

按验证粒度拆分:

- `unit`: 数学与函数级校验
- `integration`: 训练/评估流水线拼装
- `contract`: 数据和输出 schema 不变量

### 3.6 `artifacts/`

只保存可重新生成的派生对象。其真值来源是代码、配置和实验记录, 不是 artifact 自身。

封存状态下, 普通提交不得包含 generated artifact tree、checkpoint、大型 CSV 或运行缓存。checkpoint 保留规则见 `docs/governance/project_archive_status.md`。

新的 study 根目录统一放在 `artifacts/studies/YYYY-MM-DD_vX.Y_slug/`。历史目录如 `plan_a_v0_3_main`、`RUN-*` 或 `KYUC-*` 只为追溯保留, 不作为新命名模板。

## 4. 文件树约束

- 目录树优先表达职责, 不表达临时阶段
- 不建立 `misc`, `temp`, `other`, `new_files`, `artifact`, `stuidie` 之类无语义或拼写错误目录
- 训练、评估、分析禁止混放在同一模块
- notebook 只用于探索, 结论要沉淀回 `docs/` 或 `records/`
