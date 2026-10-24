# FuseIA: A Budget-Aware Index Advisor with Fine-Tuned Small LLMs via Workload-Wide Semantic–Structural Fusion

<p align="right"><a href="#english">English</a> | <a href="#简体中文">简体中文</a></p>

## English

**FuseIA** is a budget-aware index advisor that combines workload-wide SQL semantics with relational structure and database-state features. It uses two role-specific policies: a Proposer creates a short candidate list, then an Arbiter adds one index or stops. The policy is trained with paired original-name and anonymous views so it can leverage the LLM's semantic understanding of queries while learning structure that is independent of names.

This repository contains the core model and policy code, training/evaluation data, the SFT best checkpoints, and the GRPO `(30, 30)` checkpoint pair. The Qwen base model is not included.

### How the method works

1. **Represent the whole workload.** SQL is parsed into parameterized structures and predicates. Query text is encoded in chunks with Qwen2.5; schema, predicates, joins, and relations become a typed relational graph. Database statistics, candidate metadata, the current index set, and remaining storage budget form the decision state.
2. **Fuse semantic and structural evidence.** A relation-aware graph encoder represents structural nodes and edges. Candidate representations attend over semantic query tokens, graph nodes, current indexes, and budget/state features through cross-attention fusion.
3. **Make sequential budget-aware decisions.** The Proposer emits up to three candidates or ends its shortlist. The Arbiter chooses one candidate or `STOP`. Storage and feasibility masks constrain each action.
4. **Train across paired views.** SFT supervises the same decisions on original and anonymous views, with a symmetric action-consistency loss and relation-prediction loss. GRPO then optimizes the Proposer and Arbiter roles using workload execution and index-storage rewards.

### Code map

| Module | Responsibility |
|---|---|
| `structured_sql.py` | Parse SQL into stable, parameterized query structures and predicates. |
| `relational_graph.py` | Build relation graphs and produce consistent anonymous names. |
| `predicate_stats.py` | Read PostgreSQL statistics and estimate predicate selectivity. |
| `candidate_pool.py` | Load frozen index candidates and provide storage/budget operations without DDL. |
| `policy_inputs.py`, `structured_state.py` | Build workload context and state-conditioned candidate features. |
| `semantic_structure_model.py` | Qwen LoRA semantic encoder, relation graph encoder, cross-attention fusion, and role-specific action heads. |
| `policy_data.py`, `structured_data/` | Prepare paired-view SFT batches and structured graph inputs. |
| `policy_losses.py` | Paired SFT action, view-consistency, and relation losses. |
| `policy_actions.py` | Shared categorical action protocol for training and inference. |
| `grpo_rollout.py`, `grpo_reward.py` | Proposer→Arbiter episodes, PostgreSQL workload timing, and GRPO reward accounting. |
| `recommendation_rollout.py`, `teacher_prompts.py` | Shared text-policy recommendation protocol and role prompts. |
| `model_checkpoint.py` | Save and restore the role-specific adapter and FuseIA network weights. |
| `scripts/recommend.py`, `scripts/evaluate.py` | Generic CLIs for learned recommendation and serial actual-query evaluation. |
| `db.py`, `indexes.py` | PostgreSQL/HypoPG access and index-set utilities. |

### Repository layout

- `checkpoints/sft/{proposer,arbiter}/`: structured SFT best checkpoints.
- `checkpoints/grpo/{proposer,arbiter}/`: aligned GRPO Proposer/Arbiter update 30 checkpoints.
- `data/workloads/{tpch,tpcds,job}/`: canonical standalone evaluation workloads. See `data/workloads/workload_registry.json` for query counts and SHA-256 hashes.
- `data/statistics/`, `data/index_candidates/`: schema/statistics snapshots and candidate definitions.
- `data/structured_sft/`, `data/structured_teacher/`, `data/manifests/`: training examples, structured teacher data, and split manifests.
- `structured_data/`: graph collation and paired-view data utilities.

The checkpoint directories contain LoRA adapters and FuseIA-specific network weights, not a copy of the Qwen base model. Large checkpoint/data files are stored with Git LFS.

### Setup

Clone with Git LFS so checkpoint and large dataset files are materialized:

```bash
git lfs install
git clone https://github.com/pmss778/FuseIA.git
cd FuseIA
git lfs pull
```

Create a Python environment and install the pinned inference dependencies. The provided PyTorch wheel targets CUDA 12.8:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-local-inference.txt
```

For OpenAI-compatible teacher/API calls and PostgreSQL-backed measurement, install the optional client packages:

```bash
python -m pip install openai psycopg2-binary
```

Configure database connection variables (`PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`) before database-backed evaluation. `db.py` expects HypoPG to be installed in the target PostgreSQL database for hypothetical-index planning.

### Download the Qwen base model

The checkpoints reference `Qwen/Qwen2.5-3B-Instruct`. Download it separately from Hugging Face into a local directory:

```bash
python -m pip install --upgrade huggingface_hub
hf download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct
```

The checkpoint loader accepts an explicit local base-model path, so it does not need the original training-machine path embedded in the checkpoint metadata:

```python
from model_checkpoint import load_checkpoint

proposer, proposer_metadata = load_checkpoint(
    "checkpoints/grpo/proposer",
    device="cuda",
    backbone_name="models/Qwen2.5-3B-Instruct",
)
proposer.eval()
```

Use `checkpoints/sft/proposer` for the SFT Proposer, or the matching `arbiter` directory for the Arbiter. The GRPO bundle is aligned at update 30 for both roles. `grpo_rollout.share_frozen_base` can share frozen Qwen tensors between role models while leaving their adapters and FuseIA modules separate.

### Reusable recommendation and evaluation commands

The shared entry points are in `scripts/`; they replace workload-specific one-off launchers. Put generated recommendation and timing JSON outside this source tree (the examples use `../FuseIA_runs/`) to keep experimental outputs out of the GitHub project.

### Generate a greedy recommendation

`scripts/recommend.py` takes an ordered workload CSV, candidate pool, schema/statistics snapshot, a Proposer/Arbiter checkpoint pair, a storage budget, and a view. It runs greedy Proposer→Arbiter decisions, moves only the active role to the accelerator, and writes the selected indexes, per-step decisions, checkpoint updates, storage estimate, and run metadata to JSON.

Example: TPC-DS SF1 without primary keys, using the aligned GRPO update-30 pair and a 20% budget of 204.8 MiB:

```bash
python scripts/recommend.py \
  --workload data/workloads/tpcds/sf1/tpcds83_pg_paper.csv \
  --candidate-pool data/index_candidates/tpcds_sf1/candidates_tpds_llm.json \
  --schema data/statistics/tpcds_sf1_nopk/schema.json \
  --stats data/statistics/tpcds_sf1_nopk/pg_stats.json \
  --proposer-checkpoint checkpoints/grpo/proposer \
  --arbiter-checkpoint checkpoints/grpo/arbiter \
  --backbone models/Qwen2.5-3B-Instruct \
  --budget-mib 204.8 \
  --view original \
  --output ../FuseIA_runs/tpcds_fuseia_recommendation.json
```

Use `--view anonymous` to infer from anonymized names. Set `--budget-mib 1024` for a 100% TPC-DS SF1 budget. Replace both checkpoint arguments with `checkpoints/sft/proposer` and `checkpoints/sft/arbiter` to use SFT weights. `--feasibility-mask` opts into masking over-budget choices; by default the CLI follows the trained action protocol and reports an infeasible Arbiter choice as a stop condition.

### Measure one or more recommendation files

`scripts/evaluate.py` executes the workload against each supplied recommendation, serially, creates real indexes in a private `ix_<namespace>_` namespace, measures each query repeatedly, and stores per-query medians, total time, physical index sizes, and failures in JSON. It drops only indexes in its own namespace on exit; all other database indexes are left untouched.

```bash
python scripts/evaluate.py \
  --database indexselection_tpcds___1_no_pk \
  --workload data/workloads/tpcds/sf1/tpcds83_pg_paper.csv \
  --recommendation ../FuseIA_runs/tpcds_fuseia_recommendation.json \
  --recommendation ../FuseIA_runs/tpcds_teacher_recommendation.json \
  --runs 3 \
  --statement-timeout-seconds 600 \
  --output ../FuseIA_runs/tpcds_comparison.json
```

The database must be a test database. Configure `PGHOST`, `PGPORT`, `PGUSER`, and `PGPASSWORD` first. Use `--statement-timeout-seconds 0` to disable per-query timeouts. Each result path is write-once; choose a new output filename for a new run.

### Checkpoint pair loading API

The command uses the public checkpoint loader, which overrides the training-machine base-model path with the downloaded local model directory:

```python
from model_checkpoint import load_checkpoint

proposer, proposer_meta = load_checkpoint(
    "checkpoints/grpo/proposer",
    device="cuda",
    backbone_name="models/Qwen2.5-3B-Instruct",
)
```

For pair inference with constrained GPU memory, use `grpo_rollout.share_frozen_base` to share frozen Qwen parameters and keep role adapters and FuseIA modules independent. `scripts/recommend.py` handles sequential role placement automatically.
### Checkpoint contents

Each role checkpoint includes:

- `adapter/`: Qwen LoRA adapter weights and configuration.
- `network.pt`: FuseIA graph, fusion, state, and action-head parameters.
- `tokenizer/`: tokenizer files used by the checkpoint.
- `checkpoint.json`: model configuration, feature schema, and training metadata.
- GRPO checkpoints also include `training.pt` for optimizer and RNG resume state.

See `checkpoints/CHECKPOINTS.txt` for the selected SFT and GRPO checkpoint pairing.

<p align="right"><a href="#english">English</a> | <a href="#简体中文">简体中文</a></p>

## 简体中文

**FuseIA** 是一个预算感知的索引顾问，将整个 workload 的 SQL 语义与关系结构、数据库状态特征结合起来。系统由两个分工策略组成：Proposer 提出一组短候选列表，Arbiter 选择新增一个索引或停止。训练时同时使用原名视图和匿名视图，使策略既能发挥LLM对查询的语义理解能力，也能学习与名称无关的结构信息。

仓库包含核心模型与策略代码、训练/评测数据、SFT best checkpoint，以及 GRPO `(30, 30)` 配对 checkpoint。Qwen 底座模型不包含在仓库中。

### 方法流程

1. **表示整个 workload。** 将 SQL 解析为参数化结构和谓词；使用 Qwen 分块编码查询文本；把 schema、谓词、连接关系和关系类型编码为有类型的关系图。数据库统计、候选索引信息、当前索引集合和剩余存储预算共同构成决策状态。
2. **融合语义与结构信息。** 关系感知图编码器表示结构节点和边。候选索引通过 cross-attention 查询语义 token、图节点、当前索引和预算/状态特征，得到融合表示。
3. **逐步做预算约束决策。** Proposer 最多提出三个候选，也可以提前结束 shortlist；Arbiter 选择一个索引或返回 `STOP`。存储和可行性状态参与动作决策。
4. **在成对视图上训练。** SFT 在原名和匿名视图上监督同一决策，并使用对称动作一致性损失和关系预测损失。随后使用实际 workload 执行时间与索引存储奖励，对 Proposer 和 Arbiter 做 GRPO 优化。

### 关键模块

| 模块 | 职责 |
|---|---|
| `structured_sql.py` | 将 SQL 解析为稳定的参数化查询结构和谓词。 |
| `relational_graph.py` | 构建关系图，并生成一致的匿名名称映射。 |
| `predicate_stats.py` | 读取 PostgreSQL 统计信息并估计谓词选择率。 |
| `candidate_pool.py` | 加载冻结的索引候选，并在不执行 DDL 的情况下处理预算/存储估计。 |
| `policy_inputs.py`, `structured_state.py` | 构建 workload 上下文和条件化候选特征。 |
| `semantic_structure_model.py` | Qwen LoRA 语义编码器、关系图编码器、cross-attention 融合层和角色动作头。 |
| `policy_data.py`, `structured_data/` | 构造成对视图的 SFT batch 和结构化图输入。 |
| `policy_losses.py` | 成对 SFT 动作损失、视图一致性损失和关系预测损失。 |
| `policy_actions.py` | 训练和推理共用的分类动作协议。 |
| `grpo_rollout.py`, `grpo_reward.py` | Proposer→Arbiter episode、PostgreSQL workload 计时和 GRPO 奖励计算。 |
| `recommendation_rollout.py`, `teacher_prompts.py` | 文本策略推荐流程和角色 prompt。 |
| `model_checkpoint.py` | 保存和恢复角色 LoRA adapter 与 FuseIA 网络权重。 |
| `scripts/recommend.py`, `scripts/evaluate.py` | 通用推荐 CLI 和串行实际查询评测 CLI。 |
| `db.py`, `indexes.py` | PostgreSQL/HypoPG 连接和索引集合工具。 |

### 目录结构

- `checkpoints/sft/{proposer,arbiter}/`：结构化 SFT best checkpoint。
- `checkpoints/grpo/{proposer,arbiter}/`：Proposer/Arbiter 对齐的 GRPO update 30 checkpoint。
- `data/workloads/{tpch,tpcds,job}/`：标准 workload。查询数和 SHA-256 见 `data/workloads/workload_registry.json`。
- `data/statistics/`, `data/index_candidates/`：schema/统计快照和候选索引定义。
- `data/structured_sft/`, `data/structured_teacher/`, `data/manifests/`：结构化训练样本、教师数据和数据划分清单。
- `structured_data/`：关系图拼 batch 和成对视图数据工具。

Checkpoint 保存 LoRA adapter 和 FuseIA 自身的网络权重，不包含 Qwen 底座。大型 checkpoint 和数据文件通过 Git LFS 管理。

### 环境安装

使用 Git LFS 克隆，以下载权重和大数据文件：

```bash
git lfs install
git clone https://github.com/pmss778/FuseIA.git
cd FuseIA
git lfs pull
```

创建 Python 环境并安装推理依赖。提供的 PyTorch wheel 面向 CUDA 12.8：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-local-inference.txt
```

调用 OpenAI-compatible 教师/API 或执行 PostgreSQL 实测时，还需要：

```bash
python -m pip install openai psycopg2-binary
```

数据库评测前设置 `PGHOST`、`PGPORT`、`PGUSER`、`PGPASSWORD`。使用假设索引规划时，目标 PostgreSQL 数据库需要安装 HypoPG。

### 下载 Qwen 底座模型

Checkpoint 使用 `Qwen/Qwen2.5-3B-Instruct`。需要从 Hugging Face 单独下载：

```bash
python -m pip install --upgrade huggingface_hub
hf download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct
```

Checkpoint loader 支持显式传入本地底座路径，不依赖训练机上记录的绝对路径：

```python
from model_checkpoint import load_checkpoint

proposer, proposer_metadata = load_checkpoint(
    "checkpoints/grpo/proposer",
    device="cuda",
    backbone_name="models/Qwen2.5-3B-Instruct",
)
proposer.eval()
```

SFT Proposer 使用 `checkpoints/sft/proposer`；Arbiter 使用相应的 `arbiter` 目录。当前 GRPO bundle 为 Proposer/Arbiter update 30 对齐组合。显存受限时，可用 `grpo_rollout.share_frozen_base` 共享两角色的冻结 Qwen 权重，同时保持各自 adapter 和 FuseIA 网络独立。

### 通用推荐与评测命令

`scripts/` 提供通用入口，避免每种 workload 都单独编写启动脚本。推荐结果和计时 JSON 示例写到仓库外的 `../FuseIA_runs/`，避免把运行产物加入 GitHub。

#### 生成 greedy 推荐

`scripts/recommend.py` 接收 workload CSV、候选池、schema/统计快照、Proposer/Arbiter checkpoint、预算和视图；按 greedy Proposer→Arbiter 协议生成索引方案，并保存逐步决策、checkpoint update、存储估算和运行信息。推理时两个角色轮流使用 GPU。

示例：TPC-DS SF1 无主键库，GRPO `(30,30)` checkpoint，20% 预算 204.8 MiB：

```bash
python scripts/recommend.py \
  --workload data/workloads/tpcds/sf1/tpcds83_pg_paper.csv \
  --candidate-pool data/index_candidates/tpcds_sf1/candidates_tpds_llm.json \
  --schema data/statistics/tpcds_sf1_nopk/schema.json \
  --stats data/statistics/tpcds_sf1_nopk/pg_stats.json \
  --proposer-checkpoint checkpoints/grpo/proposer \
  --arbiter-checkpoint checkpoints/grpo/arbiter \
  --backbone models/Qwen2.5-3B-Instruct \
  --budget-mib 204.8 \
  --view original \
  --output ../FuseIA_runs/tpcds_fuseia_recommendation.json
```

使用 `--view anonymous` 可让模型从匿名名称视图推理；TPC-DS SF1 100% 预算使用 `--budget-mib 1024`。SFT 推荐时，将两个 checkpoint 参数替换为 `checkpoints/sft/proposer` 和 `checkpoints/sft/arbiter`。`--feasibility-mask` 会在动作时屏蔽超预算候选；默认保留训练动作协议，并将 Arbiter 选中不可行索引的情况报告为停止原因。

#### 实测一个或多个推荐方案

`scripts/evaluate.py` 按顺序读取一个或多个推荐 JSON，在数据库中实际创建索引并执行 workload，输出逐查询中位数、总 workload 时间、物理索引大小和失败信息。脚本只创建和删除本次运行专属的 `ix_<namespace>_` 索引，不触碰其他索引。

```bash
python scripts/evaluate.py \
  --database indexselection_tpcds___1_no_pk \
  --workload data/workloads/tpcds/sf1/tpcds83_pg_paper.csv \
  --recommendation ../FuseIA_runs/tpcds_fuseia_recommendation.json \
  --recommendation ../FuseIA_runs/tpcds_teacher_recommendation.json \
  --runs 3 \
  --statement-timeout-seconds 600 \
  --output ../FuseIA_runs/tpcds_comparison.json
```

请只在测试数据库上运行。`--statement-timeout-seconds 0` 表示不设查询超时。输出文件采用只写一次策略；重复实验请指定新的文件名。

#### Checkpoint 加载 API

统一 loader 会用指定的本地 Qwen 路径覆盖 checkpoint 内的训练机绝对路径：

```python
from model_checkpoint import load_checkpoint

proposer, proposer_metadata = load_checkpoint(
    "checkpoints/grpo/proposer",
    device="cuda",
    backbone_name="models/Qwen2.5-3B-Instruct",
)
```

<p align="right"><a href="#english">English</a> | <a href="#简体中文">简体中文</a></p>
