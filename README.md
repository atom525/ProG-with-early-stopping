# ProG 使用说明

> 本项目基于 [ProG](https://github.com/sheldonresearch/ProG) 进行 RecBole 风格改造，用于图预训练与提示学习。

## 概述

本文档说明改造后的 ProG 框架，参考 RecBole 的设计，支持：

- **自定义数据划分比例**：训练/验证/测试集按 `[train_ratio, valid_ratio, test_ratio]` 划分
- **早停策略**：基于验证集早停，可配置 `patience`
- **自动保存最佳权重**：在验证集上最优时保存，最终在测试集上评估
- **统一配置**：支持 YAML 配置文件 `config/config.yaml`
- **模块化 Prompt**：RecBole 风格注册表，新增 Prompt 无需改动核心代码，参见 `docs/ADD_NEW_PROMPT.md`

---

## 快速开始

### 1. 数据准备

**推荐先用 MUTAG 测试（图级，下载快，无需 GitHub）：**
```bash
cd /path/to/ProG-with-early-stopping
python scripts/download_data.py --data_root data --datasets MUTAG
```

或下载全部：
```bash
python scripts/download_data.py --data_root data
```
若 Planetoid（Cora 等）下载失败，可加 `-v` 查看详细错误，参见下方「下载失败时」。

### 2. 预训练

```bash
python pre_train.py \
    --pretrain_task Edgepred_Gprompt \
    --dataset_name Cora \
    --gnn_type GCN \
    --hid_dim 128 \
    --num_layer 2 \
    --epochs 1000 \
    --device 0
```

### 3. 下游任务

预训练权重会保存到 `checkpoints/pretrain/{dataset}/`，文件名带时间戳，例如：
`checkpoints/pretrain/Cora/Edgepred_Gprompt.GCN.128hidden_dim_20250303_123456.pth`

```bash
# 使用实际保存的权重路径（带时间戳）
python downstream_task.py \
    --downstream_task NodeTask \
    --pre_train_model_path checkpoints/pretrain/Cora/Edgepred_Gprompt.GCN.128hidden_dim_*.pth \
    --dataset_name Cora \
    --gnn_type GCN \
    --prompt_type GPF-plus \
    --shot_num 1 \
    --split_ratio 0.8 0.1 0.1 \
    --patience 20 \
    --device 0
```

---

## 核心参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--split_ratio` | 训练/验证/测试划分比例 | 0.8 0.1 0.1 |
| `--patience` | 早停耐心值 | 10 |
| `--save_best` | 是否保存最佳 checkpoint | True |
| `--checkpoint_dir` | 下游 checkpoint 保存目录 | checkpoints/downstream |
| `--config_file` | YAML 配置文件路径 | config/config.yaml |
| `--eval_every` | 每 N 轮在验证集上评估 | 1 |
| `--early_stopping_metric` | 早停指标：valid_acc / valid_f1 / valid_auroc / valid_auprc | valid_acc |

### 项目目录结构

| 目录 | 用途 |
|------|------|
| `data/` | 原始数据（TUDataset、Planetoid 等） |
| `checkpoints/pretrain/` | 预训练权重 |
| `checkpoints/downstream/` | 下游任务 checkpoint |
| `logs/` | 日志文件 |
| `outputs/sample_data/` | few-shot 采样数据 |
| `outputs/induced_graph/` | 诱导子图缓存 |

---

## 数据集下载链接

### 节点级数据集

| 数据集 | 来源 | 下载链接 |
|--------|------|----------|
| Cora | Planetoid | https://github.com/kimiyoung/planetoid/raw/master/data |
| CiteSeer | Planetoid | 同上 |
| PubMed | Planetoid | 同上 |
| Computers | Amazon | PyG 自动下载 |
| Photo | Amazon | PyG 自动下载 |
| Reddit | Reddit | PyG 自动下载 |
| WikiCS | WikiCS | PyG 自动下载 |
| Flickr | Flickr | PyG 自动下载 |
| Wisconsin | WebKB | PyG 自动下载 |
| Texas | WebKB | PyG 自动下载 |
| Actor | Actor | PyG 自动下载 |
| ogbn-arxiv | OGB | https://ogb.stanford.edu/docs/nodeprop/ |

### 图级数据集 (TUDataset)

| 数据集 | 下载链接 |
|--------|----------|
| MUTAG | https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip |
| ENZYMES | https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip |
| PROTEINS | https://www.chrsmrrs.com/graphkerneldatasets/PROTEINS.zip |
| COLLAB | https://www.chrsmrrs.com/graphkerneldatasets/COLLAB.zip |
| IMDB-BINARY | https://www.chrsmrrs.com/graphkerneldatasets/IMDB-BINARY.zip |
| REDDIT-BINARY | https://www.chrsmrrs.com/graphkerneldatasets/REDDIT-BINARY.zip |
| COX2 | https://www.chrsmrrs.com/graphkerneldatasets/COX2.zip |
| BZR | https://www.chrsmrrs.com/graphkerneldatasets/BZR.zip |
| PTC_MR | https://www.chrsmrrs.com/graphkerneldatasets/PTC_MR.zip |
| DD | https://www.chrsmrrs.com/graphkerneldatasets/DD.zip |

### 图级数据集 (OGB)

| 数据集 | 下载链接 |
|--------|----------|
| ogbg-ppa | https://ogb.stanford.edu/docs/graphprop/ |
| ogbg-molhiv | 同上 |
| ogbg-molpcba | 同上 |
| ogbg-code2 | 同上 |

---

## 下载失败时

若自动下载超时（如 `FSTimeoutError`、`Connection timeout to host https://github.com/...`），可手动下载后放入对应目录。

**Planetoid (Cora / CiteSeer / PubMed)**  
1. 从 [planetoid/data](https://github.com/kimiyoung/planetoid/tree/master/data) 下载对应 `ind.{name}.x`、`ind.{name}.tx`、`ind.{name}.allx`、`ind.{name}.tallx`、`ind.{name}.y`、`ind.{name}.ty`、`ind.{name}.ally`、`ind.{name}.tally`、`ind.{name}.graph`、`ind.{name}.test.index`  
2. 放入 `data/Planetoid/raw/` 目录（按需创建）  
3. 重新运行 `python scripts/download_data.py --datasets Cora`，PyG 会检测到已存在 raw 文件并跳过下载

**TUDataset**  
将对应 zip 解压到 `data/TUDataset/`，或放入 raw 目录后由 PyG 处理。

---

## 配置说明

通过 `--config_file config/config.yaml` 可加载 YAML 配置，**未在命令行显式传入的参数** 会使用配置文件中的值。

### 配置文件：`config/config.yaml`

```yaml
# 路径
data_root: data
checkpoint_dir: checkpoints/downstream
log_dir: logs

# 数据划分 [train, valid, test]
split_ratio: [0.8, 0.1, 0.1]

# 早停与验证
patience: 10                    # 连续多少轮验证指标不提升则停止
early_stopping_metric: valid_acc # valid_acc / valid_f1 / valid_auroc / valid_auprc（均越大越好）
evaluate_every: 1               # 每多少 epoch 在验证集上评估一次并打印

# 模型保存（验证集指标最优时保存 checkpoint）
save_best: true

# 训练
epochs: 1000
batch_size: 128
lr: 0.001
weight_decay: 0.0
```

### 训练输出与日志

- **每轮打印 loss**：每个 epoch 结束打印 `train_loss`
- **定期验证**：每 `evaluate_every` 个 epoch 在验证集评估，打印 `val_{metric}`
- **早停**：`early_stopping_metric` 在验证集上连续 `patience` 轮不改善则停止
- **保存最佳**：验证集指标最优时保存 checkpoint
- **最终测试**：早停后加载最佳 checkpoint，在测试集上评估
- **日志**：所有 `print` 同时写入控制台和 `log_dir` 下自动生成的 `.log` 文件

### 如何使用

1. **完全用配置文件**：只传 `--config_file`，其余从 YAML 读
   ```bash
   python downstream_task.py --downstream_task GraphTask --pre_train_model_path xxx --dataset_name MUTAG --config_file config/config.yaml
   ```

2. **命令行覆盖**：命令行参数优先于配置文件
   ```bash
   python downstream_task.py ... --split_ratio 0.7 0.15 0.15 --patience 30 --config_file config/config.yaml
   ```
   则 `split_ratio`、`patience` 使用命令行值。

3. **修改配置**：直接编辑 `config/config.yaml`，无需改代码。

---

## 依赖

```bash
pip install -r requirements.txt
```

除 ProG 原有依赖外，若使用 YAML 配置需安装 PyYAML（已包含在 requirements.txt 中）。

**若出现 `GLIBCXX_3.4.29' not found`**，使用 conda 自带的 libstdc++：
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## 全流程测试

**方式一：一键脚本**
```bash
cd /path/to/ProG-with-early-stopping
chmod +x scripts/run_mutag_full.sh
./scripts/run_mutag_full.sh
```

**方式二：分步执行（MUTAG 图级任务）**
```bash
cd /path/to/ProG-with-early-stopping

# 1. 下载数据（约 5 秒）
python scripts/download_data.py --data_root data --datasets MUTAG

# 2. 预训练
python pre_train.py \
    --pretrain_task GraphMultiGprompt \
    --dataset_name MUTAG \
    --gnn_type GIN \
    --hid_dim 128 \
    --num_layer 2 \
    --epochs 500 \
    --device 0 \
    --config_file config/config.yaml

# 3. 下游任务
python downstream_task.py \
    --downstream_task GraphTask \
    --pre_train_model_path checkpoints/pretrain/MUTAG/MultiGprompt.pth \
    --dataset_name MUTAG \
    --gnn_type GIN \
    --prompt_type Gprompt \
    --shot_num 5 \
    --split_ratio 0.8 0.1 0.1 \
    --patience 20 \
    --epochs 500 \
    --device 0 \
    --config_file config/config.yaml
```

**节点级任务（Cora，需能访问 GitHub）：**
```bash
python scripts/download_data.py --data_root data --datasets Cora
python pre_train.py --pretrain_task Edgepred_Gprompt --dataset_name Cora --gnn_type GCN --epochs 500 --device 0
python downstream_task.py --downstream_task NodeTask --pre_train_model_path checkpoints/pretrain/Cora/Edgepred_Gprompt.GCN.128hidden_dim.pth --dataset_name Cora --prompt_type GPF-plus --shot_num 1 --device 0
```

---

## 路径规范（固定目录）

| 目录 | 用途 |
|------|------|
| `data/` | 数据集（Planetoid、TUDataset 等） |
| `checkpoints/pretrain/` | 预训练权重 |
| `checkpoints/downstream/` | 下游任务 checkpoint |
| `logs/` | 训练/评估日志 |
| `outputs/sample_data/` | 样本划分（few-shot） |
| `outputs/induced_graph/` | 诱导子图 |

以上路径在 `config/config.yaml` 中配置，可通过 `--checkpoint_dir` 等参数覆盖。

---

## 目录结构

```
ProG/
├── config/              # RecBole 风格配置
│   ├── config.yaml
│   └── configurator.py
├── docs/                # 文档
│   └── ADD_NEW_PROMPT.md   # 新增 Prompt 模块化指南
├── prompt_graph/        # 核心模块
│   ├── data/            # 数据加载与划分
│   ├── tasker/          # NodeTask, GraphTask, LinkTask
│   ├── model/           # GNN 模型
│   ├── prompt/          # Prompt 模块
│   ├── pretrain/        # 预训练策略
│   ├── evaluation/      # 评估
│   ├── registry.py      # RecBole 风格注册表
│   └── registry_config.py   # Prompt/Evaluator 注册配置
├── scripts/
│   └── download_data.py
├── pre_train.py
├── downstream_task.py
└── README.md
```
