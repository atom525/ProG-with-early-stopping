# 如何以模块化方式添加新提示模型 (RecBole 风格)

本项目已实现 RecBole 风格的**注册表机制**，添加新提示模型时支持两种方式：

1. **完全插件化**：注册 `train_fn`、`optimizer_init_fn`，无需修改 Task 代码
2. **部分插件化**：注册 Prompt 类、Evaluator，在 Task 中增加训练/优化器分支

---

## 1. 注册表组件

| 组件 | 作用 |
|------|------|
| `PromptRegistry` | 统一注册 Prompt 类、Evaluator、train_fn、optimizer_init_fn、数据模式 |
| `DataMode` | `NODE_FULL`（全图+idx）、`GRAPH_INDUCED`（诱导子图）、`MULTI_SPECIAL`（自定义） |
| `registry_config.py` | 内置所有 Prompt 的注册配置 |

---

## 2. 完全插件化：train_fn + optimizer_init_fn

当注册了 `train_fn` 时，Task 的 `run()` 会在每个 epoch 调用它，**无需在 node_task.py / graph_task.py 中增加任何分支**。

### 2.1 train_fn 签名与 EpochContext

```python
def my_train_fn(task, epoch_context: dict):
    """
    Args:
        task: BaseTask 实例，可访问 task.gnn, task.prompt, task.answering, task.device 等
        epoch_context: 当前 epoch 的上下文，字段见下表
    Returns:
        - loss: float  # 仅返回 loss
        - (loss, extra_dict): tuple  # extra_dict 可包含 'center' 用于 Gprompt 类 center 更新
    """
    ...
    return loss  # 或 (loss, {'center': center_tensor})
```

**EpochContext 字段（部分 key 依 task_type / prompt_type 可能为 None）：**

| 字段 | NodeTask | GraphTask | 说明 |
|------|----------|-----------|------|
| `epoch` | ✓ | ✓ | 当前 epoch 数 |
| `train_loader` | ✓ | ✓ | DataLoader |
| `valid_loader` | ✓ | ✓ | DataLoader 或 None |
| `test_loader` | ✓ | ✓ | DataLoader |
| `data` | ✓ | - | 全图 Data（NodeTask） |
| `idx_train`, `idx_valid`, `idx_test` | ✓ | - | 节点索引 |
| `train_dataset`, `valid_dataset`, `test_dataset` | - | ✓ | 图列表 |
| `train_embs`, `train_embs1`, `train_lbls_mg` | - | MultiGprompt | 图级 MultiGprompt |
| `valid_embs`, `valid_embs1`, `valid_lbls_mg` | - | MultiGprompt | 同上 |
| `test_embs`, `test_embs1`, `test_lbls_mg` | - | MultiGprompt | 同上 |
| `pretrain_embs` | MultiGprompt | - | 节点级 MultiGprompt |
| `train_lbls` | ✓ | - | 训练标签 |
| `best_center` | ✓ | ✓ | Gprompt 类 center，可更新 |
| `answer_epoch`, `prompt_epoch` | ✓ | ✓ | All-in-one 等 |

### 2.2 optimizer_init_fn 签名

```python
def my_optimizer_init_fn(task):
    """在 BaseTask.initialize_optimizer() 时调用，应设置 task.optimizer（及可选的 task.pg_opi, task.answer_opi）"""
    task.optimizer = optim.Adam([...], lr=task.lr, weight_decay=task.wd)
```

### 2.3 完全插件化注册示例

```python
# registry_config.py

def my_train_fn(task, ctx):
    task.gnn.train()
    task.prompt.train()
    task.optimizer.zero_grad()
    total_loss = 0.0
    for batch in ctx['train_loader']:
        batch = batch.to(task.device)
        # GPF 风格需先 batch.x = task.prompt.add(batch.x)；其他 prompt 按需
        out = task.gnn(batch.x, batch.edge_index, batch.batch, prompt=task.prompt, prompt_type=task.prompt_type)
        out = task.answering(out)
        loss = task.criterion(out, batch.y)
        loss.backward()
        task.optimizer.step()
        total_loss += loss.item()
    return total_loss / len(ctx['train_loader'])

def my_optimizer_init_fn(task):
    from torch import optim
    task.optimizer = optim.Adam(
        list(task.prompt.parameters()) + list(task.answering.parameters()),
        lr=task.lr, weight_decay=task.wd
    )

PromptRegistry.register_prompt(
    'MyPrompt',
    MyPrompt,
    data_mode=DataMode.GRAPH_INDUCED,
    needs_center=False,
    train_fn=my_train_fn,
    optimizer_init_fn=my_optimizer_init_fn,
    init_kwargs_fn=lambda t: {'in_channels': t.input_dim}  # GPF 用 in_channels；其他 Prompt 按构造函数参数名
)
# 仍需注册 evaluator
PromptRegistry.register_evaluator('MyPrompt', 'NodeTask', my_eval)
PromptRegistry.register_evaluator('MyPrompt', 'GraphTask', my_eval)
```

---

## 3. 部分插件化：复用或扩展 Task 分支

若训练逻辑与 GPF、GPPT 等相似，可：

- **复用 train_method**：注册 `train_method='GPFTrain'`，但需在 Task 中增加 `elif self.prompt_type == 'MyPrompt'` 分支并调用 `self.GPFTrain(...)`（或复用同一分支）
- **复用 evaluator**：直接 `register_evaluator('MyPrompt', 'NodeTask', node_gpf_eval)`

### 3.1 在 Task 中增加分支

当不注册 `train_fn` 时，需在 `node_task.py` 或 `graph_task.py` 的 `run()` 中增加：

```python
elif self.prompt_type == 'MyPrompt':
    loss = self.MyPromptTrain(train_loader)
```

并在 `task.py` 的 `initialize_optimizer()` 中增加：

```python
elif self.prompt_type == 'MyPrompt':
    self.optimizer = optim.Adam([...], lr=self.lr, weight_decay=self.wd)
```

---

## 4. 实现 Prompt 类与 Evaluator

### 4.1 Prompt 类

在 `prompt_graph/prompt/` 下创建，继承 `nn.Module`。`init_kwargs_fn` 用于从 Task 传入初始化参数。

### 4.2 Evaluator

在 `prompt_graph/evaluation/` 下创建，统一适配器签名：

```python
def my_eval(loader, data, idx, gnn, prompt, answering, output_dim, device, **extra):
    # 返回 (acc, f1, roc, prc)
    ...
```

可从 `extra` 中获取自定义参数（如 MultiGprompt 的 `valid_embs`、`DownPrompt` 等）。

---

## 5. 数据加载

| DataMode | 说明 |
|----------|------|
| `NODE_FULL` | 全图 + `idx_train/valid/test`，不加载 `graphs_list` |
| `GRAPH_INDUCED` | 使用 `load_induced_graph` 和 `graphs_list`，由 `needs_induced_graph()` 控制 |
| `MULTI_SPECIAL` | 如 MultiGprompt，在 Task 中单独实现数据加载 |

`downstream_task.py` 按 `PromptRegistry.needs_induced_graph(args.prompt_type)` 自动选择是否加载诱导图。

---

## 6. 复用示例：GPF 风格轻量 Prompt

若训练和评估与 GPF 相同：

```python
PromptRegistry.register_prompt('MyLightPrompt', MyLightPrompt, DataMode.GRAPH_INDUCED,
    train_method='GPFTrain', init_kwargs_fn=lambda t: {'in_channels': t.input_dim})
PromptRegistry.register_evaluator('MyLightPrompt', 'NodeTask', node_gpf_eval)
PromptRegistry.register_evaluator('MyLightPrompt', 'GraphTask', graph_gpf_eval)
```

仍需在 Task 的 `run()` 和 `initialize_optimizer()` 中为 `MyLightPrompt` 增加与 GPF 相同的分支（或通过 `train_fn` 实现完全插件化）。

---

## 7. 总结

| 方式 | 需修改 Task | 灵活性 |
|------|-------------|--------|
| **train_fn + optimizer_init_fn** | 否 | 完全插件化 |
| **train_method + Task 分支** | 是（run + initialize_optimizer） | 部分插件化 |
| **复用 GPF/GPPT 等** | 是（增加 elif 分支） | 最快 |

优先使用 `train_fn` + `optimizer_init_fn` 实现完全插件化；无法满足时再使用 Task 分支方式。
