# 如何以模块化方式添加新提示模型 (RecBole 风格)

本项目已实现 RecBole 风格的**注册表机制**，添加新提示模型时只需：

1. 编写新的 Prompt 类和 Evaluator
2. 在 `registry_config.py` 中注册
3. （可选）在 Task 中实现训练逻辑

**无需修改** `downstream_task.py`、`BaseTask.initialize_prompt` 及评估分支中的硬编码逻辑。

---

## 1. 注册表组件

| 组件 | 作用 |
|------|------|
| `PromptRegistry` | 统一注册 Prompt 类、Evaluator、数据模式、训练方法 |
| `DataMode` | `NODE_FULL`（全图+idx）、`GRAPH_INDUCED`（诱导子图）、`MULTI_SPECIAL`（自定义） |
| `registry_config.py` | 内置所有 Prompt 的注册配置 |

---

## 2. 添加新提示的步骤

### 步骤 1：实现 Prompt 类

在 `prompt_graph/prompt/` 下创建新文件，例如 `MyPrompt.py`：

```python
class MyPrompt(torch.nn.Module):
    def __init__(self, input_dim, ...):
        ...
```

### 步骤 2：实现 Evaluator

在 `prompt_graph/evaluation/` 下创建 `MyPromptEva.py`，实现与现有 Evaluator 一致的签名：

```python
def MyPromptEva(loader, gnn, prompt, answering, num_class, device, **kwargs):
    # 返回 (acc, f1, roc, prc)
    ...
```

或适配器：`(loader, data, idx, gnn, prompt, answering, output_dim, device, **extra) -> (acc,f1,roc,prc)`

### 步骤 3：在 registry_config.py 中注册

```python
# 在 _register_prompts() 中
PromptRegistry.register_prompt(
    'MyPrompt',
    MyPrompt,
    data_mode=DataMode.GRAPH_INDUCED,   # 或 NODE_FULL
    needs_center=False,
    train_method='MyPromptTrain',       # Task 中的方法名
    init_kwargs_fn=lambda t: {'input_dim': t.input_dim, ...}
)

# 在 _register_evaluators() 中
def my_eval(loader, data, idx, gnn, prompt, answering, output_dim, device, **extra):
    return MyPromptEva(loader, gnn, prompt, answering, output_dim, device)

PromptRegistry.register_evaluator('MyPrompt', 'NodeTask', my_eval)
PromptRegistry.register_evaluator('MyPrompt', 'GraphTask', my_eval)
```

### 步骤 4：实现训练逻辑（二选一）

**方式 A：在 NodeTask/GraphTask 中增加方法**

在 `node_task.py` 或 `graph_task.py` 的 `run()` 中增加分支，并对应 `train_method`：

```python
elif self.prompt_type == 'MyPrompt':
    loss = self.MyPromptTrain(train_loader)
```

**方式 B：注册外部 train_fn（完全插件化）**

```python
def my_train_fn(task, epoch_context):
    # epoch_context 包含 train_loader, train_idx, data 等
    ...
    return loss  # 或 (loss, {'center': ...})

PromptRegistry.register_prompt(..., train_fn=my_train_fn)
```

---

## 3. 数据加载

- **NODE_FULL**：使用全图 + `idx_train/valid/test`，不加载 `graphs_list`
- **GRAPH_INDUCED**：使用 `load_induced_graph` 和 `graphs_list`，由 `PromptRegistry.needs_induced_graph()` 控制
- **MULTI_SPECIAL**：如 MultiGprompt，在 Task 中单独实现数据加载逻辑

`downstream_task.py` 已按 `PromptRegistry.needs_induced_graph(args.prompt_type)` 自动选择是否加载诱导图。

---

## 4. 复用现有模块

| 模块 | 复用方式 |
|------|----------|
| 评估 | 所有 Evaluator 统一通过 `PromptRegistry.get_evaluator()` 调用 |
| 数据划分 | `load4node`、`load4graph`、`split_induced_graphs` 保持不变 |
| 早停/checkpoint | Task 内部逻辑不变，新 Prompt 自动复用 |
| 日志 | `train_info` 等统一 logger |

---

## 5. 示例：添加一个 GPF 风格的轻量 Prompt

```python
# registry_config.py 中
PromptRegistry.register_prompt('MyLightPrompt', MyLightPrompt, DataMode.GRAPH_INDUCED,
    train_method='GPFTrain')  # 若与 GPF 训练逻辑相同，可直接复用
PromptRegistry.register_evaluator('MyLightPrompt', 'NodeTask', node_gpf_eval)  # 复用 GPF evaluator
```

若训练和评估与 GPF 相同，只需注册 Prompt 类和复用 GPF 的 evaluator/train，即可获得完整流水线。
