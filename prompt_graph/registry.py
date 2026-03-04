"""
RecBole-style Registry for modular prompt/evaluator/train integration.
Add new prompt models by registering in your module - no need to modify task/train/eval core code.

Pluggable Components:
- prompt_class + init_kwargs_fn: Prompt initialization
- evaluator: (loader, data, idx, gnn, prompt, answering, output_dim, device, **extra) -> (acc, f1, roc, prc)
- train_fn: (task, epoch_context) -> loss | (loss, extra_dict)  # invoked each training epoch
- optimizer_init_fn: (task) -> None  # invoked once before training
"""

from typing import Callable, Dict, Any, Optional, Tuple, Union
from enum import Enum


# ---------------------------------------------------------------------------
# EpochContext: 传递给 train_fn 的上下文，包含当前 epoch 可用的数据与状态
# ---------------------------------------------------------------------------
#  schema (根据 task_type 和 prompt_type，部分 key 可能为 None):
#   - epoch: int
#   - train_loader: DataLoader | None
#   - valid_loader: DataLoader | None
#   - test_loader: DataLoader | None
#   - data: Data (全图，NodeTask) | None
#   - idx_train, idx_valid, idx_test: LongTensor | None
#   - train_embs, train_embs1, train_lbls_mg: (GraphTask MultiGprompt) | None
#   - pretrain_embs, train_lbls: (NodeTask MultiGprompt) | None
#   - answer_epoch, prompt_epoch: int (All-in-one)
#   - best_center: Tensor | None (Gprompt 等)
# train_fn 可从 context 中提取所需字段，无需关心未使用字段
# ---------------------------------------------------------------------------
# train_fn 返回值:
#   - loss: float  -> 仅返回 loss
#   - (loss, extra_dict): tuple  -> extra_dict 可包含:
#       - 'center': Tensor 用于 Gprompt 类 center 更新，Task 会赋给 best_center
# ---------------------------------------------------------------------------


class DataMode(str, Enum):
    """Data loading mode for prompt types."""
    NODE_FULL = "node_full"       # Full graph + idx_train/valid/test (GPPT, None)
    GRAPH_INDUCED = "graph_induced"  # Induced subgraphs (Gprompt, All-in-one, GPF, GPF-plus)
    MULTI_SPECIAL = "multi_special"  # MultiGprompt etc. - custom data path


class PromptRegistry:
    """
    Central registry for prompt models.
    Register: prompt_class, evaluator, train_fn, data_mode, and optional metadata.
    """

    # prompt_type -> prompt class (or factory callable)
    _prompt_classes: Dict[str, Any] = {}
    # (prompt_type, task_type) -> evaluator adapter: (task, loader, data, idx, extra) -> (acc, f1, roc, prc)
    _evaluators: Dict[Tuple[str, str], Callable] = {}
    # prompt_type -> train method name on Task (e.g. "GPFTrain", "GpromptTrain") for built-in prompts
    _train_methods: Dict[str, str] = {}
    # prompt_type -> optional external train_fn(task, epoch_context) -> loss or (loss, extra_dict) for pluggable prompts
    _train_fns: Dict[str, Callable] = {}
    # prompt_type -> DataMode
    _data_modes: Dict[str, DataMode] = {}
    # prompt_type -> bool (needs center embedding, e.g. Gprompt)
    _needs_center: Dict[str, bool] = {}
    # prompt_type -> init kwargs override (optional, for dataset-specific params)
    _prompt_init_kwargs: Dict[str, Callable] = {}
    # prompt_type -> optimizer_init_fn(task) -> None，用于可插拔 prompt 的优化器初始化
    _optimizer_init_fns: Dict[str, Callable] = {}

    @classmethod
    def register_prompt(cls, prompt_type: str, prompt_class: Any, data_mode: DataMode = DataMode.NODE_FULL,
                       needs_center: bool = False, train_method: Optional[str] = None,
                       train_fn: Optional[Callable] = None, init_kwargs_fn: Optional[Callable[[Any], dict]] = None,
                       optimizer_init_fn: Optional[Callable[[Any], None]] = None):
        """
        Register a prompt model.

        - train_fn: Fully pluggable. Takes (task, epoch_context) -> loss | (loss, extra_dict).
          When registered, Task.run() will call it each epoch instead of hardcoded branches.
        - train_method: Legacy/documentation. Name of Task method for built-in prompts.
        - optimizer_init_fn: When set, BaseTask.initialize_optimizer() will call it for this prompt_type
          instead of hardcoded branches. Signature: (task) -> None.
        """
        cls._prompt_classes[prompt_type] = prompt_class
        cls._data_modes[prompt_type] = data_mode
        cls._needs_center[prompt_type] = needs_center
        if train_method:
            cls._train_methods[prompt_type] = train_method
        if train_fn:
            cls._train_fns[prompt_type] = train_fn
        if init_kwargs_fn:
            cls._prompt_init_kwargs[prompt_type] = init_kwargs_fn
        if optimizer_init_fn:
            cls._optimizer_init_fns[prompt_type] = optimizer_init_fn

    @classmethod
    def register_evaluator(cls, prompt_type: str, task_type: str, evaluator_fn: Callable):
        """Register evaluator for (prompt_type, task_type). Evaluator: fn(loader, data, idx, gnn, prompt, answering, output_dim, device, **extra) -> (acc, f1, roc, prc)."""
        cls._evaluators[(prompt_type, task_type)] = evaluator_fn

    @classmethod
    def get_prompt_class(cls, prompt_type: str):
        return cls._prompt_classes.get(prompt_type)

    @classmethod
    def get_evaluator(cls, prompt_type: str, task_type: str) -> Optional[Callable]:
        return cls._evaluators.get((prompt_type, task_type))

    @classmethod
    def get_train_method(cls, prompt_type: str) -> Optional[str]:
        return cls._train_methods.get(prompt_type)

    @classmethod
    def get_train_fn(cls, prompt_type: str) -> Optional[Callable]:
        """
        External train function for pluggable prompts.
        Signature: (task, epoch_context: dict) -> float | (float, dict)
        When returns (loss, extra_dict), extra_dict may contain 'center' for Gprompt-style updates.
        """
        return cls._train_fns.get(prompt_type)

    @classmethod
    def get_optimizer_init_fn(cls, prompt_type: str) -> Optional[Callable]:
        """
        Optimizer init function for pluggable prompts.
        Signature: (task) -> None. Should set task.optimizer (and optionally task.pg_opi, task.answer_opi).
        """
        return cls._optimizer_init_fns.get(prompt_type)

    @classmethod
    def get_data_mode(cls, prompt_type: str) -> DataMode:
        return cls._data_modes.get(prompt_type, DataMode.NODE_FULL)

    @classmethod
    def needs_induced_graph(cls, prompt_type: str) -> bool:
        return cls._data_modes.get(prompt_type, DataMode.NODE_FULL) == DataMode.GRAPH_INDUCED

    @classmethod
    def needs_center(cls, prompt_type: str) -> bool:
        return cls._needs_center.get(prompt_type, False)

    @classmethod
    def get_prompt_init_kwargs(cls, prompt_type: str, task: Any) -> dict:
        fn = cls._prompt_init_kwargs.get(prompt_type)
        return fn(task) if fn else {}

    @classmethod
    def list_prompts(cls) -> list:
        return list(cls._prompt_classes.keys())
