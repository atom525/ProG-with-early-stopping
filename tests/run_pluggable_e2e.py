#!/usr/bin/env python
"""
E2E 测试：可插拔 train_fn + optimizer_init_fn 全流程
1. 注册带 train_fn 的测试 prompt
2. 运行下游任务（GraphTask MUTAG，2 epochs）
3. 验证无报错并清理生成的 checkpoints/logs
"""
import os
import sys
import shutil
import tempfile
from datetime import datetime

# 确保项目根目录在 path 中
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# 导入后 setup_registry 已执行，再注册测试 prompt
import torch
from torch import optim
from prompt_graph.registry import PromptRegistry, DataMode
from prompt_graph.registry_config import setup_registry
from prompt_graph.prompt import GPF
from prompt_graph.evaluation import GPFEva
from prompt_graph.utils import seed_everything
from prompt_graph.data import load4graph
from prompt_graph.tasker import GraphTask

TEST_PROMPT = '_TestPluggableE2E_'


def _opt_init(task):
    """与 GPF 相同的优化器初始化"""
    task.optimizer = optim.Adam(
        list(task.prompt.parameters()) + list(task.answering.parameters()),
        lr=task.lr, weight_decay=task.wd
    )


def _train_fn(task, ctx):
    """与 GPFTrain 相同的训练逻辑"""
    return task.GPFTrain(ctx['train_loader'])


def _graph_eval(loader, data, idx, gnn, prompt, answering, output_dim, device, **extra):
    return GPFEva(loader, gnn, prompt, answering, output_dim, device)


def register_test_prompt():
    """注册测试用可插拔 prompt"""
    PromptRegistry.register_prompt(
        TEST_PROMPT,
        GPF,
        data_mode=DataMode.GRAPH_INDUCED,
        needs_center=False,
        train_fn=_train_fn,
        optimizer_init_fn=_opt_init,
        init_kwargs_fn=lambda t: {'in_channels': t.input_dim}
    )
    PromptRegistry.register_evaluator(TEST_PROMPT, 'GraphTask', _graph_eval)


def unregister_test_prompt():
    """移除测试 prompt 注册"""
    for d in [PromptRegistry._prompt_classes, PromptRegistry._data_modes, PromptRegistry._needs_center,
              PromptRegistry._train_fns, PromptRegistry._optimizer_init_fns, PromptRegistry._prompt_init_kwargs]:
        d.pop(TEST_PROMPT, None)
    PromptRegistry._evaluators.pop((TEST_PROMPT, 'GraphTask'), None)


def main():
    print("=" * 50)
    print("E2E Test: Pluggable train_fn + optimizer_init_fn")
    print("=" * 50)

    # 使用临时目录，便于清理
    ckpt_dir = os.path.join(ROOT, 'checkpoints', 'downstream', f'MUTAG_{TEST_PROMPT}_e2etest')
    log_dir = os.path.join(ROOT, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'downstream_GraphTask_MUTAG_{datetime.now().strftime("%Y%m%d_%H%M%S")}_e2etest.log')

    register_test_prompt()
    try:
        seed_everything(42)
        print("Loading MUTAG...")
        input_dim, output_dim, dataset = load4graph('MUTAG', 5)
        print("Creating GraphTask with pluggable prompt...")
        device_id = 0 if torch.cuda.is_available() else 'cpu'
        tasker = GraphTask(
            dataset_name='MUTAG', num_layer=2, gnn_type='GIN', hid_dim=128,
            prompt_type=TEST_PROMPT, epochs=2, shot_num=5, device=device_id,
            lr=0.001, wd=0, batch_size=32, dataset=dataset,
            input_dim=input_dim, output_dim=output_dim,
            split_ratio=[0.8, 0.1, 0.1], patience=20, save_best=True,
            checkpoint_dir=ckpt_dir, eval_every=1, early_stopping_metric='valid_acc',
        )
        print("Running task.run() (2 epochs)...")
        _, test_acc, std_acc, f1, std_f1, roc, std_roc, prc, std_prc = tasker.run()
        print(f"Done. test_acc={test_acc:.4f}, f1={f1:.4f}")
    finally:
        unregister_test_prompt()

    # 清理
    print("Cleaning up test artifacts...")
    if os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        print(f"  Removed {ckpt_dir}")
    if os.path.isfile(log_file):
        os.remove(log_file)
        print(f"  Removed {log_file}")
    # 可能日志写入到其他路径，检查 logs 下最近的 e2etest 文件
    for f in os.listdir(log_dir) if os.path.isdir(log_dir) else []:
        if 'e2etest' in f:
            p = os.path.join(log_dir, f)
            if os.path.isfile(p):
                os.remove(p)
                print(f"  Removed {p}")

    print("=" * 50)
    print("E2E test PASSED. Pluggable module works correctly.")
    print("=" * 50)


if __name__ == '__main__':
    main()
