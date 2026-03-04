"""
统一路径配置：数据、权重、日志分别存放在固定目录，带时间戳

路径来源：config/config.yaml（优先级） > 默认值
  - data_root       → data/                      # 原始数据集，load4data & download_data 共用
  - pretrain_save   → checkpoints/pretrain/     # 预训练 {dataset}/{Model}_{ts}.pth
  - checkpoint_dir → checkpoints/downstream/   # 下游 {dataset}_{ts}/*_best.pt
  - log_dir         → logs/                     # 日志 {prefix}{task}_{dataset}_{ts}.log
  - sample_data    → outputs/sample_data/       # few-shot Node/Graph/{dataset}/
  - induced_graph   → outputs/induced_graph/    # 诱导子图 {dataset}/induced_graph_*.pkl
"""
import os
from datetime import datetime


def _load_config():
    cfg = {}
    root = os.path.abspath(os.getcwd())
    p = os.path.join(root, 'config', 'config.yaml')
    if os.path.exists(p):
        try:
            import yaml
            with open(p, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            pass
    return cfg, root


_CFG, _ROOT = _load_config()


def _join(*parts):
    return os.path.normpath(os.path.join(_ROOT, *parts))


def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def get_data_root():
    return _join(_CFG.get('data_root', 'data'))


def get_log_dir():
    return _join(_CFG.get('log_dir', 'logs'))


def get_pretrain_save_dir():
    return _join(_CFG.get('pretrain_save_dir', 'checkpoints/pretrain'))


def get_checkpoint_dir():
    return _join(_CFG.get('checkpoint_dir', 'checkpoints/downstream'))


def get_sample_data_dir():
    return _join(_CFG.get('sample_data_dir', 'outputs/sample_data'))


def get_induced_graph_dir():
    return _join(_CFG.get('induced_graph_dir', 'outputs/induced_graph'))


def get_induced_graph_path(dataset_name, min_size=None, max_size=300):
    """induced graph 文件路径：outputs/induced_graph/{dataset}/induced_graph_min{min}_max{max}.pkl"""
    d = os.path.join(get_induced_graph_dir(), dataset_name)
    os.makedirs(d, exist_ok=True)
    min_s = min_size if min_size is not None else 100
    return os.path.join(d, f'induced_graph_min{min_s}_max{max_size}.pkl')


def get_sample_data_node_dir(dataset_name):
    d = os.path.join(get_sample_data_dir(), 'Node', dataset_name)
    os.makedirs(d, exist_ok=True)
    return d


def get_sample_data_graph_dir(dataset_name):
    d = os.path.join(get_sample_data_dir(), 'Graph', dataset_name)
    os.makedirs(d, exist_ok=True)
    return d


def get_pretrain_save_path(dataset_name, model_name='MultiGprompt', suffix='', ext='.pth'):
    """预训练权重路径，带时间戳。suffix 如 '.GCN.128hidden_dim' 用于 Edgepred 等"""
    d = os.path.join(get_pretrain_save_dir(), dataset_name)
    os.makedirs(d, exist_ok=True)
    name = f"{model_name}{suffix}_{get_timestamp()}{ext}"
    return os.path.join(d, name)


def get_downstream_checkpoint_dir(dataset_name, run_ts=None):
    """下游 checkpoint 目录，带时间戳"""
    ts = run_ts or get_timestamp()
    d = os.path.join(get_checkpoint_dir(), f"{dataset_name}_{ts}")
    os.makedirs(d, exist_ok=True)
    return d


def get_log_path(task_type, dataset_name, prefix=''):
    """日志路径，带时间戳"""
    d = get_log_dir()
    os.makedirs(d, exist_ok=True)
    name = f"{prefix}{task_type}_{dataset_name}_{get_timestamp()}.log"
    return os.path.join(d, name)
