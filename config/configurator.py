"""
RecBole-style Configurator for ProG
配置优先级：config.yaml 为默认，命令行参数可覆盖
"""
import os
import sys
from argparse import Namespace
try:
    import yaml
except ImportError:
    yaml = None
from typing import Any, Optional


def _cli_override(flag: str) -> bool:
    """是否在命令行显式传入了该参数"""
    return flag in sys.argv


class Config:
    DEFAULT_SPLIT_RATIO = [0.8, 0.1, 0.1]
    DEFAULT_PATIENCE = 20
    DEFAULT_SAVE_BEST = True

    def __init__(self, config_file: Optional[str] = None, **kwargs):
        self._config = {}
        if config_file and os.path.exists(config_file) and yaml:
            with open(config_file, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
        self._config.update(kwargs)

    def __getitem__(self, key: str) -> Any:
        return self._config.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def get_split_ratio(self):
        r = self._config.get('split_ratio', self.DEFAULT_SPLIT_RATIO)
        if isinstance(r, (list, tuple)) and len(r) >= 3:
            t = sum(r[:3])
            return [x / t for x in r[:3]]
        return list(self.DEFAULT_SPLIT_RATIO)

    def get_patience(self) -> int:
        return self._config.get('patience', self.DEFAULT_PATIENCE)

    def get_checkpoint_dir(self) -> str:
        return self._config.get('checkpoint_dir', 'checkpoints/downstream')

    def to_namespace(self) -> Namespace:
        return Namespace(**self._config)


def merge_config_with_args(config: Config, args: Namespace) -> Namespace:
    """config 为默认，命令行显式传入则覆盖。优先 config，再被 CLI 覆盖"""
    merged = vars(args).copy()
    # (args_key, config_val_getter, cli_flag)
    merge_items = [
        ('split_ratio', config.get_split_ratio, '--split_ratio'),
        ('seed', lambda: config.get('seed', 42), '--seed'),
        ('runseed', lambda: config.get('runseed', 0), '--runseed'),
        ('patience', config.get_patience, '--patience'),
        ('save_best', lambda: config.get('save_best', Config.DEFAULT_SAVE_BEST), '--save_best'),
        ('checkpoint_dir', config.get_checkpoint_dir, '--checkpoint_dir'),
        ('eval_every', lambda: config.get('evaluate_every', 1), '--eval_every'),
        ('early_stopping_metric', lambda: config.get('early_stopping_metric', 'valid_acc'), '--early_stopping_metric'),
        ('log_dir', lambda: config.get('log_dir', 'logs'), '--log_dir'),
        ('log_file', lambda: config.get('log_file'), '--log_file'),
        ('epochs', lambda: config.get('epochs', 1000), '--epochs'),
        ('batch_size', lambda: config.get('batch_size', 128), '--batch_size'),
        ('lr', lambda: config.get('lr', 0.001), '--lr'),
        ('decay', lambda: config.get('weight_decay', config.get('decay', 0)), '--decay'),
    ]
    for key, get_val, flag in merge_items:
        if _cli_override(flag):
            merged[key] = getattr(args, key, merged.get(key))
        else:
            val = get_val()
            if key == 'log_file' and val is None:
                merged[key] = getattr(args, key, None)
            else:
                merged[key] = val
    return Namespace(**merged)
