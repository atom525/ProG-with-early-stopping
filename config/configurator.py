"""
RecBole-style Configurator for ProG
"""
import os
from argparse import Namespace
try:
    import yaml
except ImportError:
    yaml = None
from typing import Any, Optional


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
    """config 覆盖 parser 默认值，保证 config 为统一配置来源"""
    merged = vars(args).copy()
    # 这些 key 始终以 config 为准（config 作为唯一配置源）
    merge_map = {
        'split_ratio': config.get_split_ratio(),
        'patience': config.get_patience(),
        'save_best': config.get('save_best', Config.DEFAULT_SAVE_BEST),
        'checkpoint_dir': config.get_checkpoint_dir(),
        'eval_every': config.get('evaluate_every', 1),
        'early_stopping_metric': config.get('early_stopping_metric', 'valid_acc'),
        'log_dir': config.get('log_dir', 'logs'),
        'log_file': config.get('log_file'),
    }
    for key, val in merge_map.items():
        if key == 'log_file' and val is None:
            merged[key] = getattr(args, key, None)
        else:
            merged[key] = val
    return Namespace(**merged)
