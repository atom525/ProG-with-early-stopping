import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of pre-training of graph neural networks"
    )
    parser.add_argument("--pretrain_task", type=str)
    parser.add_argument("--downstream_task", "--task", type=str, dest="downstream_task", help="NodeTask, GraphTask, or LinkTask")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Cora",
        help="Choose the dataset of pretrainor downstream task",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--gnn_type",
        type=str,
        default="GCN",
        help="We support gnn like GCN GAT GT GCov GIN GraphSAGE, please read ProG.model module",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="None",
        help="Choose the prompt type for node or graph task, for node task,we support GPPT, All-in-one, Gprompt for graph task , All-in-one, Gprompt, GPF, GPF-plus ",
    )
    parser.add_argument(
        "--hid_dim",
        type=int,
        default=128,
        help="hideen layer of GNN dimensions (default: 128)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs to train (default: 50)",
    )
    parser.add_argument("--shot_num", type=int, default=1, help="Number of shots")
    parser.add_argument(
        "--pre_train_model_path",
        type=str,
        default="None",
        help="add pre_train_model_path to the downstream task, the model is self-supervise model if the path is None and prompttype is None.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--decay", type=float, default=0, help="Weight decay (default: 0)"
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        default=2,
        help="Number of GNN message passing layers (default: 2).",
    )

    parser.add_argument(
        "--dropout_ratio", type=float, default=0.5, help="Dropout ratio (default: 0.5)"
    )
    parser.add_argument(
        "--graph_pooling",
        type=str,
        default="mean",
        help="Graph level pooling (sum, mean, max, set2set, attention)",
    )
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        help="How the node features across layers are combined. last, sum, max or concat",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for splitting dataset."
    )
    parser.add_argument(
        "--runseed", type=int, default=0, help="Seed for running experiments."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for dataset loading",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="A range of [1,2,3]-layer MLPs with equal width",
    )
    parser.add_argument(
        "--pnum",
        type=int,
        default=5,
        help="The number of independent basis for GPF-plus",
    )
    parser.add_argument(
        "--task_num",
        type=int,
        default=5,
        help="The number of tasks for computing the mean metrices",
    )
    # RecBole-style: 自定义数据划分比例 [train, valid, test]
    parser.add_argument(
        "--split_ratio",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=('TRAIN', 'VALID', 'TEST'),
        help="Split ratio for train/valid/test (default: 0.8 0.1 0.1)",
    )
    # 早停策略
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)",
    )
    # 模型保存
    parser.add_argument(
        "--save_best",
        action="store_true",
        default=True,
        help="Save best model checkpoint based on validation",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/downstream",
        help="Directory to save best model checkpoints",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to YAML config file (RecBole-style), e.g. config/config.yaml",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=None,
        help="Every N epochs evaluate on validation set and print (default: 1)",
    )
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        default=None,
        choices=["valid_acc", "valid_f1", "valid_auroc", "valid_auprc"],
        help="Metric for early stopping (no loss). valid_acc/valid_f1/valid_auroc/valid_auprc",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for log files (default: logs)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Log file path. If None, auto-generate from task_dataset_timestamp.log",
    )
    args = parser.parse_args()
    # RecBole-style: 优先加载 config，未指定时尝试多种路径解析
    def _resolve_config_path():
        if getattr(args, 'config_file', None):
            return args.config_file
        cwd_cfg = os.path.join(os.getcwd(), 'config', 'config.yaml')
        if os.path.exists(cwd_cfg):
            return cwd_cfg
        _dir = os.path.dirname(os.path.abspath(__file__))
        proj_root = os.path.dirname(os.path.dirname(_dir))
        proj_cfg = os.path.join(proj_root, 'config', 'config.yaml')
        if os.path.exists(proj_cfg):
            return proj_cfg
        return 'config/config.yaml'
    config_path = _resolve_config_path()
    if config_path and os.path.exists(config_path):
        try:
            from config import Config
            from config.configurator import merge_config_with_args
            config = Config(config_file=config_path)
            args = merge_config_with_args(config, args)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to merge config from {config_path}: {e}")
    return args


def get_args_by_call(
    pretrain_task: str = None,
    downstream_task: str = None,
    dataset_name: str = "Cora",
    device: int = 0,
    gnn_type: str = "GCN",
    prompt_type: str = "None",
    hid_dim: int = 128,
    batch_size: int = 128,
    epochs: int = 1000,
    shot_num: int = 1,
    pre_train_model_path: str = "None",
    lr: float = 0.001,
    decay: float = 0,
    num_layer: int = 2,
    dropout_ratio: float = 0.5,
    graph_pooling: str = "mean",
    JK: str = "last",
    seed: int = 42,
    runseed: int = 0,
    num_workers: int = 0,
    num_layers: int = 1,
    pnum: int = 5,
    task_num: int = 5,
    split_ratio: list = None,
    patience: int = 20,
    save_best: bool = True,
    checkpoint_dir: str = "checkpoints/downstream",
    **kwargs
) -> argparse.Namespace:
    
    if len(kwargs)> 0:
        import warnings
        warnings.warn("Unexpected argument input: {}".format(list(kwargs.keys())))
    return argparse.Namespace(
        pretrain_task=pretrain_task,
        downstream_task=downstream_task,
        dataset_name=dataset_name,
        device=device,
        gnn_type=gnn_type,
        prompt_type=prompt_type,
        hid_dim=hid_dim,
        batch_size=batch_size,
        epochs=epochs,
        shot_num=shot_num,
        pre_train_model_path=pre_train_model_path,
        lr=lr,
        decay=decay,
        num_layer=num_layer,
        dropout_ratio=dropout_ratio,
        graph_pooling=graph_pooling,
        JK=JK,
        seed=seed,
        runseed=runseed,
        num_workers=num_workers,
        num_layers=num_layers,
        pnum=pnum,
        task_num=task_num,
        split_ratio=split_ratio if split_ratio is not None else [0.8, 0.1, 0.1],
        patience=patience,
        save_best=save_best,
        checkpoint_dir=checkpoint_dir,
        **kwargs
    )



DEFAULT_ARG_DICT = dict(vars(get_args_by_call()))
DEFAULT_ARG_KEYS = list(DEFAULT_ARG_DICT.keys())