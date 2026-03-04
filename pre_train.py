import argparse
import os
from datetime import datetime
from prompt_graph.defines import GRAPH_TASKS, NODE_TASKS
from prompt_graph.pretrain import Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE, NodePrePrompt, GraphPrePrompt, DGI, GraphMAE
from prompt_graph.utils import seed_everything, mkdir, get_args
from prompt_graph.utils.tee_logger import setup_logging
from prompt_graph.utils.paths import get_log_path, get_pretrain_save_dir
from prompt_graph.utils.train_logger import train_info, log_args
from prompt_graph.data import load4node, load4graph


def get_pretrain_task_by_dataset_name(dataset_name:str)->str:
    if dataset_name in GRAPH_TASKS:
        return "GraphTask"
    elif dataset_name in NODE_TASKS:
        return "NodeTask"
    else:
        raise ValueError(f"Does not support this kind of dataset {dataset_name}.")
def _valid_ratio_from_split(split_ratio):
    """从 split_ratio [train, valid, test] 推导预训练验证集比例（与下游统一）"""
    if not split_ratio or len(split_ratio) < 2:
        return 0.1
    total = sum(split_ratio[:3]) if len(split_ratio) >= 3 else 1.0
    return split_ratio[1] / total if total > 0 else 0.1

def get_pretrain_task_delegate(args:argparse.Namespace):
    seed_everything(args.seed)
    split_ratio = getattr(args, 'split_ratio', [0.8, 0.1, 0.1])
    pretrain_val_ratio = _valid_ratio_from_split(split_ratio)
    if args.pretrain_task == 'SimGRACE':
        pt = SimGRACE(dataset_name=args.dataset_name, gnn_type=args.gnn_type, hid_dim=args.hid_dim, gln=args.num_layer, num_epoch=args.epochs, device=args.device, num_workers=args.num_workers, pretrain_val_ratio=pretrain_val_ratio)
    elif args.pretrain_task == 'GraphCL':
        pt = GraphCL(dataset_name=args.dataset_name, gnn_type=args.gnn_type, hid_dim=args.hid_dim, gln=args.num_layer, num_epoch=args.epochs, device=args.device, num_workers=args.num_workers, pretrain_val_ratio=pretrain_val_ratio)
    elif args.pretrain_task == 'Edgepred_GPPT':
        pt = Edgepred_GPPT(dataset_name=args.dataset_name, gnn_type=args.gnn_type, hid_dim=args.hid_dim, gln=args.num_layer, num_epoch=args.epochs, device=args.device, num_workers=args.num_workers, pretrain_val_ratio=pretrain_val_ratio)
    elif args.pretrain_task == 'Edgepred_Gprompt':
        pt = Edgepred_Gprompt(dataset_name=args.dataset_name, gnn_type=args.gnn_type, hid_dim=args.hid_dim, gln=args.num_layer, num_epoch=args.epochs, device=args.device, num_workers=args.num_workers, pretrain_val_ratio=pretrain_val_ratio)
    elif args.pretrain_task == 'DGI':
        pt = DGI(dataset_name=args.dataset_name, gnn_type=args.gnn_type, hid_dim=args.hid_dim, gln=args.num_layer, num_epoch=args.epochs, device=args.device, num_workers=args.num_workers, pretrain_val_ratio=pretrain_val_ratio)
    elif args.pretrain_task in ('NodeMultiGprompt','MultiGprompt','GraphMultiGprompt'):
        if args.pretrain_task == "NodeMultiGprompt" or args.dataset_name in NODE_TASKS:
            nonlinearity = 'prelu'
            pt = NodePrePrompt(args.dataset_name, args.hid_dim, nonlinearity, 0.9, 0.9, 0.1, 0.001, 1, 0.3, device=args.device)
        elif args.pretrain_task == 'GraphMultiGprompt'or args.dataset_name in GRAPH_TASKS:
            nonlinearity = 'prelu'
            input_dim, out_dim, graph_list = load4graph(args.dataset_name, pretrained=True)
            pt = GraphPrePrompt(graph_list, input_dim, out_dim, args.dataset_name, args.hid_dim, nonlinearity, 0.9, 0.9, 0.1, 1, 0.3, device=args.device, pretrain_val_ratio=pretrain_val_ratio)
        else:
            raise ValueError(f"Unsupported args.pretrain_task type for MultiGprompt {args.pretrain_task}")
    elif args.pretrain_task == 'GraphMAE':
        pt = GraphMAE(dataset_name=args.dataset_name, gnn_type=args.gnn_type, hid_dim=args.hid_dim, gln=args.num_layer, num_epoch=args.epochs, device=args.device,
                     mask_rate=0.75, drop_edge_rate=0.0, replace_rate=0.1, loss_fn='sce', alpha_l=2, num_workers=args.num_workers, pretrain_val_ratio=pretrain_val_ratio)
    else:
        raise ValueError(f"Unexpected args.pretrain_task type: {args.pretrain_task}.")
    return pt



if __name__ == "__main__":
    args = get_args()
    log_file = getattr(args, 'log_file', None)
    if not log_file:
        log_file = get_log_path(f"pretrain_{args.pretrain_task}", args.dataset_name)
    _logger = setup_logging(log_file)
    train_info('pretrain_task {} | dataset {} | log -> {}'.format(args.pretrain_task, args.dataset_name, log_file))
    log_args(args, "pretrain args")

    pt = get_pretrain_task_delegate(args=args)
    pt.pretrain(args=args)