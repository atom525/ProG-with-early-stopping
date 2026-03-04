import argparse
import os
import pickle
import sys
from datetime import datetime
from prompt_graph.tasker import NodeTask, GraphTask, LinkTask
from prompt_graph.registry import PromptRegistry
from prompt_graph.utils import seed_everything, get_args
from prompt_graph.utils.tee_logger import setup_logging
from prompt_graph.utils.paths import get_induced_graph_path, get_downstream_checkpoint_dir, get_log_path
from prompt_graph.utils.train_logger import train_info, log_args
from prompt_graph.data import load4node, load4graph, split_induced_graphs


def load_induced_graph(dataset_name, data, device):
    file_path = get_induced_graph_path(dataset_name, min_size=100, max_size=300)
    if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                train_info("loading induced graph...")
                graphs_list = pickle.load(f)
                train_info("induced graph loaded")
    else:
        train_info("Begin split_induced_graphs")
        split_induced_graphs(data, os.path.dirname(file_path), device, smallest_size=100, largest_size=300)
        with open(file_path, 'rb') as f:
            graphs_list = pickle.load(f)
    graphs_list = [graph.to(device) for graph in graphs_list]
    return graphs_list




def get_downstream_task_delegate(args:argparse.Namespace):
    
    seed_everything(args.seed)
    
    if args.downstream_task == 'NodeTask':
        data, input_dim, output_dim = load4node(args.dataset_name)   
        data = data.to(args.device)
        if PromptRegistry.needs_induced_graph(args.prompt_type):
            graphs_list = load_induced_graph(args.dataset_name, data, args.device) 
        else:
            graphs_list = None 
        run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        ckpt_dir = get_downstream_checkpoint_dir(args.dataset_name, run_ts)
        tasker = NodeTask(pre_train_model_path = args.pre_train_model_path, 
                        dataset_name = args.dataset_name, num_layer = args.num_layer,
                        gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type,
                        epochs = args.epochs, shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                        batch_size = args.batch_size, data = data, input_dim = input_dim, output_dim = output_dim, graphs_list = graphs_list,
                        split_ratio=getattr(args, 'split_ratio', [0.8, 0.1, 0.1]), patience=getattr(args, 'patience', 20),
                        save_best=getattr(args, 'save_best', True), checkpoint_dir=ckpt_dir,
                        eval_every=getattr(args, 'eval_every', 1), early_stopping_metric=getattr(args, 'early_stopping_metric', 'valid_acc'),
                        log_dir=getattr(args, 'log_dir', 'logs'), log_file=getattr(args, 'log_file', None))


    elif args.downstream_task == 'GraphTask':
        split_ratio = getattr(args, 'split_ratio', [0.8, 0.1, 0.1])
        seed = getattr(args, 'seed', 42)
        input_dim, output_dim, dataset = load4graph(
            args.dataset_name, getattr(args, 'shot_num', 10),
            split_ratio=split_ratio, seed=seed
        )
        run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        ckpt_dir = get_downstream_checkpoint_dir(args.dataset_name, run_ts)
        tasker = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                        dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type, epochs = args.epochs,
                        shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                        batch_size = args.batch_size, dataset = dataset, input_dim = input_dim, output_dim = output_dim,
                        split_ratio=getattr(args, 'split_ratio', [0.8, 0.1, 0.1]), patience=getattr(args, 'patience', 20),
                        save_best=getattr(args, 'save_best', True), checkpoint_dir=ckpt_dir,
                        eval_every=getattr(args, 'eval_every', 1), early_stopping_metric=getattr(args, 'early_stopping_metric', 'valid_acc'),
                        log_dir=getattr(args, 'log_dir', 'logs'), log_file=getattr(args, 'log_file', None))

    elif args.downstream_task == 'LinkTask':
        run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        ckpt_dir = get_downstream_checkpoint_dir(args.dataset_name, run_ts)
        # LinkTask 默认 valid_auroc；用户显式传 --early_stopping_metric 时用其值
        link_metric = args.early_stopping_metric if '--early_stopping_metric' in sys.argv else 'valid_auroc'
        tasker = LinkTask(
            dataset_name=args.dataset_name, gnn_type=args.gnn_type, hid_dim=args.hid_dim, num_layer=args.num_layer,
            epochs=args.epochs, device=args.device, lr=args.lr, wd=args.decay,
            split_ratio=getattr(args, 'split_ratio', [0.8, 0.1, 0.1]), patience=getattr(args, 'patience', 20),
            save_best=getattr(args, 'save_best', True), checkpoint_dir=ckpt_dir,
            eval_every=getattr(args, 'eval_every', 1), early_stopping_metric=link_metric,
            log_dir=getattr(args, 'log_dir', 'logs'), log_file=getattr(args, 'log_file', None),
            pre_train_model_path=args.pre_train_model_path,
        )
    else:
        raise ValueError(f"Unexpected args.downstream_task type {args.downstream_task}.")

    return tasker

if __name__ == "__main__":
    args = get_args()
    log_file = getattr(args, 'log_file', None)
    if not log_file:
        log_file = get_log_path(f"downstream_{args.downstream_task}", args.dataset_name)
    args.log_file = log_file
    _logger = setup_logging(log_file)
    train_info('dataset_name {} | log -> {}'.format(args.dataset_name, log_file))
    log_args(args, "downstream args")

    tasker = get_downstream_task_delegate(args=args)

    _, test_acc, std_test_acc, f1, std_f1, roc, std_roc, prc, std_prc = tasker.run()
    
    train_info("Final Accuracy {:.4f}±{:.4f}(std)".format(test_acc, std_test_acc))
    train_info("Final F1 {:.4f}±{:.4f}(std)".format(f1, std_f1))
    train_info("Final AUROC {:.4f}±{:.4f}(std)".format(roc, std_roc))
    train_info("Final AUPRC {:.4f}±{:.4f}(std)".format(prc, std_prc)) 

    pre_train_type = tasker.pre_train_type



