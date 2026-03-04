import random
import torch
from prompt_graph.utils.labels import safe_graph_label


def graph_split(graph_list, shot_num, split_ratio=None):
    r"""A data object describing a homogeneous graph.
      The data object can hold node-level, link-level and graph-level attributes.
      In general, :class:`~torch_geometric.data.Data` tries to mimic the
      behavior of a regular :python:`Python` dictionary.
      In addition, it provides useful functionality for analyzing graph
      structures, and provides basic PyTorch tensor functionalities.
      See `here <https://pytorch-geometric.readthedocs.io/en/latest/get_started/
      introduction.html#data-handling-of-graphs>`__ for the accompanying
      tutorial.

      .. code-block:: python

          from torch_geometric.data import Data

          data = Data(x=x, edge_index=edge_index, ...)

          # Add additional arguments to `data`:
          data.train_idx = torch.tensor([...], dtype=torch.long)
          data.test_mask = torch.tensor([...], dtype=torch.bool)

          # Analyzing the graph structure:
          data.num_nodes
          >>> 23

          data.is_directed()
          >>> False

          # PyTorch tensor functionality:
          data = data.pin_memory()
          data = data.to('cuda:0', non_blocking=True)

      Args:
          x (torch.Tensor, optional): Node feature matrix with shape
              :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
          edge_index (LongTensor, optional): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
          edge_attr (torch.Tensor, optional): Edge feature matrix with shape
              :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
          y (torch.Tensor, optional): Graph-level or node-level ground-truth
              labels with arbitrary shape. (default: :obj:`None`)
          pos (torch.Tensor, optional): Node position matrix with shape
              :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
          time (torch.Tensor, optional): The timestamps for each event with shape
              :obj:`[num_edges]` or :obj:`[num_nodes]`. (default: :obj:`None`)
          **kwargs (optional): Additional attributes.
      """

    if split_ratio is None:
        split_ratio = [0.8, 0.1, 0.1]
    total = sum(split_ratio[:3])
    train_r, valid_r, test_r = [r / total for r in split_ratio[:3]]

    class_datasets = {}
    for data in graph_list:
        label = safe_graph_label(data.y)
        if label not in class_datasets:
            class_datasets[label] = []
        class_datasets[label].append(data)

    train_data = []
    remaining_data = []
    for label, data_list in class_datasets.items():
        train_data.extend(data_list[:shot_num])
        random.shuffle(train_data)
        remaining_data.extend(data_list[shot_num:])

    random.shuffle(remaining_data)
    n = len(remaining_data)
    n_valid = max(1, int(valid_r * n))
    n_test = min(n - n_valid, max(1, int(test_r * n)))
    val_dataset = remaining_data[:n_valid]
    test_dataset = remaining_data[n_valid:n_valid + n_test] if n_valid + n_test <= n else remaining_data[n_valid:]
    return train_data, test_dataset, val_dataset


def split_graph_dataset_full(graph_list, split_ratio=None, seed=42):
    """全量划分：按 split_ratio 将图列表划分为 train/valid/test，用于 shot_num=0 场景。"""
    if split_ratio is None:
        split_ratio = [0.8, 0.1, 0.1]
    total = sum(split_ratio[:3])
    train_r, valid_r, test_r = [r / total for r in split_ratio[:3]]
    n = len(graph_list)
    n_test = max(0, int(test_r * n))
    n_valid = max(0, int(valid_r * n))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    test_list = [graph_list[i] for i in perm[:n_test].tolist()]
    valid_list = [graph_list[i] for i in perm[n_test:n_test + n_valid].tolist()]
    train_list = [graph_list[i] for i in perm[n_test + n_valid:].tolist()]
    return train_list, valid_list, test_list
