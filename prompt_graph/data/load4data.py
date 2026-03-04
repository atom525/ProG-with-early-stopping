import torch
import pickle as pk
from random import shuffle
import random
from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr, WebKB, Actor
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import Data,Batch
from torch_geometric.utils import negative_sampling
import os
# OGB 延迟导入，避免 MUTAG 等 TUDataset 运行时触发 sklearn/scipy 的 libstdc++ 兼容问题

from ..defines import GRAPH_TASKS, LINK_TASKS
from ..utils.paths import get_data_root
from ..utils.train_logger import train_info

def node_sample_and_save(data, k, folder, num_classes, split_ratio=[0.8, 0.1, 0.1]):
    """按自定义比例划分训练/验证/测试集，RecBole-style RS (Ratio-based Splitting)"""
    labels = data.y.to('cpu')
    if labels.dim() > 1:
        labels = labels.squeeze()
    num_nodes = data.num_nodes
    
    # 归一化比例
    total = sum(split_ratio[:3])
    train_r, valid_r, test_r = [r / total for r in split_ratio[:3]]
    
    perm = torch.randperm(num_nodes)
    
    # 按比例划分
    num_test = int(test_r * num_nodes)
    num_valid = int(valid_r * num_nodes)
    num_train = num_nodes - num_test - num_valid
    
    test_idx = perm[:num_test]
    valid_idx = perm[num_test:num_test + num_valid]
    train_pool_idx = perm[num_test + num_valid:]
    
    # Few-shot: 从训练池中每类选k个
    remaining_labels = labels[train_pool_idx]
    train_idx = torch.cat([train_pool_idx[remaining_labels == i][:k] for i in range(num_classes)])
    train_idx = train_idx[torch.randperm(train_idx.size(0))]
    train_labels = labels[train_idx]
    valid_labels = labels[valid_idx]
    test_labels = labels[test_idx]

    torch.save(train_idx, os.path.join(folder, 'train_idx.pt'))
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
    torch.save(valid_idx, os.path.join(folder, 'valid_idx.pt'))
    torch.save(valid_labels, os.path.join(folder, 'valid_labels.pt'))
    torch.save(test_idx, os.path.join(folder, 'test_idx.pt'))
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))

def graph_sample_and_save(dataset, k, folder, num_classes, split_ratio=[0.8, 0.1, 0.1]):
    """按自定义比例划分图级训练/验证/测试集，RecBole-style RS"""
    num_graphs = len(dataset)
    total = sum(split_ratio[:3])
    train_r, valid_r, test_r = [r / total for r in split_ratio[:3]]
    
    num_test = int(test_r * num_graphs)
    num_valid = int(valid_r * num_graphs)
    num_train_pool = num_graphs - num_test - num_valid

    labels = torch.tensor([g.y.item() if g.y.dim() == 0 else g.y.squeeze().item() for g in dataset])
    all_indices = torch.randperm(num_graphs)
    
    test_indices = all_indices[:num_test]
    valid_indices = all_indices[num_test:num_test + num_valid]
    remaining_indices = all_indices[num_test + num_valid:]

    torch.save(test_indices, os.path.join(folder, 'test_idx.pt'))
    torch.save(labels[test_indices], os.path.join(folder, 'test_labels.pt'))
    torch.save(valid_indices, os.path.join(folder, 'valid_idx.pt'))
    torch.save(labels[valid_indices], os.path.join(folder, 'valid_labels.pt'))

    train_indices = []
    for i in range(num_classes):
        class_indices = [idx for idx in remaining_indices.tolist() if labels[idx].item() == i]
        selected_indices = class_indices[:k]
        train_indices.extend(selected_indices)

    train_indices = torch.tensor(train_indices)
    train_indices = train_indices[torch.randperm(train_indices.size(0))]
    torch.save(train_indices, os.path.join(folder, 'train_idx.pt'))
    torch.save(labels[train_indices], os.path.join(folder, 'train_labels.pt'))

def node_degree_as_features(data_list):
    from torch_geometric.utils import degree
    for data in data_list:
        # 计算所有节点的度数，这将返回一个张量
        deg = degree(data.edge_index[0], dtype=torch.long)

        # 将度数张量变形为[nodes, 1]以便与其他特征拼接
        deg = deg.view(-1, 1).float()
        
        # 如果原始数据没有节点特征，可以直接使用度数作为特征
        if data.x is None:
            data.x = deg
        else:
            # 将度数特征拼接到现有的节点特征上
            data.x = torch.cat([data.x, deg], dim=1)

def load4graph(dataset_name, shot_num= 10, num_parts=None, pretrained=False):
    r"""A plain old python object modeling a batch of graphs as one big
        (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
        base class, all its methods can also be used here.
        In addition, single graphs can be reconstructed via the assignment vector
        :obj:`batch`, which maps each node to its respective graph identifier.
        """

    if dataset_name in GRAPH_TASKS:
        dataset = TUDataset(root=os.path.join(get_data_root(), 'TUDataset'), name=dataset_name, use_node_attr=True)  # use_node_attr=False时，节点属性为one-hot编码的节点类别
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

        if dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY']:
            graph_list = [g for g in graph_list]
            node_degree_as_features(graph_list)
            input_dim = graph_list[0].x.size(1)        

        # # 分类并选择每个类别的图
        # class_datasets = {}
        # for data in dataset:
        #     label = data.y.item()
        #     if label not in class_datasets:
        #         class_datasets[label] = []
        #     class_datasets[label].append(data)

        # train_data = []
        # remaining_data = []
        # for label, data_list in class_datasets.items():
        #     train_data.extend(data_list[:shot_num])
        #     random.shuffle(train_data)
        #     remaining_data.extend(data_list[shot_num:])

        # # 将剩余的数据 1：9 划分为测试集和验证集
        # random.shuffle(remaining_data)
        # val_dataset_size = len(remaining_data) // 9
        # val_dataset = remaining_data[:val_dataset_size]
        # test_dataset = remaining_data[val_dataset_size:]
        

        if pretrained:
            return input_dim, out_dim, graph_list
        if shot_num == 0:
            from .graph_split import split_graph_dataset_full
            train_list, valid_list, test_list = split_graph_dataset_full(graph_list)
            return input_dim, out_dim, (train_list, valid_list, test_list)
        return input_dim, out_dim, dataset

    elif dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name = dataset_name, root=os.path.join(get_data_root(), 'ogbg'))
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

        graph_list = [g for g in graph_list]
        node_degree_as_features(graph_list)
        input_dim = graph_list[0].x.size(1)

        for g in graph_list:
            g.y = g.y.squeeze(0)

        if pretrained:
            return input_dim, out_dim, graph_list
        if shot_num == 0:
            from .graph_split import split_graph_dataset_full
            train_list, valid_list, test_list = split_graph_dataset_full(graph_list)
            return input_dim, out_dim, (train_list, valid_list, test_list)
        return input_dim, out_dim, dataset
    else:
        raise ValueError(f"Unsupported GraphTask on dataset: {dataset_name}.")
    
def load4node(dataname):
    train_info("loading node dataset: {}".format(dataname))
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root=os.path.join(get_data_root(), 'Planetoid'), name=dataname, transform=NormalizeFeatures())
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root=os.path.join(get_data_root(), 'amazon'), name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Reddit':
        dataset = Reddit(root=os.path.join(get_data_root(), 'Reddit'))
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'WikiCS':
        dataset = WikiCS(root=os.path.join(get_data_root(), 'WikiCS'))
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Flickr':
        dataset = Flickr(root=os.path.join(get_data_root(), 'Flickr'))
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname in ['Wisconsin', 'Texas']:
        dataset = WebKB(root=os.path.join(get_data_root(), dataname), name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Actor':
        dataset = Actor(root=os.path.join(get_data_root(), 'Actor'))
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'ogbn-arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=get_data_root())
        data = dataset[0]
        input_dim = data.x.shape[1]
        out_dim = dataset.num_classes
    elif dataname in ['ENZYMES', 'PROTEINS']:
        # 实现TUDataset中两个multi graphs dataset的节点分类
        dataset = TUDataset(root=os.path.join(get_data_root(), 'TUDataset'), name=dataname, use_node_attr=True)
        node_class = dataset.data.x[:,-3:]
        input_dim = dataset.num_node_features
        out_dim = dataset.num_node_labels
        data = Batch.from_data_list(dataset)  # 将dataset中小图合并成一个大图
        data.y = node_class.nonzero().T[1]
    else:
        raise ValueError(f"Unsupported NodeTask on dataset: {dataname}.")
    return data, input_dim, out_dim


def load4link_downstream(dataset_name, split_ratio=None):
    """下游链接预测：加载节点图数据，用 RandomLinkSplit 按 split_ratio 划分 train/val/test"""
    if split_ratio is None:
        split_ratio = [0.8, 0.1, 0.1]
    total = sum(split_ratio[:3])
    valid_r = split_ratio[1] / total if total > 0 else 0.1
    test_r = split_ratio[2] / total if total > 0 else 0.1
    if dataset_name not in LINK_TASKS:
        raise ValueError(f"Unsupported LinkTask on dataset: {dataset_name}. Supported: {LINK_TASKS}")
    transform = NormalizeFeatures()
    dataset = Planetoid(root=os.path.join(get_data_root(), 'Planetoid'), name=dataset_name, transform=transform)
    data = dataset[0]
    link_transform = RandomLinkSplit(num_val=valid_r, num_test=test_r, is_undirected=True, add_negative_train_samples=False)
    train_data, val_data, test_data = link_transform(data)
    input_dim = dataset.num_features
    output_dim = 2
    return train_data, val_data, test_data, input_dim, output_dim


def load4link_prediction_single_graph(dataname, num_per_samples=1):
    data, input_dim, output_dim = load4node(dataname)

    
    r"""Perform negative sampling to generate negative neighbor samples"""
    if data.is_directed():
        row, col = data.edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
    else:
        edge_index = data.edge_index
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges * num_per_samples,
    )

    edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))], dim=0)

    return data, edge_label, edge_index, input_dim, output_dim

def load4link_prediction_multi_graph(dataset_name, num_per_samples=1):
    if dataset_name in GRAPH_TASKS:
        dataset = TUDataset(root=os.path.join(get_data_root(), 'TUDataset'), name=dataset_name, use_node_attr=True)

    if dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name = dataset_name, root=os.path.join(get_data_root(), 'ogbg'))
    
    input_dim = dataset.num_features
    output_dim = 2 # link prediction的输出维度应该是2，0代表无边，1代表右边

    if dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY']:
        dataset = [g for g in dataset]
        node_degree_as_features(dataset)
        input_dim = dataset[0].x.size(1)

    elif dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = [g for g in dataset]
        node_degree_as_features(dataset)
        input_dim = dataset[0].x.size(1)
        for g in dataset:
            g.y = g.y.squeeze(1)

    data = Batch.from_data_list(dataset)
    
    r"""Perform negative sampling to generate negative neighbor samples"""
    if data.is_directed():
        row, col = data.edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
    else:
        edge_index = data.edge_index
        
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges * num_per_samples,
    )

    edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))], dim=0)
    
    return data, edge_label, edge_index, input_dim, output_dim

# 未完待续，需要重写一个能够对large-scale图分类数据集的划分代码，避免node-level和edge-level的预训练算法或prompt方法显存溢出的问题
def load4link_prediction_multi_large_scale_graph(dataset_name, num_per_samples=1):
    if dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name = dataset_name, root=os.path.join(get_data_root(), 'ogbg'))
    
    input_dim = dataset.num_features
    output_dim = 2 # link prediction的输出维度应该是2，0代表无边，1代表右边

    dataset = [g for g in dataset]
    node_degree_as_features(dataset)
    input_dim = dataset[0].x.size(1)
    for g in dataset:
        g.y = g.y.squeeze(1)

    batch_graph_num = 20000
    split_num = int(len(dataset)/batch_graph_num)
    data_list = []
    edge_label_list = []
    edge_index_list = []
    for i in range(split_num+1):
        if(i==0):
            data = Batch.from_data_list(dataset[0:batch_graph_num])
        elif(i<=split_num):
            data = Batch.from_data_list(dataset[i*batch_graph_num:(i+1)*batch_graph_num])
        elif len(dataset)>((i-1)*batch_graph_num):
            data = Batch.from_data_list(dataset[i*batch_graph_num:(i+1)*batch_graph_num])
        

        data_list.append(data)
        
        r"""Perform negative sampling to generate negative neighbor samples"""
        if data.is_directed():
            row, col = data.edge_index
            row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
            edge_index = torch.stack([row, col], dim=0)
        else:
            edge_index = data.edge_index
            
        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.num_edges * num_per_samples,
        )

        edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
        edge_label = torch.cat([torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))], dim=0)
    
    return data, edge_label, edge_index, input_dim, output_dim

# used in pre_train.py
def NodePretrain(data, num_parts=200, split_method='Random Walk'):

    # if(dataname=='Cora'):
    #     num_parts=220
    # elif(dataname=='Texas'):
    #     num_parts=20
    if(split_method=='Cluster'):
        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)
        
        graph_list = list(ClusterData(data=data, num_parts=num_parts))
    elif(split_method=='Random Walk'):
        from torch_cluster import random_walk
        split_ratio = 0.1
        walk_length = 30
        all_random_node_list = torch.randperm(data.num_nodes)
        selected_node_num_for_random_walk = int(split_ratio * data.num_nodes)
        random_node_list = all_random_node_list[:selected_node_num_for_random_walk]
        walk_list = random_walk(data.edge_index[0], data.edge_index[1], random_node_list, walk_length=walk_length)

        graph_list = [] 
        skip_num = 0        
        for walk in walk_list:   
            subgraph_nodes = torch.unique(walk)
            if(len(subgraph_nodes)<5):
                skip_num+=1
                continue
            subgraph_data = data.subgraph(subgraph_nodes)

            graph_list.append(subgraph_data)

        train_info("Total {} random walk subgraphs (nodes>5), {} skipped".format(len(graph_list), skip_num))

    else:
        raise ValueError("None split method!")
    
    return graph_list


