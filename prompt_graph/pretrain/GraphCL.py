import sys
import torch
from tqdm import tqdm
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from random import shuffle
import random
from ..defines import NODE_TASKS
from prompt_graph.utils import mkdir, graph_views
from prompt_graph.data import load4node, load4graph, NodePretrain
from torch.optim import Adam
import os
from .base import PreTrain
from prompt_graph.utils.train_logger import train_info, epoch_training, epoch_evaluating, valid_result, finished_training, early_stopping_msg, model_saved, metric_from_dict, to_ordered_metrics, best_valid_ordered
import time

class GraphCL(PreTrain):
    def __init__(self, *args, **kwargs):
        self._val_ratio = kwargs.pop('pretrain_val_ratio', 0.1) or 0.1
        super().__init__(*args, **kwargs)
        self.load_graph_data()
        self.initialize_gnn(self.input_dim, self.hid_dim)
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.hid_dim),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(self.hid_dim, self.hid_dim)).to(self.device)

    def load_graph_data(self):
        if self.dataset_name in NODE_TASKS:
            data, self.input_dim, _ = load4node(self.dataset_name)
            self.graph_list = NodePretrain(data=data, num_parts=200, split_method='Cluster')
        else:
            self.input_dim, self.out_dim, self.graph_list = load4graph(self.dataset_name, pretrained=True)
        import numpy as np
        n = len(self.graph_list)
        perm = np.random.RandomState(42).permutation(n)
        n_val = max(1, int(n * self._val_ratio))
        self.val_graph_list = [self.graph_list[int(i)] for i in perm[:n_val]]
        self.graph_list = [self.graph_list[int(i)] for i in perm[n_val:]]
    
    def get_loader(self, graph_list, batch_size,aug1=None, aug2=None, aug_ratio=None):

        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in GraphCL!")
        
        shuffle(graph_list)
        if aug1 is None:
            aug1 = random.sample(['dropN', 'permE', 'maskN'], k=1)
        if aug2 is None:
            aug2 = random.sample(['dropN', 'permE', 'maskN'], k=1)
        if aug_ratio is None:
            aug_ratio = random.randint(1, 3) * 1.0 / 10  # 0.1,0.2,0.3

        train_info("graph views: {} and {} aug_ratio: {}".format(aug1, aug2, aug_ratio))

        view_list_1 = []
        view_list_2 = []
        for g in graph_list:
            view_g = graph_views(data=g, aug=aug1, aug_ratio=aug_ratio)
            view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
            view_list_1.append(view_g)
            view_g = graph_views(data=g, aug=aug2, aug_ratio=aug_ratio)
            view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
            view_list_2.append(view_g)

        loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False,
                                num_workers=self.num_workers)  # you must set shuffle=False !
        loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                                num_workers=self.num_workers)  # you must set shuffle=False !

        return loader1, loader2
    
    def forward_cl(self, x, edge_index, batch):
        x = self.gnn(x, edge_index, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = - torch.log(pos_sim / (sim_matrix.sum(dim=1) + 1e-4)).mean()
        # loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
        # loss = - torch.log(loss).mean() 
        return loss

    def evaluate_valid_acc(self):
        """验证集：同一图两个视图的嵌入相似度高于阈值的比例"""
        self.eval()
        device = self.device
        correct, total = 0, 0
        aug1, aug2 = 'dropN', 'permE'
        aug_ratio = 0.1
        with torch.no_grad():
            for g in self.val_graph_list:
                v1 = graph_views(data=g, aug=aug1, aug_ratio=aug_ratio)
                v2 = graph_views(data=g, aug=aug2, aug_ratio=aug_ratio)
                v1 = Data(x=v1.x, edge_index=v1.edge_index).to(device)
                v2 = Data(x=v2.x, edge_index=v2.edge_index).to(device)
                batch1 = torch.zeros(v1.num_nodes, dtype=torch.long, device=device)
                batch2 = torch.zeros(v2.num_nodes, dtype=torch.long, device=device)
                x1 = self.forward_cl(v1.x, v1.edge_index, batch1).mean(dim=0, keepdim=True)
                x2 = self.forward_cl(v2.x, v2.edge_index, batch2).mean(dim=0, keepdim=True)
                sim = (x1 * x2).sum() / (x1.norm() * x2.norm() + 1e-8)
                if sim.item() > 0.5:
                    correct += 1
                total += 1
        self.train()
        return correct / total if total > 0 else 0.0

    def train_graphcl(self, loader1, loader2, optimizer):
        self.train()
        train_loss_accum = 0
        total_step = 0
        pbar = tqdm(zip(loader1, loader2), desc="Batch", leave=False, file=sys.stderr)
        for step, batch in enumerate(pbar):
            batch1, batch2 = batch
            optimizer.zero_grad()
            x1 = self.forward_cl(batch1.x.to(self.device), batch1.edge_index.to(self.device), batch1.batch.to(self.device))
            x2 = self.forward_cl(batch2.x.to(self.device), batch2.edge_index.to(self.device), batch2.batch.to(self.device))
            loss = self.loss_cl(x1, x2)

            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1
            pbar.set_postfix(loss=f"{train_loss_accum/total_step:.4f}")
        return train_loss_accum / total_step

    def pretrain(self, batch_size=10, aug1='dropN', aug2="permE", aug_ratio=None, lr=0.01, decay=0.0001, args=None):
        num_epoch = getattr(args, 'epochs', self.epochs) if args else self.epochs
        patience = getattr(args, 'patience', 20) if args else 20
        eval_every = (getattr(args, 'eval_every', 1) or 1) if args else 1
        self.to(self.device)
        if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa', 'DD']:
            batch_size = 512
        loader1, loader2 = self.get_loader(self.graph_list, batch_size, aug1=aug1, aug2=aug2)
        train_info('start training {} | {} | {}'.format(self.dataset_name, 'GraphCL', self.gnn_type))
        optimizer = Adam(self.parameters(), lr=lr, weight_decay=decay)
        early_stopping_metric = getattr(args, 'early_stopping_metric', 'valid_acc') or 'valid_acc'
        met_name = early_stopping_metric
        best_val_metric = -1.0
        best_epoch = 0
        cnt_wait = 0
        best_state = None
        best_valid_metrics = {}
        for epoch in range(1, num_epoch + 1):
            st_time = time.time()
            train_loss = self.train_graphcl(loader1, loader2, optimizer)
            train_time = time.time() - st_time
            epoch_training(epoch, train_time, train_loss)
            if epoch % eval_every == 0:
                eval_st = time.time()
                valid_acc = self.evaluate_valid_acc()
                eval_time = time.time() - eval_st
                valid_metrics = {"valid_acc": valid_acc}
                val_metric = metric_from_dict(valid_metrics, met_name)
                epoch_evaluating(epoch, eval_time, val_metric, met_name)
                valid_result(valid_metrics)
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_valid_metrics = valid_metrics
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in self.gnn.state_dict().items()}
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                    if cnt_wait >= patience:
                        early_stopping_msg(epoch, patience, met_name)
                        break
        if best_state is not None:
            self.gnn.load_state_dict(best_state)
            self.gnn.to(self.device)
        finished_training(best_epoch if best_epoch > 0 else 1)
        if best_valid_metrics:
            best_valid_ordered(best_valid_metrics)
        from prompt_graph.utils.paths import get_pretrain_save_path
        suffix = ".{}.{}hidden_dim".format(self.gnn_type, self.hid_dim)
        save_path = get_pretrain_save_path(self.dataset_name, 'GraphCL', suffix=suffix)
        torch.save(self.gnn.state_dict(), save_path)
        model_saved(save_path)
