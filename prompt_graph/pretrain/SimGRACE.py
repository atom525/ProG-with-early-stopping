import sys
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from ..defines import NODE_TASKS
from prompt_graph.utils import mkdir
from torch.optim import Adam
from prompt_graph.data import load4node, load4graph, NodePretrain
from copy import deepcopy
from .base import PreTrain
from prompt_graph.utils.train_logger import train_info, epoch_training, epoch_evaluating, valid_result, finished_training, early_stopping_msg, model_saved, metric_from_dict, to_ordered_metrics, best_valid_ordered
import time
import os

class SimGRACE(PreTrain):
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
        
    def get_loader(self, graph_list, batch_size):

        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in SimGRACE!")

        loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        return loader
    
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

    def perturbate_gnn(self, data):
        vice_model = deepcopy(self).to(self.device)

        for (vice_name, vice_model_param) in vice_model.named_parameters():
            if vice_name.split('.')[0] != 'projection_head':
                std = vice_model_param.data.std() if vice_model_param.data.numel() > 1 else torch.tensor(1.0)
                noise = 0.1 * torch.normal(0, torch.ones_like(vice_model_param.data) * std)
                vice_model_param.data += noise
        z2 = vice_model.forward_cl(data.x, data.edge_index, data.batch)
        return z2
    
    def evaluate_valid_acc(self):
        """验证集：扰动后同一图嵌入相似度高于阈值的比例"""
        self.eval()
        device = self.device
        correct, total = 0, 0
        with torch.no_grad():
            for data in self.val_graph_list:
                data = data.to(device)
                x1 = self.forward_cl(data.x, data.edge_index, data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(data.num_nodes, dtype=torch.long, device=device))
                x2 = self.perturbate_gnn(data)
                x1 = x1.mean(dim=0, keepdim=True)
                x2 = x2.mean(dim=0, keepdim=True)
                sim = (x1 * x2).sum() / (x1.norm() * x2.norm() + 1e-8)
                if sim.item() > 0.5:
                    correct += 1
                total += 1
        self.train()
        return correct / total if total > 0 else 0.0

    def train_simgrace(self, loader, optimizer):
        self.train()
        train_loss_accum = 0
        total_step = 0
        pbar = tqdm(loader, desc="Batch", leave=False, file=sys.stderr)
        for step, data in enumerate(pbar):
            optimizer.zero_grad()
            data = data.to(self.device)
            x2 = self.perturbate_gnn(data) 
            x1 = self.forward_cl(data.x, data.edge_index, data.batch)
            x2 = Variable(x2.detach().data.to(self.device), requires_grad=False)
            loss = self.loss_cl(x1, x2)
            loss.backward()
            optimizer.step()
            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1
            pbar.set_postfix(loss=f"{train_loss_accum/total_step:.4f}")
        return train_loss_accum / total_step

    def pretrain(self, batch_size=10, lr=0.01, decay=0.0001, args=None):
        num_epoch = getattr(args, 'epochs', self.epochs) if args else self.epochs
        patience = getattr(args, 'patience', 20) if args else 20
        eval_every = (getattr(args, 'eval_every', 1) or 1) if args else 1
        if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa', 'DD']:
            batch_size = 512
        loader = self.get_loader(self.graph_list, batch_size)
        train_info('start training {} | {} | {}'.format(self.dataset_name, 'SimGRACE', self.gnn_type))
        optimizer = optim.Adam(self.gnn.parameters(), lr=lr, weight_decay=decay)
        early_stopping_metric = getattr(args, 'early_stopping_metric', 'valid_acc') or 'valid_acc'
        met_name = early_stopping_metric
        best_val_metric = -1.0
        best_epoch = 0
        cnt_wait = 0
        best_state = None
        best_valid_metrics = {}
        for epoch in range(1, num_epoch + 1):
            st_time = time.time()
            train_loss = self.train_simgrace(loader, optimizer)
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
        save_path = get_pretrain_save_path(self.dataset_name, 'SimGRACE', suffix=suffix)
        torch.save(self.gnn.state_dict(), save_path)
        model_saved(save_path)
