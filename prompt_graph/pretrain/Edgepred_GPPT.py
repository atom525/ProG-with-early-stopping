import sys
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset, Subset
from tqdm import tqdm
from prompt_graph.data import load4link_prediction_multi_graph, load4link_prediction_single_graph
from torch.optim import Adam
import numpy as np
import time

from ..defines import GRAPH_TASKS, NODE_TASKS
from .base import PreTrain
from prompt_graph.utils.paths import get_pretrain_save_path
from prompt_graph.utils.train_logger import train_info, epoch_training, epoch_evaluating, valid_result, finished_training, early_stopping_msg, model_saved, metric_from_dict, to_ordered_metrics
import os

class Edgepred_GPPT(PreTrain):
    def __init__(self, *args, **kwargs):
        self._val_ratio = kwargs.pop('pretrain_val_ratio', 0.1) or 0.1
        super().__init__(*args, **kwargs)
        self.dataloader, self.val_dataloader = self.generate_loader_data()
        self.initialize_gnn(self.input_dim, self.hid_dim)
        self.graph_pred_linear = torch.nn.Linear(self.hid_dim, self.hid_dim).to(self.device)

    def generate_loader_data(self):
        if self.dataset_name in NODE_TASKS:
            self.data, edge_label, edge_index, self.input_dim, self.output_dim = load4link_prediction_single_graph(self.dataset_name)
            self.data.to(self.device)
            edge_index = edge_index.transpose(0, 1)
            data = TensorDataset(edge_label, edge_index)
            bs = 1024 if self.dataset_name in ['ogbn-arxiv', 'Flickr'] else 64
        elif self.dataset_name in GRAPH_TASKS:
            self.data, edge_label, edge_index, self.input_dim, self.output_dim = load4link_prediction_multi_graph(self.dataset_name)
            self.data.to(self.device)
            edge_index = edge_index.transpose(0, 1)
            data = TensorDataset(edge_label, edge_index)
            if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa', 'DD']:
                self.batch_dataloader = DataLoader(self.data.to_data_list(), batch_size=256, shuffle=False, num_workers=self.num_workers)
            bs = 512000 if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa', 'DD'] else 64
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")
        n = len(data)
        perm = np.random.RandomState(42).permutation(n)
        n_val = max(1, int(n * self._val_ratio))
        train_ds, val_ds = Subset(data, perm[n_val:]), Subset(data, perm[:n_val])
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_ds, batch_size=min(bs, len(val_ds)), shuffle=False, num_workers=0)
        return train_loader, val_loader

    def evaluate_valid_acc(self):
        """验证集上边预测二分类准确率"""
        self.gnn.eval()
        device = self.device
        correct, total = 0, 0
        with torch.no_grad():
            for batch_edge_label, batch_edge_index in self.val_dataloader:
                batch_edge_label = batch_edge_label.to(device)
                batch_edge_index = batch_edge_index.to(device)
                if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa', 'DD']:
                    out_list = []
                    for bg in self.batch_dataloader:
                        bg = bg.to(device)
                        out_list.append(self.gnn(bg.x, bg.edge_index))
                    out = torch.cat(out_list, dim=0)
                else:
                    out = self.gnn(self.data.x, self.data.edge_index)
                node_emb = self.graph_pred_linear(out)
                batch_edge_index = batch_edge_index.transpose(0, 1)
                pred_log = self.gnn.decode(node_emb, batch_edge_index).view(-1)
                pred = (pred_log > 0).long()
                correct += (pred == batch_edge_label.long()).sum().item()
                total += batch_edge_label.size(0)
        self.gnn.train()
        return correct / total if total > 0 else 0.0
      
    def pretrain_one_epoch(self):

        accum_loss, total_step = 0, 0
        device = self.device

        criterion = torch.nn.BCEWithLogitsLoss()
        
        self.gnn.train()
        pbar = tqdm(self.dataloader, desc="Batch", leave=False, file=sys.stderr)
        for step, (batch_edge_label, batch_edge_index) in enumerate(pbar):
            self.optimizer.zero_grad()

            batch_edge_label = batch_edge_label.to(device)
            batch_edge_index = batch_edge_index.to(device)

            # 如果graph datasets经过Batch图太大了，那就分开操作
            if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa', 'DD']:
                for batch_id, batch_graph in enumerate(self.batch_dataloader):
                    batch_graph.to(device)
                    if(batch_id==0):
                        out = self.gnn(batch_graph.x, batch_graph.edge_index)
                    else:
                        out = torch.concatenate([out, self.gnn(batch_graph.x, batch_graph.edge_index)],dim=0)
            else:
                out = self.gnn(self.data.x, self.data.edge_index)
                
            
            node_emb = self.graph_pred_linear(out)
          
            batch_edge_index = batch_edge_index.transpose(0,1)
            batch_pred_log = self.gnn.decode(node_emb,batch_edge_index).view(-1)
            loss = criterion(batch_pred_log, batch_edge_label)

            loss.backward()
            self.optimizer.step()

            accum_loss += float(loss.detach().cpu().item())
            total_step += 1
            pbar.set_postfix(loss=f"{accum_loss/total_step:.4f}")
        return accum_loss / total_step

    def pretrain(self, args=None):
        num_epoch = getattr(args, 'epochs', self.epochs) if args else self.epochs
        patience = getattr(args, 'patience', 20) if args else 20
        eval_every = (getattr(args, 'eval_every', 1) or 1) if args else 1
        early_stopping_metric = getattr(args, 'early_stopping_metric', 'valid_acc') or 'valid_acc'
        met_name = early_stopping_metric if early_stopping_metric == 'valid_acc' else 'valid_acc'
        best_val_metric = -1.0
        best_epoch = 0
        cnt_wait = 0
        best_state = None
        best_valid_metrics = {}
        for epoch in range(1, num_epoch + 1):
            st_time = time.time()
            train_loss = self.pretrain_one_epoch()
            train_time = time.time() - st_time
            epoch_training(epoch, train_time, train_loss)
            if epoch % eval_every == 0 or epoch == 1:
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
            train_info("best valid: {}".format(best_valid_metrics))
        suffix = ".{}.{}hidden_dim".format(self.gnn_type, self.hid_dim)
        save_path = get_pretrain_save_path(self.dataset_name, 'Edgepred_GPPT', suffix=suffix)
        torch.save(self.gnn.state_dict(), save_path)
        model_saved(save_path)

