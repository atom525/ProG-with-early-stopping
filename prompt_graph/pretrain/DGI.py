import sys
from ..defines import GRAPH_TASKS, NODE_TASKS
from .base import PreTrain
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn.inits import reset, uniform
from torch.optim import Adam
import torch
from tqdm import tqdm
from torch import nn
import time
from prompt_graph.utils import generate_corrupted_graph
from prompt_graph.utils.train_logger import train_info, epoch_training, epoch_evaluating, valid_result, finished_training, early_stopping_msg, model_saved, metric_from_dict, to_ordered_metrics, best_valid_ordered
from prompt_graph.data import load4node, load4graph, NodePretrain
import os
import numpy as np
import copy

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class DGI(PreTrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._val_ratio = kwargs.pop('pretrain_val_ratio', 0.1) or 0.1
        self.disc = Discriminator(self.hid_dim).to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.graph_data, self.val_data = self.load_data()
        self.initialize_gnn(self.input_dim, self.hid_dim)
        self.optimizer = Adam(self.gnn.parameters(), lr=0.001, weight_decay=0.0)

    # def load_data(self):
    #     if self.dataset_name in ['PubMed', 'CiteSeer', 'Cora','Computers', 'Photo', 'Reddit', 'WikiCS', 'Flickr', 'ogbn-arxiv']:
    #         data, input_dim, _ = load4node(self.dataset_name)
    #         self.input_dim = input_dim
    #     elif self.dataset_name in GRAPH_TASKS:
    #         input_dim, _, graph_list= load4graph(self.dataset_name,pretrained=True) # need graph list not dataset object, so the pretrained = True
    #         self.input_dim = input_dim
    #         graph_data_batch = Batch.from_data_list(graph_list)
    #         data= Data(x=graph_data_batch.x, edge_index=graph_data_batch.edge_index)
            
    #         if self.dataset_name in  ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY']:
    #             from torch_geometric import loader
    #             self.batch_dataloader = loader.DataLoader(graph_list,batch_size=512,shuffle=False)

    #     return data


    # def pretrain_one_epoch(self):
    #     self.gnn.train()
    #     self.optimizer.zero_grad()
    #     device = self.device

    #     graph_original = self.graph_data
    #     graph_corrupted = copy.deepcopy(graph_original)
    #     idx_perm = np.random.permutation(graph_original.x.size(0))
    #     graph_corrupted.x = graph_original.x[idx_perm].to(self.device)

    #     graph_original.to(device)
    #     graph_corrupted.to(device)

    #     pos_z = self.gnn(graph_original.x, graph_original.edge_index)
    #     neg_z = self.gnn(graph_corrupted.x, graph_corrupted.edge_index)

    #     s = torch.sigmoid(torch.mean(pos_z, dim=0)).to(device)
        

    #     logits = self.disc(s, pos_z, neg_z)

    #     lbl_1 = torch.ones((pos_z.shape[0], 1))
    #     lbl_2 = torch.zeros((neg_z.shape[0], 1))
    #     lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

    #     loss = self.loss_fn(logits, lbl)
    #     loss.backward()
    #     self.optimizer.step()

    #     accum_loss = float(loss.detach().cpu().item())
    #     return accum_loss

    def load_data(self):
        if self.dataset_name in NODE_TASKS:
            data, input_dim, _ = load4node(self.dataset_name)
            self.input_dim = input_dim
            self.val_batch_dataloader = None
            return data, None
        elif self.dataset_name in GRAPH_TASKS:
            input_dim, _, graph_list = load4graph(self.dataset_name, pretrained=True)
            self.input_dim = input_dim
            from torch_geometric import loader
            n = len(graph_list)
            perm = np.random.RandomState(42).permutation(n)
            n_val = max(1, int(n * self._val_ratio))
            train_list, val_list = [graph_list[i] for i in perm[n_val:]], [graph_list[i] for i in perm[:n_val]]
            self.batch_dataloader = loader.DataLoader(train_list, batch_size=512, shuffle=False, num_workers=self.num_workers)
            self.val_batch_dataloader = loader.DataLoader(val_list, batch_size=min(512, len(val_list)), shuffle=False, num_workers=0)
            return train_list, val_list
        return None, None

    def evaluate_valid_acc(self):
        """验证集：判别器对正/负样本的预测准确率"""
        self.gnn.eval()
        self.disc.eval()
        device = self.device
        correct, total = 0, 0
        with torch.no_grad():
            if self.dataset_name in NODE_TASKS or self.val_batch_dataloader is None:
                graph_original = self.graph_data
                graph_corrupted = copy.deepcopy(graph_original)
                idx_perm = np.random.RandomState(0).permutation(graph_original.x.size(0))
                graph_corrupted.x = graph_original.x[idx_perm]
                graph_original, graph_corrupted = graph_original.to(device), graph_corrupted.to(device)
                pos_z = self.gnn(graph_original.x, graph_original.edge_index)
                neg_z = self.gnn(graph_corrupted.x, graph_corrupted.edge_index)
                s = torch.sigmoid(torch.mean(pos_z, dim=0))
                logits = self.disc(s, pos_z, neg_z)
                pred = (logits[:, 0] > logits[:, 1]).long()
                lbl = torch.ones(pos_z.shape[0], dtype=torch.long, device=device)
                correct = (pred == lbl).sum().item()
                total = pos_z.shape[0]
            else:
                for batch_graph in self.val_batch_dataloader:
                    graph_original = batch_graph.to(device)
                    graph_corrupted = copy.deepcopy(graph_original)
                    idx_perm = np.random.RandomState(0).permutation(graph_original.x.size(0))
                    graph_corrupted.x = graph_original.x[idx_perm].to(device)
                    pos_z = self.gnn(graph_original.x, graph_original.edge_index)
                    neg_z = self.gnn(graph_corrupted.x, graph_corrupted.edge_index)
                    s = torch.sigmoid(torch.mean(pos_z, dim=0))
                    logits = self.disc(s, pos_z, neg_z)
                    pred = (logits[:, 0] > logits[:, 1]).long()
                    lbl = torch.ones(pos_z.shape[0], dtype=torch.long, device=device)
                    correct += (pred == lbl).sum().item()
                    total += pos_z.shape[0]
        self.gnn.train()
        self.disc.train()
        return correct / total if total > 0 else 0.0

    def pretrain_one_epoch(self):
        self.gnn.train()
        self.optimizer.zero_grad()
        device = self.device

        if self.dataset_name in NODE_TASKS:
            graph_original = self.graph_data
            graph_corrupted = copy.deepcopy(graph_original)
            idx_perm = np.random.permutation(graph_original.x.size(0))
            graph_corrupted.x = graph_original.x[idx_perm].to(self.device)

            graph_original.to(device)
            graph_corrupted.to(device)

            pos_z = self.gnn(graph_original.x, graph_original.edge_index)
            neg_z = self.gnn(graph_corrupted.x, graph_corrupted.edge_index)

            s = torch.sigmoid(torch.mean(pos_z, dim=0)).to(device)

            logits = self.disc(s, pos_z, neg_z)

            lbl_1 = torch.ones((pos_z.shape[0], 1))
            lbl_2 = torch.zeros((neg_z.shape[0], 1))
            lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

            loss = self.loss_fn(logits, lbl)
            loss.backward()
            self.optimizer.step()

            accum_loss = float(loss.detach().cpu().item())            
        elif self.dataset_name in GRAPH_TASKS:
            accum_loss = torch.tensor(0.0)
            pbar = tqdm(self.batch_dataloader, desc="Batch", leave=False, file=sys.stderr)
            for batch_id, batch_graph in enumerate(pbar):
                graph_original = batch_graph.to(device)
                graph_corrupted = copy.deepcopy(graph_original)
                idx_perm = np.random.permutation(graph_original.x.size(0))
                graph_corrupted.x = graph_original.x[idx_perm].to(self.device)

                graph_original.to(device)
                graph_corrupted.to(device)

                pos_z = self.gnn(graph_original.x, graph_original.edge_index)
                neg_z = self.gnn(graph_corrupted.x, graph_corrupted.edge_index)
        
                s = torch.sigmoid(torch.mean(pos_z, dim=0)).to(device)

                logits = self.disc(s, pos_z, neg_z)

                lbl_1 = torch.ones((pos_z.shape[0], 1))
                lbl_2 = torch.zeros((neg_z.shape[0], 1))
                lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

                loss = self.loss_fn(logits, lbl)
                loss.backward()
                self.optimizer.step()

                accum_loss += float(loss.detach().cpu().item())
                pbar.set_postfix(loss=f"{accum_loss/(batch_id+1):.4f}")
            accum_loss = accum_loss/(batch_id+1)

        return accum_loss    
            


    def pretrain(self, args=None):
        num_epoch = getattr(args, 'epochs', self.epochs) if args else self.epochs
        patience = getattr(args, 'patience', 20) if args else 20
        eval_every = (getattr(args, 'eval_every', 1) or 1) if args else 1
        early_stopping_metric = getattr(args, 'early_stopping_metric', 'valid_acc') or 'valid_acc'
        met_name = early_stopping_metric
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
        save_path = get_pretrain_save_path(self.dataset_name, 'DGI', suffix=suffix)
        torch.save(self.gnn.state_dict(), save_path)
        model_saved(save_path)
