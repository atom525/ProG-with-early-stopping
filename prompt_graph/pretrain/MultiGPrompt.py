import torch
import torch.nn as nn
import torch.nn.functional as F
from prompt_graph.prompt import DGI,GraphCL,Lp,AvgReadout, DGIprompt,GraphCLprompt,Lpprompt, GcnLayers
import scipy.sparse as sp
import numpy as np
from prompt_graph.utils import process
import prompt_graph.utils.aug as aug
import os
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange
from prompt_graph.utils.paths import get_pretrain_save_path
from prompt_graph.utils.train_logger import train_info, epoch_training, epoch_evaluating, valid_result, finished_training, early_stopping_msg, model_saved, metric_from_dict, to_ordered_metrics, best_valid_ordered
import time

class NodePrePrompt(nn.Module):
    def __init__(self, dataset_name, n_h, activation,a1,a2,a3, a4, num_layers_num, dropout, device):
        super(NodePrePrompt, self).__init__()
        self.dataset_name = dataset_name
        self.device = torch.device('cuda:' + str(device) if torch.cuda.is_available() else 'cpu')
        self.hid_dim = n_h
        n_in, self.nb_nodes = self.load_data()
        self.dgi = DGI(n_in, n_h, activation)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.lp = Lp(n_in, n_h)
        self.gcn = GcnLayers(n_in, n_h, num_layers_num, dropout)
        self.read = AvgReadout()
        self.weighted_feature=weighted_feature(a1,a2,a3)
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.dgiprompt = DGIprompt(n_in, n_h, activation)
        self.graphcledgeprompt = GraphCLprompt(n_in, n_h, activation)
        self.lpprompt = Lpprompt(n_in, n_h)
        sample = self.negetive_sample
        self.sample = torch.tensor(sample, dtype=int).to(self.device)
        self.loss = nn.BCEWithLogitsLoss()
        self.act = nn.ELU()

    def load_data(self):
        self.adj, features, self.labels = process.load_data(self.dataset_name)
        # self.adj, features, self.labels = process.load_data(self.dataset_name)  
        self.features, _ = process.preprocess_features(features)
        
        if self.dataset_name in ['Texas','Wisconsin']:
            self.negetive_sample = prompt_pretrain_sample(self.adj,50)
        else:
            self.negetive_sample = prompt_pretrain_sample(self.adj,200)
        # prompt_pretrain_sample为图中的每个节点提供了一个正样本和多个负样本的索引
        nb_nodes = self.features.shape[0]  # node number
        ft_size = self.features.shape[1]  # node features dim
        nb_classes = self.labels.shape[1]  # classes = 6
        return ft_size, nb_nodes

    def forward(self, seq1, seq2, seq3, seq4, seq5, seq6, adj, aug_adj1edge, aug_adj2edge, aug_adj1mask, aug_adj2mask,
                sparse, msk, samp_bias1, samp_bias2, lbl):
        seq1 = torch.squeeze(seq1,0)
        seq2 = torch.squeeze(seq2,0)
        seq3 = torch.squeeze(seq3,0)
        seq4 = torch.squeeze(seq4,0)
        logits1 = self.dgi(self.gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2)
        logits2 = self.graphcledge(self.gcn, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge, sparse, msk,
                                   samp_bias1,
                                   samp_bias2, aug_type='edge')
        logits3 = self.lp(self.gcn,seq1,adj,sparse)
        
        
        logits4 = self.dgiprompt(self.gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2)
        logits5 = self.graphcledgeprompt(self.gcn, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge, sparse, msk,
                                   samp_bias1,
                                   samp_bias2, aug_type='edge')
        logits6 = self.lpprompt(self.gcn,seq1,adj,sparse)


        logits11 = logits1 + self.a4*logits4
        logits22 = logits2 + self.a4*logits5
        logits33 = logits3 + self.a4*logits6

        dgiloss = self.loss(logits11, lbl)
        graphcledgeloss = self.loss(logits22, lbl)
        lploss = compareloss(logits33,self.sample,temperature=1.5, device = self.device)
        lploss.requires_grad_(True)
        
        ret = self.a1 * dgiloss + self.a2 * graphcledgeloss + self.a3 * lploss

        return ret

    def embed(self, seq, adj, sparse, msk,LP):
        h_1 = self.gcn(seq, adj, sparse,LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()
    
    def pretrain(self, args=None):
        batch_size = 1
        nb_epochs = getattr(args, 'epochs', 1000) if args else 1000
        patience = getattr(args, 'patience', 20) if args else 20
        eval_every = (getattr(args, 'eval_every', 1) or 1) if args else 1
        early_stopping_metric = getattr(args, 'early_stopping_metric', 'valid_acc') or 'valid_acc'
        met_name = early_stopping_metric
        best_val_metric = -1.0
        lr = 0.0001
        l2_coef = 0.0
        hid_units = 256
        sparse = True

        features = torch.FloatTensor(self.features[np.newaxis])
        # 将features数组转换为PyTorch的FloatTensor类型，并增加一个新的维度
        '''
        # ------------------------------------------------------------
        # edge node mask subgraph
        # ------------------------------------------------------------
        '''
        # print("Begin Aug:[{}]".format(args.aug_type))
        # if args.aug_type == 'edge':
        adj = self.adj
        aug_features1edge = features
        aug_features2edge = features

        drop_percent = 0.1
        aug_adj1edge = aug.aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges
        aug_adj2edge = aug.aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges


        aug_features1mask = aug.aug_random_mask(features, drop_percent=drop_percent)
        aug_features2mask = aug.aug_random_mask(features, drop_percent=drop_percent)

        aug_adj1mask = adj
        aug_adj2mask = adj

        '''
        # ------------------------------------------------------------
        '''

        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        aug_adj1edge = process.normalize_adj(aug_adj1edge + sp.eye(aug_adj1edge.shape[0]))
        aug_adj2edge = process.normalize_adj(aug_adj2edge + sp.eye(aug_adj2edge.shape[0]))

        aug_adj1mask = process.normalize_adj(aug_adj1mask + sp.eye(aug_adj1mask.shape[0]))
        aug_adj2mask = process.normalize_adj(aug_adj2mask + sp.eye(aug_adj2mask.shape[0]))

        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        sp_aug_adj1edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj1edge)
        sp_aug_adj2edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj2edge)

        sp_aug_adj1mask = process.sparse_mx_to_torch_sparse_tensor(aug_adj1mask)
        sp_aug_adj2mask = process.sparse_mx_to_torch_sparse_tensor(aug_adj2mask)

        labels = torch.FloatTensor(self.labels[np.newaxis])
        # print("labels",labels)
        train_info("adj {} feature {}".format(sp_adj.shape, features.shape))
        LP = False
        lr = 0.0001

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2_coef)
        if torch.cuda.is_available():
            train_info('Using CUDA')
            self = self.to(self.device)
            features = features.to(self.device)
            aug_features1edge = aug_features1edge.to(self.device)
            aug_features2edge = aug_features2edge.to(self.device)
            aug_features1mask = aug_features1mask.to(self.device)
            aug_features2mask = aug_features2mask.to(self.device)
     
            sp_adj = sp_adj.to(self.device)
            sp_aug_adj1edge = sp_aug_adj1edge.to(self.device)
            sp_aug_adj2edge = sp_aug_adj2edge.to(self.device)
            sp_aug_adj1mask = sp_aug_adj1mask.to(self.device)
            sp_aug_adj2mask = sp_aug_adj2mask.to(self.device)

            labels = labels.to(self.device)
  
    
        cnt_wait = 0
        best_epoch = 0
        best_state = None

        for epoch in range(nb_epochs):
            st_time = time.time()
            self.train()
            optimizer.zero_grad()
            idx = np.random.permutation(self.nb_nodes)
            shuf_fts = features[:, idx, :]
            lbl_1 = torch.ones(batch_size, self.nb_nodes)
            lbl_2 = torch.zeros(batch_size, self.nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)
            if torch.cuda.is_available():
                shuf_fts = shuf_fts.to(self.device)
                lbl = lbl.to(self.device)
            loss = self(features, shuf_fts, aug_features1edge, aug_features2edge, aug_features1mask, aug_features2mask,
                        sp_adj if sparse else adj,
                        sp_aug_adj1edge if sparse else aug_adj1edge,
                        sp_aug_adj2edge if sparse else aug_adj2edge,
                        sp_aug_adj1mask if sparse else aug_adj1mask,
                        sp_aug_adj2mask if sparse else aug_adj2mask,
                        sparse, None, None, None, lbl=lbl)
            loss.backward()
            optimizer.step()
            train_time = time.time() - st_time
            train_loss = loss.item()

            if (epoch + 1) % eval_every == 0 or epoch == 0:
                epoch_training(epoch + 1, train_time, train_loss)
                eval_st = time.time()
                valid_acc = self.evaluate_valid_acc()
                eval_time = time.time() - eval_st
                valid_metrics = {"valid_acc": valid_acc}
                val_metric = metric_from_dict(valid_metrics, met_name)
                epoch_evaluating(epoch + 1, eval_time, val_metric, met_name)
                valid_result(valid_metrics)
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_epoch = epoch + 1
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                    if cnt_wait >= patience:
                        early_stopping_msg(epoch + 1, patience, met_name)
                        break

        if best_state is not None:
            self.load_state_dict(best_state)
            self.to(self.device)
        finished_training(best_epoch if best_epoch > 0 else 1)
        best_valid_ordered({"valid_acc": best_val_metric})
        save_path = get_pretrain_save_path(self.dataset_name, 'MultiGprompt')
        torch.save(self.state_dict(), save_path)
        model_saved(save_path)

    def evaluate_valid_acc(self):
        """验证：DGI 判别器准确率（正/负样本预测正确比例）"""
        self.eval()
        device = self.device
        with torch.no_grad():
            adj = process.normalize_adj(self.adj + sp.eye(self.adj.shape[0]))
            sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj).to(device)
            seq = torch.FloatTensor(self.features[np.newaxis]).to(device)
            idx_perm = np.random.RandomState(0).permutation(self.nb_nodes)
            shuf_fts = seq[:, idx_perm, :].clone()
            logits = self.dgi(self.gcn, seq.squeeze(0), shuf_fts.squeeze(0), sp_adj, True, None, None, None)
            pred = (logits[:, 0] > logits[:, 1]).long()
            lbl = torch.ones(self.nb_nodes, dtype=torch.long, device=device)
            acc = (pred == lbl).float().mean().item()
        self.train()
        return acc


class GraphPrePrompt(nn.Module):
    def __init__(self, graph, n_in, n_out, dataset_name, n_h, activation, a1, a2, a3, num_layers_num, p, device, pretrain_val_ratio=0.1):
        super(GraphPrePrompt, self).__init__()
        self.dataset_name = dataset_name
        n = len(graph)
        perm = np.random.RandomState(42).permutation(n)
        n_val = max(1, int(n * pretrain_val_ratio))
        self.graph_list = [graph[int(i)] for i in perm[n_val:]]
        self.val_graph_list = [graph[int(i)] for i in perm[:n_val]]
        self.loader = self.get_loader()
        self.device = torch.device('cuda:' + str(device) if torch.cuda.is_available() else 'cpu')
        self.dgi = DGI(n_in, n_h, activation)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.lp = Lp(n_in, n_h)
        self.gcn = GcnLayers(n_in, n_h,num_layers_num,p)
        self.read = AvgReadout()
        self.input_dim = n_in
        self.output_dim = n_out
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

        self.loss = nn.BCEWithLogitsLoss()

    def get_loader(self):
        loader = DataLoader(self.graph_list, batch_size = 32, shuffle=True,drop_last=True)
        return loader
    
    def forward(self, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge, 
                sparse, msk, samp_bias1, samp_bias2,
                lbl,sample):
        negative_sample = torch.tensor(sample,dtype=int).to(self.device)
        seq1 = torch.squeeze(seq1,0)
        seq2 = torch.squeeze(seq2,0)
        seq3 = torch.squeeze(seq3,0)
        seq4 = torch.squeeze(seq4,0)
        logits1 = self.dgi(self.gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2)
        logits2 = self.graphcledge(self.gcn, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge, sparse, msk,
                                   samp_bias1,
                                   samp_bias2, aug_type='edge')
        logits3 = self.lp(self.gcn,seq1,adj,sparse)
        dgiloss = self.loss(logits1, lbl)
        graphcledgeloss = self.loss(logits2, lbl)
        lploss = compareloss(logits3,negative_sample,temperature=1.5, device = self.device)
        lploss.requires_grad_(True)
        
        ret =self.a1*dgiloss+self.a2*graphcledgeloss+self.a3*lploss
        return ret

    def embed(self, seq, adj, sparse, msk, LP):
        h_1 = self.gcn(seq, adj, sparse, LP)
        c = self.read(h_1, msk)
        return h_1.detach(), c.detach()

    def evaluate_valid_acc(self):
        """验证集：链接预测准确率 (cos_sim(node,pos) > cos_sim(node,neg))"""
        self.eval()
        device = self.device
        correct, total = 0, 0
        with torch.no_grad():
            for g in self.val_graph_list:
                try:
                    features, adj_scipy = process.process_tu(g, self.output_dim, self.input_dim)
                except Exception:
                    continue
                nb_nodes = features.shape[0]
                tuples = tu_prompt_pretrain_sample(adj_scipy, 50)
                tuples = torch.LongTensor(tuples).to(device)
                features = torch.FloatTensor(features[np.newaxis]).to(device)
                adj = process.normalize_adj(adj_scipy + sp.eye(adj_scipy.shape[0]))
                adj_dense = torch.FloatTensor(adj.todense()[np.newaxis]).to(device)
                h = self.gcn(features.squeeze(0), adj_dense.squeeze(0), False, False)
                if h.dim() == 3:
                    h = h[0]
                else:
                    h = h.squeeze(0)
                for i in range(min(nb_nodes, tuples.shape[0])):
                    pos_i, neg_i = int(tuples[i, 0]), int(tuples[i, 1])
                    if pos_i >= h.size(0) or neg_i >= h.size(0):
                        continue
                    sim_pos = F.cosine_similarity(h[i:i+1], h[pos_i:pos_i+1], dim=1).item()
                    sim_neg = F.cosine_similarity(h[i:i+1], h[neg_i:neg_i+1], dim=1).item()
                    if sim_pos > sim_neg:
                        correct += 1
                    total += 1
        self.train()
        return correct / total if total > 0 else 0.0

    def pretrain(self, args=None):
        nb_epochs = getattr(args, 'epochs', 1000) if args else 1000
        patience = getattr(args, 'patience', 20) if args else 20
        eval_every = (getattr(args, 'eval_every', 1) or 1) if args else 1
        early_stopping_metric = getattr(args, 'early_stopping_metric', 'valid_acc') or 'valid_acc'
        met_name = early_stopping_metric
        best_val_metric = -1.0
        best_epoch = 0
        cnt_wait = 0
        best_state = None
        self.to(self.device)
        optimiser = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0)
        for epoch in range(nb_epochs):
            loss = 0
            drop_percent = 0.1
            for step, batch in enumerate(self.loader):

                features,adj =  process.process_tu(batch, self.output_dim, self.input_dim)
                negetive_sample = tu_prompt_pretrain_sample(adj,50)
                nb_nodes = features.shape[0]  # node number
                features = torch.FloatTensor(features[np.newaxis])

                
                aug_features1edge = features
                aug_features2edge = features

                aug_adj1edge = aug.aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges
                aug_adj2edge = aug.aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges


                adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
                aug_adj1edge = process.normalize_adj(aug_adj1edge + sp.eye(aug_adj1edge.shape[0]))
                aug_adj2edge = process.normalize_adj(aug_adj2edge + sp.eye(aug_adj2edge.shape[0]))

                adj = (adj + sp.eye(adj.shape[0])).todense()
                aug_adj1edge = (aug_adj1edge + sp.eye(aug_adj1edge.shape[0])).todense()
                aug_adj2edge = (aug_adj2edge + sp.eye(aug_adj2edge.shape[0])).todense()

        
                adj = torch.FloatTensor(adj[np.newaxis])
                aug_adj1edge = torch.FloatTensor(aug_adj1edge[np.newaxis])
                aug_adj2edge = torch.FloatTensor(aug_adj2edge[np.newaxis])

                if torch.cuda.is_available():
                    # print('Using CUDA')
                    # model = torch.nn.DataParallel(model, device_ids=[0,1]).to(self.device)
                    features = features.to(self.device)
                    aug_features1edge = aug_features1edge.to(self.device)
                    aug_features2edge = aug_features2edge.to(self.device)
                    adj = adj.to(self.device)
                    aug_adj1edge = aug_adj1edge.to(self.device)
                    aug_adj2edge = aug_adj2edge.to(self.device)
                b_xent = nn.BCEWithLogitsLoss()
                xent = nn.CrossEntropyLoss()
                self.train()
                optimiser.zero_grad()
                idx = np.random.permutation(nb_nodes)
                shuf_fts = features[:, idx, :]
                lbl_1 = torch.ones(1, nb_nodes)
                lbl_2 = torch.zeros(1, nb_nodes)
                lbl = torch.cat((lbl_1, lbl_2), 1)
                if torch.cuda.is_available():
                    shuf_fts = shuf_fts.to(self.device)
                    lbl = lbl.to(self.device)
                logit = self(features, shuf_fts, aug_features1edge, aug_features2edge,
                            adj,
                            aug_adj1edge,
                            aug_adj2edge,
                            False, None, None, None, lbl=lbl,sample=negetive_sample)
                logit.backward()
                optimiser.step()
                loss = loss + logit.item()
            loss = loss / (step + 1)
            if (epoch + 1) % eval_every == 0 or epoch == 0:
                epoch_training(epoch + 1, 0.0, loss)
                eval_st = time.time()
                valid_acc = self.evaluate_valid_acc()
                eval_time = time.time() - eval_st
                valid_metrics = {"valid_acc": valid_acc}
                val_metric = metric_from_dict(valid_metrics, met_name)
                epoch_evaluating(epoch + 1, eval_time, val_metric, met_name)
                valid_result(valid_metrics)
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_epoch = epoch + 1
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                    if cnt_wait >= patience:
                        early_stopping_msg(epoch + 1, patience, met_name)
                        break

        if best_state is not None:
            self.load_state_dict(best_state)
            self.to(self.device)
        finished_training(best_epoch if best_epoch > 0 else 1)
        best_valid_ordered({"valid_acc": best_val_metric})
        save_path = get_pretrain_save_path(self.dataset_name, 'MultiGprompt')
        torch.save(self.state_dict(), save_path)
        model_saved(save_path)


def mygather(feature, index): 
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size,-1,feature.size(1))


def compareloss(feature,tuples,temperature,device):
    h_tuples=mygather(feature,tuples)
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    temp=temp.to(device)
    h_i = mygather(feature, temp)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    # print("sim",sim)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()


def prompt_pretrain_sample(adj,n):
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    res=np.zeros((nodenum,1+n))
    whole=np.array(range(nodenum))
    train_info("start sampling disconnected tuples")
    for i in trange(nodenum):
        nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
        zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row)==0:
            res[i][0] = i
        else:
            res[i][0]=nonzero_index_i_row[0]
        res[i][1:1+n]=zero_index_i_row[0:n]
    return res.astype(int)

def tu_prompt_pretrain_sample(adj,n):
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    res=np.zeros((nodenum,1+n))
    whole=np.array(range(nodenum))
    # print("#############")
    # print("start sampling disconnected tuples")
    for i in range(nodenum):
        nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
        zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row)==0:
            res[i][0] = i
        else:
            res[i][0]=nonzero_index_i_row[0]
        res[i][1:1+n]=zero_index_i_row[0:n]
    return res.astype(int)

class weighted_feature(nn.Module):
    def __init__(self,a1,a2,a3):
        super(weighted_feature, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,3), requires_grad=True)
        self.reset_parameters(a1,a2,a3)
    def reset_parameters(self,a1,a2,a3):
        # torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(a1)
        self.weight[0][1].data.fill_(a2)
        self.weight[0][2].data.fill_(a3)
    def forward(self, graph_embedding1,graph_embedding2,graph_embedding3):
        graph_embedding= self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2 + self.weight[0][2] * graph_embedding3
        return graph_embedding