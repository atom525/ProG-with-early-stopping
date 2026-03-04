"""
LinkTask: 链接预测下游任务
- 与 NodeTask/GraphTask 统一配置：split_ratio, patience, early_stopping_metric
- 使用 train_logger 统一日志
- 仅支持 prompt_type='None'，使用 GNN.decode 做边预测
"""
import os
import time
import torch
from torch import nn, optim
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

from .task import BaseTask
from prompt_graph.data import load4link_downstream
from prompt_graph.model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer
from prompt_graph.utils.train_logger import (
    train_info, epoch_training, epoch_evaluating, valid_result,
    best_valid_ordered, test_result_ordered, finished_training,
    early_stopping_msg, model_loaded,
)


class LinkTask(BaseTask):
    """链接预测任务：仅 GNN + decode，无 prompt/answering"""

    def __init__(self, dataset_name='Cora', gnn_type='GCN', hid_dim=128, num_layer=2,
                 epochs=100, device=0, lr=0.005, wd=5e-4,
                 split_ratio=None, patience=20, save_best=True, checkpoint_dir='checkpoints/downstream',
                 eval_every=1, early_stopping_metric='valid_auroc', log_dir='logs', log_file=None,
                 pre_train_model_path='None', **kwargs):
        self.task_type = 'LinkTask'
        self.prompt_type = 'None'
        self.device = torch.device('cuda:{}'.format(device) if torch.cuda.is_available() else 'cpu')
        split_ratio = split_ratio or [0.8, 0.1, 0.1]
        self.split_ratio = split_ratio
        self.patience = patience
        self.epochs = epochs
        super().__init__(
            pre_train_model_path=pre_train_model_path, gnn_type=gnn_type, hid_dim=hid_dim,
            num_layer=num_layer, dataset_name=dataset_name, prompt_type='None', epochs=epochs,
            shot_num=0, device=device, lr=lr, wd=wd, batch_size=1,
            split_ratio=split_ratio, patience=patience, save_best=save_best,
            checkpoint_dir=checkpoint_dir, eval_every=eval_every,
            early_stopping_metric=early_stopping_metric or 'valid_auroc',
            log_dir=log_dir, log_file=log_file,
        )
        self.pre_train_type = self.return_pre_train_type(pre_train_model_path)
        train_data, val_data, test_data, input_dim, output_dim = load4link_downstream(dataset_name, split_ratio)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_data = train_data.to(self.device)
        self.val_data = val_data.to(self.device)
        self.test_data = test_data.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.initialize_gnn()
        self.optimizer = optim.Adam(self.gnn.parameters(), lr=lr, weight_decay=wd)
        train_info("LinkTask initialized | {} | {} | split_ratio {}".format(dataset_name, gnn_type, split_ratio))

    def initialize_gnn(self):
        if self.gnn_type == 'GAT':
            self.gnn = GAT(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GCN':
            self.gnn = GCN(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphSAGE':
            self.gnn = GraphSAGE(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GIN':
            self.gnn = GIN(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GCov':
            self.gnn = GCov(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphTransformer':
            self.gnn = GraphTransformer(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        else:
            raise ValueError("Unsupported GNN type: {}".format(self.gnn_type))
        self.gnn.to(self.device)
        if self.pre_train_model_path != 'None':
            if self.gnn_type not in self.pre_train_model_path:
                raise ValueError("GNN type '{}' does not match pre-train path".format(self.gnn_type))
            if self.dataset_name not in self.pre_train_model_path:
                raise ValueError("Dataset '{}' does not match pre-train path".format(self.dataset_name))
            self.gnn.load_state_dict(torch.load(self.pre_train_model_path, map_location='cpu'))
            self.gnn.to(self.device)
            train_info("Successfully loaded pre-trained weights from {}".format(self.pre_train_model_path))

    def train_one_epoch(self):
        self.gnn.train()
        self.optimizer.zero_grad()
        data = self.train_data
        node_emb = self.gnn(data.x, data.edge_index)
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.edge_label_index.size(1), method='sparse')
        edge_label_index = torch.cat([data.edge_label_index, neg_edge_index], dim=-1)
        edge_label = torch.cat([
            data.edge_label,
            data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)
        out = self.gnn.decode(node_emb, edge_label_index).view(-1)
        loss = self.criterion(out, edge_label.float())
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, data):
        self.gnn.eval()
        z = self.gnn(data.x, data.edge_index)
        out = self.gnn.decode(z, data.edge_label_index).view(-1).sigmoid()
        y = data.edge_label.cpu().numpy()
        pred = out.cpu().numpy()
        auroc = roc_auc_score(y, pred)
        auprc = average_precision_score(y, pred)
        pred_binary = (pred >= 0.5).astype(float)
        acc = (pred_binary == y).mean()
        from sklearn.metrics import f1_score
        f1 = f1_score(y, pred_binary, zero_division=0) if len(set(y)) > 1 else 0.0
        return acc, f1, auroc, auprc

    def run(self):
        patience = self.patience
        eval_every = self.eval_every or 1
        met_name = self.early_stopping_metric or 'valid_auroc'
        higher_is_better = True
        best_val_metric = -1e9
        best_epoch = 0
        best_valid_metrics = {}
        cnt_wait = 0
        ckpt_path = os.path.join(self.checkpoint_dir, f"{self.dataset_name}_LinkTask_{self.gnn_type}_best.pt") if self.save_best else None
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            loss = self.train_one_epoch()
            epoch_training(epoch, time.time() - t0, loss)

            if epoch % eval_every != 0:
                continue

            eval_st = time.time()
            va, vf1, vroc, vprc = self.evaluate(self.val_data)
            eval_time = time.time() - eval_st
            val_metric = self.metric_from_eva(va, vf1, vroc, vprc)
            valid_metrics = {"valid_acc": va, "valid_f1": vf1, "valid_auroc": vroc, "valid_auprc": vprc}
            epoch_evaluating(epoch, eval_time, val_metric, met_name)
            valid_result(valid_metrics)

            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_epoch = epoch
                best_valid_metrics = valid_metrics.copy()
                cnt_wait = 0
                if self.save_best and ckpt_path:
                    torch.save(self.gnn.state_dict(), ckpt_path)
                    train_info("  -> Checkpoint saved (best)")
            else:
                cnt_wait += 1
                if cnt_wait >= patience:
                    early_stopping_msg(epoch, patience, met_name)
                    break

        if self.save_best and ckpt_path and os.path.exists(ckpt_path):
            self.gnn.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            model_loaded(ckpt_path)

        finished_training(best_epoch if best_epoch > 0 else self.epochs)
        if best_valid_metrics:
            best_valid_ordered(best_valid_metrics)

        test_acc, test_f1, test_auroc, test_auprc = self.evaluate(self.test_data)
        test_result_ordered({"test_acc": test_acc, "test_f1": test_f1, "test_auroc": test_auroc, "test_auprc": test_auprc})
        train_info("{} {} LinkTask completed".format(self.pre_train_type, self.gnn_type))

        return 0.0, test_acc, 0.0, test_f1, 0.0, test_auroc, 0.0, test_auprc, 0.0
