import torch
from tqdm import tqdm
from prompt_graph.data import load4graph, load4node, graph_sample_and_save
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from ..defines import GRAPH_TASKS
from .task import BaseTask
from prompt_graph.registry import PromptRegistry
from prompt_graph.utils.paths import get_sample_data_graph_dir
from prompt_graph.utils import center_embedding, Gprompt_tuning_loss, constraint
from prompt_graph.utils.labels import safe_graph_label
from prompt_graph.evaluation import GpromptEva, GNNGraphEva, GPFEva, AllInOneEva, GPPTGraphEva, GraphMultiGpromptEva
from prompt_graph.utils.train_logger import train_info, epoch_training, epoch_evaluating, valid_result, best_valid_ordered, test_result_ordered, finished_training, early_stopping_msg, model_loaded, to_ordered_metrics
from prompt_graph.utils import process
import scipy.sparse as sp
import time
import os 
import numpy as np

class GraphTask(BaseTask):
    def __init__(self, input_dim, output_dim, dataset, task_num = 5 , *args, **kwargs):    
        super().__init__(*args, **kwargs)
        self.task_type = 'GraphTask'
        self.task_num = task_num
        # self.load_data()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dataset = dataset
        if self.shot_num > 0:
            self.create_few_data_folder()
        self.initialize_gnn()
        self.initialize_prompt()
        self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                            torch.nn.Softmax(dim=1)).to(self.device)
        self.initialize_optimizer()

    def create_few_data_folder(self):
            k = self.shot_num
            task_num = self.task_num
            base = get_sample_data_graph_dir(self.dataset_name)
            for k in range(1, task_num+1):
                k_shot_folder = os.path.join(base, str(k) + '_shot')
                os.makedirs(k_shot_folder, exist_ok=True)
                for i in range(1, task_num+1):
                    folder = os.path.join(k_shot_folder, str(i))
                    if not os.path.exists(folder):
                        os.makedirs(folder, exist_ok=True)
                        graph_sample_and_save(self.dataset, k, folder, self.output_dim, self.split_ratio)
                        train_info("{} shot {} th saved".format(k, i))

    def load_data(self):
        if self.dataset_name in GRAPH_TASKS:
            self.input_dim, self.output_dim, self.dataset= load4graph(self.dataset_name, self.shot_num)

    def node_degree_as_features(self, data_list):
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

    def Train(self, train_loader):

        self.gnn.train()
        total_loss = 0.0 
        for batch in train_loader:  
            self.optimizer.zero_grad() 
            batch = batch.to(self.device)
            out = self.gnn(batch.x, batch.edge_index, batch.batch)
            out = self.answering(out)
            loss = self.criterion(out, batch.y)  

            
            loss.backward()  
            self.optimizer.step()  
            total_loss += loss.item()  
        return total_loss / len(train_loader)  
        
    def AllInOneTrain(self, train_loader, answer_epoch=1, prompt_epoch=1):
        #we update answering and prompt alternately.
        
        # answer_epoch = 1  # 50
        # prompt_epoch = 1  # 50
        # answer_epoch = 5  # 50  #PROTEINS # COX2
        # prompt_epoch = 1  # 50
        
        # tune task head
        self.answering.train()
        self.prompt.eval()
        for epoch in range(1, answer_epoch + 1):
            answer_loss = self.prompt.Tune(train_loader, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)
            train_info("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f}".format(epoch, answer_epoch, answer_loss))

        self.answering.eval()
        self.prompt.train()
        for epoch in range(1, prompt_epoch + 1):
            pg_loss = self.prompt.Tune( train_loader,  self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
            train_info("frozen gnn | *tune prompt | frozen answering function... {}/{} ,loss: {:.4f}".format(epoch, prompt_epoch, pg_loss))
        
        return pg_loss

    def GPFTrain(self, train_loader):
        self.prompt.train()
        total_loss = 0.0 
        for batch in train_loader:  
            self.optimizer.zero_grad() 
            batch = batch.to(self.device)
            batch.x = self.prompt.add(batch.x)
            out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = self.prompt_type)
            out = self.answering(out)
            loss = self.criterion(out, batch.y)  
            loss.backward()  
            self.optimizer.step()  
            total_loss += loss.item()  
        return total_loss / len(train_loader)  

    def GpromptTrain(self, train_loader):
        self.prompt.train()
        total_loss = 0.0
        accumulated_centers = None
        accumulated_counts = None

        for batch in train_loader:
            
            # archived code for complete prototype embeddings of each labels. Not as well as batch version
            # # compute the prototype embeddings of each type of label

            self.pg_opi.zero_grad() 
            batch = batch.to(self.device)
            out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = 'Gprompt')
            # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
            center, class_counts = center_embedding(out,batch.y, self.output_dim)
            # 累积中心向量和样本数
            if accumulated_centers is None:
                accumulated_centers = center
                accumulated_counts = class_counts
            else:
                accumulated_centers += center * class_counts
                accumulated_counts += class_counts
            criterion = Gprompt_tuning_loss()
            loss = criterion(out, center, batch.y)  
            loss.backward()  
            self.pg_opi.step()  
            total_loss += loss.item()
            # 计算加权平均中心向量
            mean_centers = accumulated_centers / accumulated_counts

            return total_loss / len(train_loader), mean_centers

    def GPPTtrain(self, train_loader):
        self.prompt.train()
        for batch in train_loader:
            temp_loss=torch.tensor(0.0,requires_grad=True).to(self.device)
            graph_list = batch.to_data_list()        
            for index, graph in enumerate(graph_list):
                graph=graph.to(self.device)              
                node_embedding = self.gnn(graph.x,graph.edge_index)
                out = self.prompt(node_embedding, graph.edge_index) # gppt下游在1-shot的时候，prompt结果为nan
                loss = self.criterion(out, torch.full((1,graph.x.shape[0]), safe_graph_label(graph.y)).reshape(-1).to(self.device))
                temp_loss += loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())           
            temp_loss = temp_loss/(index+1)
            self.pg_opi.zero_grad()
            temp_loss.backward()
            self.pg_opi.step()
            self.prompt.update_StructureToken_weight(self.prompt.get_mid_h())
        return temp_loss.item()

    def _get_graph_embs_multigprompt(self, data_list):
        """Get graph-level embeddings from Preprompt (GraphPrePrompt) for a list of graphs."""
        from torch_geometric.data import Batch
        from torch_geometric.nn import global_mean_pool
        self.Preprompt.eval()
        embs_list, embs1_list = [], []
        with torch.no_grad():
            loader = DataLoader(data_list, batch_size=32, shuffle=False)
            for batch in loader:
                features, adj_scipy = process.process_tu(batch, self.output_dim, self.input_dim)
                features = torch.FloatTensor(features).to(self.device)
                adj = process.normalize_adj(adj_scipy + sp.eye(adj_scipy.shape[0]))
                adj_dense = torch.FloatTensor(adj.todense()).to(self.device)
                h, _ = self.Preprompt.embed(features.unsqueeze(0), adj_dense.unsqueeze(0), False, None, False)
                h_node = h[0] if h.dim() == 3 else h
                batch_batch = batch.batch.to(self.device)
                graph_emb = global_mean_pool(h_node, batch_batch)
                embs_list.append(graph_emb)
                embs1_list.append(graph_emb)
        return torch.cat(embs_list, 0), torch.cat(embs1_list, 0)

    def GraphMultiGpromptTrain(self, train_embs, train_embs1, train_lbls):
        """Train DownPrompt for graph-level MultiGprompt (few-shot)."""
        self.DownPrompt.train()
        self.optimizer.zero_grad()
        logits = self.DownPrompt(train_embs, train_embs1, train_lbls, 1).float().to(self.device)
        loss = self.criterion(logits, train_lbls)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item()

    def run(self):
        test_accs = []
        f1s = []
        rocs = []
        prcs = []
        batch_best_loss = []
        if self.prompt_type == 'All-in-one':
            # self.answer_epoch = 5 MUTAG Graph MAE / GraphCL
            # self.prompt_epoch = 1
            self.answer_epoch = 50
            self.prompt_epoch = 50
            self.epochs = int(self.epochs/self.answer_epoch)
        if self.shot_num > 0:
            task_pbar = tqdm(range(1, 6), desc="下游任务 Task", unit="task")
            for i in task_pbar:
                sample_dir = os.path.join(get_sample_data_graph_dir(self.dataset_name), "{}_shot".format(self.shot_num), str(i))
                idx_train = torch.load(os.path.join(sample_dir, "train_idx.pt")).type(torch.long).to(self.device)
                train_info("task {} idx_train len {}".format(i, len(idx_train)))
                train_lbls = torch.load(os.path.join(sample_dir, "train_labels.pt")).type(torch.long).squeeze().to(self.device)
                train_info("task {} train_labels len {}".format(i, len(train_lbls)))
                idx_test = torch.load(os.path.join(sample_dir, "test_idx.pt")).type(torch.long).to(self.device)
                test_lbls = torch.load(os.path.join(sample_dir, "test_labels.pt")).type(torch.long).squeeze().to(self.device)
                valid_path = os.path.join(sample_dir, "valid_idx.pt")
                idx_valid = torch.load(valid_path).type(torch.long) if os.path.exists(valid_path) else None
            
                train_dataset = self.dataset[idx_train.tolist()]
                test_dataset = self.dataset[idx_test.tolist()]
                valid_dataset = self.dataset[idx_valid.tolist()] if idx_valid is not None else None

                if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa']:
                    from torch_geometric.data import Batch
                    train_dataset = [train_g for train_g in train_dataset]
                    test_dataset = [test_g for test_g in test_dataset]
                    self.node_degree_as_features(train_dataset)
                    self.node_degree_as_features(test_dataset)
                    if self.prompt_type == 'GPPT':
                        processed_dataset = [g for g in self.dataset]
                        self.node_degree_as_features(processed_dataset)
                        processed_dataset = Batch.from_data_list([g for g in processed_dataset])
                    self.input_dim = train_dataset[0].x.size(1)

                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False) if valid_dataset is not None else None
                train_info("prepare data is finished!")
    
                patience = self.patience
                eval_every = getattr(self, 'eval_every', 1)
                higher_is_better = getattr(self, '_higher_is_better', True)
                best_val_metric = -1e9 if higher_is_better else 1e9
                best_epoch = 0
                best_valid_metrics = {}
                cnt_wait = 0
                best_center = None
                ckpt_path = os.path.join(self.checkpoint_dir, f"{self.dataset_name}_GraphTask_{self.prompt_type}_task{i}_best.pt") if self.save_best else None
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                
                if self.prompt_type == 'GPPT':
                    # initialize the GPPT hyperparametes via graph data
                    if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa']:
                        # total_num_nodes = sum([data.num_nodes for data in train_dataset])
                        # train_node_ids = torch.arange(0,total_num_nodes).squeeze().to(self.device)
                        # self.gppt_loader = DataLoader(processed_dataset, batch_size=1, shuffle=True)
                        # for i, batch in enumerate(self.gppt_loader):
                        #     if(i==0):
                        #         node_for_graph_labels = torch.full((1,batch.x.shape[0]), batch.y.item())
                        #     else:                   
                        #         node_for_graph_labels = torch.concat([node_for_graph_labels,torch.full((1,batch.x.shape[0]), batch.y.item())],dim=1)
                        
                        # node_embedding = self.gnn(processed_dataset.x.to(self.device), processed_dataset.edge_index.to(self.device))
                        # node_for_graph_labels=node_for_graph_labels.reshape((-1)).to(self.device)             
                        # self.prompt.weigth_init(node_embedding,processed_dataset.edge_index.to(self.device), node_for_graph_labels, train_node_ids)

                        # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

                        total_num_nodes = sum([data.num_nodes for data in train_dataset])
                        train_node_ids = torch.arange(0,total_num_nodes).squeeze().to(self.device)
                        self.gppt_loader = DataLoader(processed_dataset.to_data_list(), batch_size=1, shuffle=False)
                        for i, batch in enumerate(self.gppt_loader):
                            if(i==0):
                                node_for_graph_labels = torch.full((1,batch.x.shape[0]), safe_graph_label(batch.y))
                                node_embedding = self.gnn(batch.x.to(self.device), batch.edge_index.to(self.device))
                            else:                   
                                node_for_graph_labels = torch.concat([node_for_graph_labels,torch.full((1,batch.x.shape[0]), safe_graph_label(batch.y))],dim=1)
                                node_embedding = torch.concat([node_embedding,self.gnn(batch.x.to(self.device), batch.edge_index.to(self.device))],dim=0)
                        
                        node_for_graph_labels=node_for_graph_labels.reshape((-1)).to(self.device)
                        self.prompt.weigth_init(node_embedding,processed_dataset.edge_index.to(self.device), node_for_graph_labels, train_node_ids)

                        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)                    
                    else:
                        train_node_ids = torch.arange(0,train_dataset.x.shape[0]).squeeze().to(self.device)
                        # 将子图的节点id转换为全图的节点id
                        iterate_id_num = 0
                        for index, g in enumerate(train_dataset):
                            current_node_ids = iterate_id_num+torch.arange(0,g.x.shape[0]).squeeze().to(self.device)
                            iterate_id_num += g.x.shape[0]
                            previous_node_num = sum([self.dataset[i].x.shape[0] for i in range(idx_train[index]-1)])
                            train_node_ids[current_node_ids] += previous_node_num

                        self.gppt_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
                        for i, batch in enumerate(self.gppt_loader):
                            if(i==0):
                                node_for_graph_labels = torch.full((1,batch.x.shape[0]), safe_graph_label(batch.y))
                            else:                   
                                node_for_graph_labels = torch.concat([node_for_graph_labels,torch.full((1,batch.x.shape[0]), safe_graph_label(batch.y))],dim=1)
                        
                        node_embedding = self.gnn(self.dataset.x.to(self.device), self.dataset.edge_index.to(self.device))
                        node_for_graph_labels=node_for_graph_labels.reshape((-1)).to(self.device)
                        self.prompt.weigth_init(node_embedding,self.dataset.edge_index.to(self.device), node_for_graph_labels, train_node_ids)

                        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

                    # from torch_geometric.nn import global_mean_pool
                    # self.gppt_pool = global_mean_pool
                    # train_ids = torch.nonzero(idx_train, as_tuple=False).squeeze()
                    # self.gppt_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)          
                    # for i, batch in enumerate(self.gppt_loader):
                    #     batch.to(self.device)
                    #     node_embedding = self.gnn(batch.x, batch.edge_index)
                    #     if(i==0):
                    #         graph_embedding = self.gppt_pool(node_embedding,batch.batch.long())
                    #     else:
                    #         graph_embedding = torch.concat([graph_embedding,self.gppt_pool(node_embedding,batch.batch.long())],dim=0)
                
                if self.prompt_type == 'MultiGprompt':
                    train_embs, train_embs1 = self._get_graph_embs_multigprompt(train_dataset)
                    test_embs, test_embs1 = self._get_graph_embs_multigprompt(test_dataset)
                    valid_embs, valid_embs1 = self._get_graph_embs_multigprompt(valid_dataset) if valid_dataset is not None else (None, None)
                    train_lbls_mg = torch.tensor([safe_graph_label(g.y) for g in train_dataset], dtype=torch.long, device=self.device)
                    valid_lbls_mg = torch.tensor([safe_graph_label(g.y) for g in valid_dataset], dtype=torch.long, device=self.device) if valid_dataset is not None else None
                    test_lbls_mg = torch.tensor([safe_graph_label(g.y) for g in test_dataset], dtype=torch.long, device=self.device)

                task_pbar.set_postfix(task=i)
                for epoch in range(1, self.epochs + 1):
                    t0 = time.time()

                    if self.prompt_type == 'None':
                        loss = self.Train(train_loader)
                    elif self.prompt_type == 'All-in-one':
                        loss = self.AllInOneTrain(train_loader,self.answer_epoch,self.prompt_epoch)
                    elif self.prompt_type in ['GPF', 'GPF-plus']:
                        loss = self.GPFTrain(train_loader)
                    elif self.prompt_type =='Gprompt':
                        loss, center = self.GpromptTrain(train_loader)
                        best_center = center.detach()
                    elif self.prompt_type =='GPPT':
                        loss = self.GPPTtrain(train_loader)
                    elif self.prompt_type == 'MultiGprompt':
                        loss = self.GraphMultiGpromptTrain(train_embs, train_embs1, train_lbls_mg)
                    else:
                        raise ValueError("GraphTask: unsupported prompt_type '{}'".format(self.prompt_type))

                    epoch_training(epoch, time.time() - t0, loss)

                    if epoch % eval_every != 0:
                        continue

                    val_metric_raw = 0.0
                    val_metric = 0.0
                    va, vf1, vroc, vprc = 0.0, 0.0, 0.0, 0.0
                    if valid_loader is not None:
                        eval_loader = valid_loader
                        eval_st = time.time()
                        eval_fn = PromptRegistry.get_evaluator(self.prompt_type, 'GraphTask')
                        if eval_fn:
                            if self.prompt_type == 'MultiGprompt':
                                va, vf1, vroc, vprc = eval_fn(loader=eval_loader, data=None, idx=None, gnn=self.gnn, prompt=self.prompt, answering=self.answering, output_dim=self.output_dim, device=self.device,
                                                              valid_embs=valid_embs, valid_embs1=valid_embs1, valid_lbls=valid_lbls_mg, DownPrompt=self.DownPrompt)
                            else:
                                va, vf1, vroc, vprc = eval_fn(loader=eval_loader, data=None, idx=None, gnn=self.gnn, prompt=self.prompt, answering=self.answering, output_dim=self.output_dim, device=self.device, center=best_center)
                        elif self.prompt_type == 'None':
                            va, vf1, vroc, vprc = GNNGraphEva(eval_loader, self.gnn, self.answering, self.output_dim, self.device)
                        elif self.prompt_type =='GPPT':
                            va, vf1, vroc, vprc = GPPTGraphEva(eval_loader, self.gnn, self.prompt, self.output_dim, self.device)
                        elif self.prompt_type == 'All-in-one':
                            va, vf1, vroc, vprc = AllInOneEva(eval_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                        elif self.prompt_type in ['GPF', 'GPF-plus']:
                            va, vf1, vroc, vprc = GPFEva(eval_loader, self.gnn, self.prompt, self.answering, self.output_dim, self.device)
                        elif self.prompt_type =='Gprompt':
                            va, vf1, vroc, vprc = GpromptEva(eval_loader, self.gnn, self.prompt, best_center, self.output_dim, self.device)
                        elif self.prompt_type == 'MultiGprompt':
                            va, vf1, vroc, vprc = GraphMultiGpromptEva(valid_embs, valid_embs1, valid_lbls_mg, self.DownPrompt, self.output_dim, self.device)
                        val_metric = self.metric_from_eva(va, vf1, vroc, vprc)
                        val_metric_raw = val_metric
                        met_name = self.early_stopping_metric or 'valid_acc'
                        epoch_evaluating(epoch, time.time() - eval_st, val_metric_raw, met_name)
                        valid_result({"valid_acc": va, "valid_f1": vf1, "valid_auroc": vroc, "valid_auprc": vprc})

                    if valid_loader is not None:
                        improved = (val_metric > best_val_metric) if self._higher_is_better else (val_metric < best_val_metric)
                        if improved:
                            best_val_metric = val_metric
                            best_epoch = epoch
                            best_valid_metrics = {"valid_acc": va, "valid_f1": vf1, "valid_auroc": vroc, "valid_auprc": vprc}
                            cnt_wait = 0
                            if self.save_best and ckpt_path:
                                ckpt = {'gnn': self.gnn.state_dict(), 'answering': self.answering.state_dict()}
                                if self.prompt is not None:
                                    ckpt['prompt'] = self.prompt.state_dict()
                                if best_center is not None:
                                    ckpt['center'] = best_center.cpu()
                                if self.prompt_type == 'MultiGprompt':
                                    ckpt['DownPrompt'] = self.DownPrompt.state_dict()
                                    ckpt['feature_prompt'] = self.feature_prompt.state_dict()
                                torch.save(ckpt, ckpt_path)
                                train_info("Checkpoint saved (best)")
                        else:
                            cnt_wait += 1
                            if cnt_wait >= patience:
                                early_stopping_msg(epoch, patience, self.early_stopping_metric or 'valid_acc')
                                break
                import math
                if not math.isnan(loss):
                    batch_best_loss.append(loss)
                    if self.save_best and ckpt_path and os.path.exists(ckpt_path):
                        ckpt = torch.load(ckpt_path, map_location=self.device)
                        self.gnn.load_state_dict(ckpt['gnn'])
                        self.answering.load_state_dict(ckpt['answering'])
                        if 'prompt' in ckpt and self.prompt is not None:
                            self.prompt.load_state_dict(ckpt['prompt'])
                        if 'center' in ckpt and self.prompt_type == 'Gprompt':
                            best_center = ckpt['center'].to(self.device)
                        if self.prompt_type == 'MultiGprompt' and 'DownPrompt' in ckpt:
                            self.DownPrompt.load_state_dict(ckpt['DownPrompt'])
                            if 'feature_prompt' in ckpt:
                                self.feature_prompt.load_state_dict(ckpt['feature_prompt'])
                        model_loaded(ckpt_path)
                train_info('Begin to evaluate')
                
                eval_fn = PromptRegistry.get_evaluator(self.prompt_type, 'GraphTask')
                if eval_fn:
                    if self.prompt_type == 'MultiGprompt':
                        test_acc, f1, roc, prc = eval_fn(loader=test_loader, data=None, idx=None, gnn=self.gnn, prompt=self.prompt, answering=self.answering, output_dim=self.output_dim, device=self.device,
                                                        test_embs=test_embs, test_embs1=test_embs1, test_lbls=test_lbls_mg, DownPrompt=self.DownPrompt)
                    else:
                        test_acc, f1, roc, prc = eval_fn(loader=test_loader, data=None, idx=None, gnn=self.gnn, prompt=self.prompt, answering=self.answering, output_dim=self.output_dim, device=self.device, center=best_center)
                elif self.prompt_type == 'None':
                    test_acc, f1, roc, prc = GNNGraphEva(test_loader, self.gnn, self.answering, self.output_dim, self.device)
                elif self.prompt_type =='GPPT':
                    test_acc, f1, roc, prc = GPPTGraphEva(test_loader, self.gnn, self.prompt, self.output_dim, self.device)
                elif self.prompt_type == 'MultiGprompt':
                    test_acc, f1, roc, prc = GraphMultiGpromptEva(test_embs, test_embs1, test_lbls_mg, self.DownPrompt, self.output_dim, self.device)
                else:
                    test_acc, f1, roc, prc = GpromptEva(test_loader, self.gnn, self.prompt, best_center, self.output_dim, self.device)

                finished_training(best_epoch if best_epoch > 0 else epoch)
                if best_valid_metrics:
                    best_valid_ordered(best_valid_metrics)
                test_result_ordered({"test_acc": test_acc, "test_f1": f1, "test_auroc": roc, "test_auprc": prc})
                test_accs.append(test_acc)
                f1s.append(f1)
                rocs.append(roc)
                prcs.append(prc)
            
            mean_test_acc = np.mean(test_accs)
            std_test_acc = np.std(test_accs)    
            mean_f1 = np.mean(f1s)
            std_f1 = np.std(f1s)   
            mean_roc = np.mean(rocs)
            std_roc = np.std(rocs)   
            mean_prc = np.mean(prcs)
            std_prc = np.std(prcs) 
            train_info("Final best | test Accuracy {:.4f}±{:.4f}(std)".format(mean_test_acc, std_test_acc))
            train_info("Final best | test F1 {:.4f}±{:.4f}(std)".format(mean_f1, std_f1))
            train_info("Final best | AUROC {:.4f}±{:.4f}(std)".format(mean_roc, std_roc))
            train_info("Final best | AUPRC {:.4f}±{:.4f}(std)".format(mean_prc, std_prc))
            train_info("{} {} {} Graph Task completed".format(self.pre_train_type, self.gnn_type, self.prompt_type))
            mean_best = np.mean(batch_best_loss)

            return  mean_best, mean_test_acc, std_test_acc, mean_f1, std_f1, mean_roc, std_roc, mean_prc, std_prc

        

        
        else:
            train_dataset, valid_dataset, test_dataset = self.dataset
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa']:
                self.node_degree_as_features(train_dataset)
                self.node_degree_as_features(valid_dataset)
                self.node_degree_as_features(test_dataset)
            train_info("prepare data is finished!")

            patience = self.patience
            eval_every = getattr(self, 'eval_every', 1)
            higher_is_better = getattr(self, '_higher_is_better', True)
            best_val_metric = -1e9 if higher_is_better else 1e9
            best_epoch = 0
            best_valid_metrics = {}
            cnt_wait = 0
            best_center = None
            ckpt_path = os.path.join(self.checkpoint_dir, f"{self.dataset_name}_GraphTask_{self.prompt_type}_best.pt") if self.save_best else None
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            full_list = train_dataset + valid_dataset + test_dataset
            processed_dataset = full_list if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa'] else None
            if processed_dataset is not None:
                from torch_geometric.data import Batch
                processed_dataset = [g for g in processed_dataset]
                self.node_degree_as_features(processed_dataset)
                processed_dataset = Batch.from_data_list(processed_dataset)
        
            if self.prompt_type == 'All-in-one':
                # self.answer_epoch = 5 MUTAG Graph MAE / GraphCL
                # self.prompt_epoch = 1
                self.answer_epoch = 5
                self.prompt_epoch = 1
                self.epochs = int(self.epochs/self.answer_epoch)
                
            elif self.prompt_type == 'GPPT':
                if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa']:
                    # total_num_nodes = sum([data.num_nodes for data in train_dataset])
                    # train_node_ids = torch.arange(0,total_num_nodes).squeeze().to(self.device)
                    # self.gppt_loader = DataLoader(processed_dataset, batch_size=1, shuffle=True)
                    # for i, batch in enumerate(self.gppt_loader):
                    #     if(i==0):
                    #         node_for_graph_labels = torch.full((1,batch.x.shape[0]), batch.y.item())
                    #     else:                   
                    #         node_for_graph_labels = torch.concat([node_for_graph_labels,torch.full((1,batch.x.shape[0]), batch.y.item())],dim=1)
                    
                    # node_embedding = self.gnn(processed_dataset.x.to(self.device), processed_dataset.edge_index.to(self.device))
                    # node_for_graph_labels=node_for_graph_labels.reshape((-1)).to(self.device)             
                    # self.prompt.weigth_init(node_embedding,processed_dataset.edge_index.to(self.device), node_for_graph_labels, train_node_ids)

                    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

                    total_num_nodes = sum([data.num_nodes for data in train_dataset])
                    train_node_ids = torch.arange(0,total_num_nodes).squeeze().to(self.device)
                    self.gppt_loader = DataLoader(processed_dataset.to_data_list(), batch_size=1, shuffle=False)
                    for i, batch in enumerate(self.gppt_loader):
                        if(i==0):
                            node_for_graph_labels = torch.full((1,batch.x.shape[0]), safe_graph_label(batch.y))
                            node_embedding = self.gnn(batch.x.to(self.device), batch.edge_index.to(self.device))
                        else:                   
                            node_for_graph_labels = torch.concat([node_for_graph_labels,torch.full((1,batch.x.shape[0]), safe_graph_label(batch.y))],dim=1)
                            node_embedding = torch.concat([node_embedding,self.gnn(batch.x.to(self.device), batch.edge_index.to(self.device))],dim=0)
                    
                    node_for_graph_labels=node_for_graph_labels.reshape((-1)).to(self.device)
                    self.prompt.weigth_init(node_embedding,processed_dataset.edge_index.to(self.device), node_for_graph_labels, train_node_ids)

                    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)                    
                else:
                    from torch_geometric.data import Batch
                    batched_full = Batch.from_data_list(full_list)
                    n_train_nodes = sum(g.num_nodes for g in train_dataset)
                    train_node_ids = torch.arange(0, n_train_nodes).squeeze().to(self.device)
                    self.gppt_loader = DataLoader(full_list, batch_size=1, shuffle=True)
                    node_emb_list = []
                    node_for_graph_labels = []
                    for batch in self.gppt_loader:
                        batch = batch.to(self.device)
                        node_emb_list.append(self.gnn(batch.x, batch.edge_index))
                        node_for_graph_labels.append(torch.full((batch.x.shape[0],), safe_graph_label(batch.y), device=self.device))
                    node_embedding = torch.cat(node_emb_list, dim=0)
                    node_for_graph_labels = torch.cat(node_for_graph_labels)
                    self.prompt.weigth_init(node_embedding, batched_full.edge_index.to(self.device), node_for_graph_labels, train_node_ids)
                    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

                # from torch_geometric.nn import global_mean_pool
                # self.gppt_pool = global_mean_pool
                # train_ids = torch.nonzero(idx_train, as_tuple=False).squeeze()
                # self.gppt_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)              
                # for i, batch in enumerate(self.gppt_loader):
                #     batch.to(self.device)
                #     node_embedding = self.gnn(batch.x, batch.edge_index)
                #     if(i==0):
                #         graph_embedding = self.gppt_pool(node_embedding,batch.batch.long())
                #     else:
                #         graph_embedding = torch.concat([graph_embedding,self.gppt_pool(node_embedding,batch.batch.long())],dim=0)
                

            if self.prompt_type == 'MultiGprompt':
                train_embs, train_embs1 = self._get_graph_embs_multigprompt(train_dataset)
                test_embs, test_embs1 = self._get_graph_embs_multigprompt(test_dataset)
                valid_embs, valid_embs1 = self._get_graph_embs_multigprompt(valid_dataset)
                train_lbls_mg = torch.tensor([safe_graph_label(g.y) for g in train_dataset], dtype=torch.long, device=self.device)
                valid_lbls_mg = torch.tensor([safe_graph_label(g.y) for g in valid_dataset], dtype=torch.long, device=self.device)
                test_lbls_mg = torch.tensor([safe_graph_label(g.y) for g in test_dataset], dtype=torch.long, device=self.device)

            for epoch in range(1, self.epochs + 1):
                t0 = time.time()

                if self.prompt_type == 'None':
                    loss = self.Train(train_loader)
                elif self.prompt_type == 'All-in-one':
                    loss = self.AllInOneTrain(train_loader,self.answer_epoch,self.prompt_epoch)
                elif self.prompt_type in ['GPF', 'GPF-plus']:
                    loss = self.GPFTrain(train_loader)
                elif self.prompt_type =='Gprompt':
                    loss, center = self.GpromptTrain(train_loader)
                    best_center = center.detach()
                elif self.prompt_type =='GPPT':
                    loss = self.GPPTtrain(train_loader)
                elif self.prompt_type == 'MultiGprompt':
                    loss = self.GraphMultiGpromptTrain(train_embs, train_embs1, train_lbls_mg)
                else:
                    raise ValueError("GraphTask: unsupported prompt_type '{}'".format(self.prompt_type))

                epoch_training(epoch, time.time() - t0, loss)
                if epoch % eval_every != 0:
                    continue

                va, vf1, vroc, vprc = 0.0, 0.0, 0.0, 0.0
                eval_st = time.time()
                if self.prompt_type == 'None':
                    va, vf1, vroc, vprc = GNNGraphEva(valid_loader, self.gnn, self.answering, self.output_dim, self.device)
                elif self.prompt_type == 'GPPT':
                    va, vf1, vroc, vprc = GPPTGraphEva(valid_loader, self.gnn, self.prompt, self.output_dim, self.device)
                elif self.prompt_type == 'All-in-one':
                    va, vf1, vroc, vprc = AllInOneEva(valid_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                elif self.prompt_type in ['GPF', 'GPF-plus']:
                    va, vf1, vroc, vprc = GPFEva(valid_loader, self.gnn, self.prompt, self.answering, self.output_dim, self.device)
                elif self.prompt_type == 'Gprompt':
                    va, vf1, vroc, vprc = GpromptEva(valid_loader, self.gnn, self.prompt, best_center, self.output_dim, self.device)
                elif self.prompt_type == 'MultiGprompt':
                    va, vf1, vroc, vprc = GraphMultiGpromptEva(valid_embs, valid_embs1, valid_lbls_mg, self.DownPrompt, self.output_dim, self.device)
                val_metric = self.metric_from_eva(va, vf1, vroc, vprc)
                met_name = self.early_stopping_metric or 'valid_acc'
                epoch_evaluating(epoch, time.time() - eval_st, val_metric, met_name)
                valid_result({"valid_acc": va, "valid_f1": vf1, "valid_auroc": vroc, "valid_auprc": vprc})

                improved = (val_metric > best_val_metric) if higher_is_better else (val_metric < best_val_metric)
                if improved:
                    best_val_metric = val_metric
                    best_epoch = epoch
                    best_valid_metrics = {"valid_acc": va, "valid_f1": vf1, "valid_auroc": vroc, "valid_auprc": vprc}
                    cnt_wait = 0
                    if self.save_best and ckpt_path:
                        ckpt = {'gnn': self.gnn.state_dict(), 'answering': self.answering.state_dict()}
                        if self.prompt is not None:
                            ckpt['prompt'] = self.prompt.state_dict()
                        if best_center is not None:
                            ckpt['center'] = best_center.cpu()
                        if self.prompt_type == 'MultiGprompt':
                            ckpt['DownPrompt'] = self.DownPrompt.state_dict()
                            ckpt['feature_prompt'] = self.feature_prompt.state_dict()
                        torch.save(ckpt, ckpt_path)
                        train_info("Checkpoint saved (best)")
                else:
                    cnt_wait += 1
                    if cnt_wait >= patience:
                        early_stopping_msg(epoch, patience, self.early_stopping_metric or 'valid_acc')
                        break

            if self.save_best and ckpt_path and os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.gnn.load_state_dict(ckpt['gnn'])
                self.answering.load_state_dict(ckpt['answering'])
                if 'prompt' in ckpt and self.prompt is not None:
                    self.prompt.load_state_dict(ckpt['prompt'])
                if 'center' in ckpt and self.prompt_type == 'Gprompt':
                    best_center = ckpt['center'].to(self.device)
                if self.prompt_type == 'MultiGprompt' and 'DownPrompt' in ckpt:
                    self.DownPrompt.load_state_dict(ckpt['DownPrompt'])
                    if 'feature_prompt' in ckpt:
                        self.feature_prompt.load_state_dict(ckpt['feature_prompt'])
                model_loaded(ckpt_path)

            train_info('Begin to evaluate')
            
            eval_fn = PromptRegistry.get_evaluator(self.prompt_type, 'GraphTask')
            if eval_fn:
                if self.prompt_type == 'MultiGprompt':
                    test_acc, f1, roc, prc = eval_fn(loader=test_loader, data=None, idx=None, gnn=self.gnn, prompt=self.prompt, answering=self.answering, output_dim=self.output_dim, device=self.device,
                                                    test_embs=test_embs, test_embs1=test_embs1, test_lbls=test_lbls_mg, DownPrompt=self.DownPrompt)
                else:
                    test_acc, f1, roc, prc = eval_fn(loader=test_loader, data=None, idx=None, gnn=self.gnn, prompt=self.prompt, answering=self.answering, output_dim=self.output_dim, device=self.device, center=best_center)
            elif self.prompt_type == 'None':
                test_acc, f1, roc, prc = GNNGraphEva(test_loader, self.gnn, self.answering, self.output_dim, self.device)
            elif self.prompt_type =='GPPT':
                test_acc, f1, roc, prc = GPPTGraphEva(test_loader, self.gnn, self.prompt, self.output_dim, self.device)
            elif self.prompt_type == 'MultiGprompt':
                test_acc, f1, roc, prc = GraphMultiGpromptEva(test_embs, test_embs1, test_lbls_mg, self.DownPrompt, self.output_dim, self.device)
            else:
                test_acc, f1, roc, prc = GpromptEva(test_loader, self.gnn, self.prompt, best_center, self.output_dim, self.device)

            finished_training(best_epoch if best_epoch > 0 else self.epochs)
            if best_valid_metrics:
                best_valid_ordered(best_valid_metrics)
            test_result_ordered({"test_acc": test_acc, "test_f1": f1, "test_auroc": roc, "test_auprc": prc})
            train_info("{} {} {} Graph Task completed".format(self.pre_train_type, self.gnn_type, self.prompt_type))


            return  test_acc,f1,roc,prc
