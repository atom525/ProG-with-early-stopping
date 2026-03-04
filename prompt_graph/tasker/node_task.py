import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from prompt_graph.utils import constraint,  center_embedding, Gprompt_tuning_loss
from prompt_graph.evaluation import GPPTEva, GNNNodeEva, GPFEva, MultiGpromptEva
from prompt_graph.pretrain import GraphPrePrompt, NodePrePrompt, prompt_pretrain_sample
from .task import BaseTask
from prompt_graph.utils.train_logger import train_info, epoch_training, epoch_evaluating, valid_result, best_valid_ordered, test_result_ordered, finished_training, early_stopping_msg, model_loaded
import time
import warnings
import numpy as np
from prompt_graph.data import load4node, graph_split, split_induced_graphs, node_sample_and_save, GraphDataset
from prompt_graph.evaluation import GpromptEva, AllInOneEva
from prompt_graph.registry import PromptRegistry
import pickle
import os
from prompt_graph.utils import process
from prompt_graph.utils.paths import get_sample_data_node_dir, get_induced_graph_path
warnings.filterwarnings("ignore")

class NodeTask(BaseTask):
      def __init__(self, data, input_dim, output_dim, task_num = 5, graphs_list = None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.task_type = 'NodeTask'
            self.task_num = task_num  # 增加task_num的参数，控制重复数量，默认为5
            if self.prompt_type == 'MultiGprompt':
                  self.load_multigprompt_data()
            else:
                  self.data = data
                  if self.dataset_name == 'ogbn-arxiv':
                        self.data.y = self.data.y.squeeze()
                  self.input_dim = input_dim
                  self.output_dim = output_dim
                  self.graphs_list = graphs_list
            
            self.create_few_data_folder()

      def create_few_data_folder(self):
            k = self.shot_num
            task_num = self.task_num
            base = get_sample_data_node_dir(self.dataset_name)
            for k in range(1, task_num+1):
                  k_shot_folder = os.path.join(base, str(k) + '_shot')
                  os.makedirs(k_shot_folder, exist_ok=True)
                  for i in range(1, task_num+1):
                        folder = os.path.join(k_shot_folder, str(i))
                        if not os.path.exists(folder):
                              os.makedirs(folder)
                              node_sample_and_save(self.data, k, folder, self.output_dim, self.split_ratio)
                              train_info("{} shot {} th saved".format(k, i))

      def load_multigprompt_data(self):
            adj, features, labels = process.load_data(self.dataset_name)
            # adj, features, labels = process.load_data(self.dataset_name)  
            self.input_dim = features.shape[1]
            self.output_dim = labels.shape[1]
            train_info("MultiGprompt output_dim {}".format(self.output_dim))
            features, _ = process.preprocess_features(features)
            self.sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj).to(self.device)
            self.labels = torch.FloatTensor(labels[np.newaxis])
            self.features = torch.FloatTensor(features[np.newaxis]).to(self.device)
            # print("labels",labels)
            train_info("adj {} feature {}".format(self.sp_adj.shape, features.shape))

      def load_induced_graph(self):
            smallest_size = 5  # 默认为5
            if self.dataset_name in ['ENZYMES', 'PROTEINS']:
                  smallest_size = 1
            if self.dataset_name == 'PubMed':
                  smallest_size = 8
            file_path = get_induced_graph_path(self.dataset_name, min_size=smallest_size, max_size=300)
            if os.path.exists(file_path):
                  with open(file_path, 'rb') as f:
                        graphs_list = pickle.load(f)
            else:
                  train_info("Begin split_induced_graphs")
                  split_induced_graphs(self.data, os.path.dirname(file_path), self.device, smallest_size=smallest_size, largest_size=300)
                  with open(file_path, 'rb') as f:
                        graphs_list = pickle.load(f)
            self.graphs_list = []
            for i in range(len(graphs_list)):
                  graph = graphs_list[i].to(self.device)
                  self.graphs_list.append(graph)
            

      
      def load_data(self):
            self.data, self.input_dim, self.output_dim = load4node(self.dataset_name)

      def train(self, data, train_idx):
            self.gnn.train()
            self.answering.train()
            self.optimizer.zero_grad() 
            out = self.gnn(data.x, data.edge_index, batch=None) 
            out = self.answering(out)
            loss = self.criterion(out[train_idx], data.y[train_idx])
            loss.backward()  
            self.optimizer.step()  
            return loss.item()
      
      def GPPTtrain(self, data, train_idx):
            self.prompt.train()
            node_embedding = self.gnn(data.x, data.edge_index)
            out = self.prompt(node_embedding, data.edge_index)
            loss = self.criterion(out[train_idx], data.y[train_idx])
            loss = loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())
            self.pg_opi.zero_grad()
            loss.backward()
            self.pg_opi.step()
            mid_h = self.prompt.get_mid_h()
            self.prompt.update_StructureToken_weight(mid_h)
            return loss.item()
      
      def MultiGpromptTrain(self, pretrain_embs, train_lbls, train_idx):
            self.DownPrompt.train()
            self.optimizer.zero_grad()
            prompt_feature = self.feature_prompt(self.features)
            # prompt_feature = self.feature_prompt(self.data.x)
            # embeds1 = self.gnn(prompt_feature, self.data.edge_index)
            embeds1= self.Preprompt.gcn(prompt_feature, self.sp_adj , True, False)
            pretrain_embs1 = embeds1[0, train_idx]
            logits = self.DownPrompt(pretrain_embs,pretrain_embs1, train_lbls,1).float().to(self.device)
            loss = self.criterion(logits, train_lbls)           
            loss.backward(retain_graph=True)
            self.optimizer.step()
            return loss.item()
      
      def SUPTtrain(self, data):
            self.gnn.train()
            self.optimizer.zero_grad() 
            data.x = self.prompt.add(data.x)
            out = self.gnn(data.x, data.edge_index, batch=None) 
            out = self.answering(out)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])  
            orth_loss = self.prompt.orthogonal_loss()
            loss += orth_loss
            loss.backward()  
            self.optimizer.step()  
            return loss
      
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

      def AllInOneTrain(self, train_loader, answer_epoch=1, prompt_epoch=1):
            #we update answering and prompt alternately.
            # tune task head
            self.answering.train()
            self.prompt.eval()
            self.gnn.eval()
            for epoch in range(1, answer_epoch + 1):
                  answer_loss = self.prompt.Tune(train_loader, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)
                  train_info("frozen gnn | frozen prompt | *tune answering {}/{} loss: {:.4f}".format(epoch, answer_epoch, answer_loss))

            # tune prompt
            self.answering.eval()
            self.prompt.train()
            for epoch in range(1, prompt_epoch + 1):
                  pg_loss = self.prompt.Tune( train_loader,  self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
                  train_info("frozen gnn | *tune prompt | frozen answering {}/{} loss: {:.4f}".format(epoch, prompt_epoch, pg_loss))
            
            # return pg_loss
            return answer_loss
      
      def GpromptTrain(self, train_loader):
            self.prompt.train()
            total_loss = 0.0 
            accumulated_centers = None
            accumulated_counts = None
            for batch in train_loader:  
                  self.pg_opi.zero_grad() 
                  batch = batch.to(self.device)
                  out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = 'Gprompt')
                  # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
                  center, class_counts = center_embedding(out, batch.y, self.output_dim)
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
      
      def run(self):
            test_accs = []
            f1s = []
            rocs = []
            prcs = []
            batch_best_loss = []
            if self.prompt_type == 'All-in-one':
                  self.answer_epoch = 50
                  self.prompt_epoch = 50
                  self.epochs = int(self.epochs/self.answer_epoch)
            task_pbar = tqdm(range(1, self.task_num+1), desc="下游任务 Task", unit="task")
            for i in task_pbar:
                  task_pbar.set_postfix(task=i)
                  sample_data_foler_path = os.path.join(get_sample_data_node_dir(self.dataset_name), "{}_shot".format(self.shot_num), str(i))

                  if not os.path.exists(sample_data_foler_path):
                        train_info("Warning: Failed to find sample_data shot {} id {} path {} skipping".format(self.shot_num, i, sample_data_foler_path))
                        continue


                  self.initialize_gnn()
                  self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                                torch.nn.Softmax(dim=1)).to(self.device) 
                  self.initialize_prompt()
                  self.initialize_optimizer()



                  idx_train = torch.load(f"{sample_data_foler_path}/train_idx.pt").type(torch.long).to(self.device)
                  train_info("task {} idx_train len {}".format(i, len(idx_train)))
                  train_lbls = torch.load(f"{sample_data_foler_path}/train_labels.pt").type(torch.long).squeeze().to(self.device)
                  train_info("task {} train_labels len {}".format(i, len(train_lbls)))
                  idx_test = torch.load(f"{sample_data_foler_path}/test_idx.pt").type(torch.long).to(self.device)
                  test_lbls = torch.load(f"{sample_data_foler_path}/test_labels.pt").type(torch.long).squeeze().to(self.device)
                  valid_path = f"{sample_data_foler_path}/valid_idx.pt"
                  idx_valid = torch.load(valid_path).type(torch.long).to(self.device) if os.path.exists(valid_path) else None

                  # GPPT prompt initialtion
                  if self.prompt_type == 'GPPT':
                        node_embedding = self.gnn(self.data.x, self.data.edge_index)
                        self.prompt.weigth_init(node_embedding,self.data.edge_index, self.data.y, idx_train)

                  valid_loader = None
                  if PromptRegistry.needs_induced_graph(self.prompt_type):
                        train_graphs = []
                        test_graphs = []
                        valid_graphs = []
                        idx_valid_set = set(idx_valid.tolist()) if idx_valid is not None else set()
                        train_info("distinguishing train/valid/test dataset")
                        for graph in self.graphs_list:
                              if graph.index in idx_train:
                                    train_graphs.append(graph)
                              elif graph.index in idx_test:
                                    test_graphs.append(graph)
                              elif graph.index in idx_valid_set:
                                    valid_graphs.append(graph)
                        train_info("prepare induced graph train {} valid {} test {}".format(len(train_graphs), len(valid_graphs), len(test_graphs)))

                        train_dataset = GraphDataset(train_graphs)
                        test_dataset = GraphDataset(test_graphs)
                        valid_dataset = GraphDataset(valid_graphs) if valid_graphs else None

                        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False) if valid_dataset else None

                  if self.prompt_type == 'MultiGprompt':
                        embeds, _ = self.Preprompt.embed(self.features, self.sp_adj, True, None, False)
                        pretrain_embs = embeds[0, idx_train]
                        test_embs = embeds[0, idx_test]

                  patience = self.patience
                  eval_every = getattr(self, 'eval_every', 1)
                  higher_is_better = getattr(self, '_higher_is_better', True)
                  best_val_metric = -1e9 if higher_is_better else 1e9
                  best_epoch = 0
                  best_valid_metrics = {}
                  cnt_wait = 0
                  best_center = None
                  va, vf1, vroc, vprc = 0.0, 0.0, 0.0, 0.0
                  ckpt_path = os.path.join(self.checkpoint_dir, f"{self.dataset_name}_NodeTask_{self.prompt_type}_task{i}_best.pt") if self.save_best else None
                  os.makedirs(self.checkpoint_dir, exist_ok=True)

                  for epoch in range(1, self.epochs + 1):
                        t0 = time.time()

                        if self.prompt_type == 'None':
                              loss = self.train(self.data, idx_train)                             
                        elif self.prompt_type == 'GPPT':
                              loss = self.GPPTtrain(self.data, idx_train)                
                        elif self.prompt_type == 'All-in-one':
                              loss = self.AllInOneTrain(train_loader,self.answer_epoch,self.prompt_epoch)                           
                        elif self.prompt_type in ['GPF', 'GPF-plus']:
                              loss = self.GPFTrain(train_loader)                                                          
                        elif self.prompt_type =='Gprompt':
                              loss, center = self.GpromptTrain(train_loader)
                              best_center = center.detach()
                        elif self.prompt_type == 'MultiGprompt':
                              loss = self.MultiGpromptTrain(pretrain_embs, train_lbls, idx_train)
                              center = None
                        else:
                              raise ValueError("NodeTask: unsupported prompt_type '{}'".format(self.prompt_type))

                        epoch_training(epoch, time.time() - t0, loss)

                        if epoch % eval_every != 0:
                              continue

                        val_metric_raw = 0.0
                        val_metric = 0.0
                        has_valid = False
                        eval_fn = PromptRegistry.get_evaluator(self.prompt_type, 'NodeTask')
                        if idx_valid is not None and (self.prompt_type in ['None', 'GPPT'] or (eval_fn and valid_loader is None)):
                              has_valid = True
                              eval_st = time.time()
                              if eval_fn:
                                    va, vf1, vroc, vprc = eval_fn(loader=None, data=self.data, idx=idx_valid, gnn=self.gnn, prompt=self.prompt, answering=self.answering, output_dim=self.output_dim, device=self.device)
                              elif self.prompt_type == 'None':
                                    va, vf1, vroc, vprc = GNNNodeEva(self.data, idx_valid, self.gnn, self.answering, self.output_dim, self.device)
                              else:
                                    va, vf1, vroc, vprc = GPPTEva(self.data, idx_valid, self.gnn, self.prompt, self.output_dim, self.device)
                              val_metric = self.metric_from_eva(va, vf1, vroc, vprc)
                              val_metric_raw = val_metric
                              met_name = self.early_stopping_metric or 'valid_acc'
                              epoch_evaluating(epoch, time.time() - eval_st, val_metric_raw, met_name)
                              valid_result({"valid_acc": va, "valid_f1": vf1, "valid_auroc": vroc, "valid_auprc": vprc})
                        elif valid_loader is not None and (self.prompt_type in ['GPF', 'GPF-plus', 'Gprompt', 'All-in-one'] or eval_fn):
                              has_valid = True
                              eval_st = time.time()
                              if eval_fn:
                                    va, vf1, vroc, vprc = eval_fn(loader=valid_loader, data=self.data, idx=None, gnn=self.gnn, prompt=self.prompt, answering=self.answering, output_dim=self.output_dim, device=self.device, center=best_center)
                              elif self.prompt_type == 'All-in-one':
                                    va, vf1, vroc, vprc = AllInOneEva(valid_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                              elif self.prompt_type in ['GPF', 'GPF-plus']:
                                    va, vf1, vroc, vprc = GPFEva(valid_loader, self.gnn, self.prompt, self.answering, self.output_dim, self.device)
                              else:
                                    va, vf1, vroc, vprc = GpromptEva(valid_loader, self.gnn, self.prompt, best_center, self.output_dim, self.device)
                              val_metric = self.metric_from_eva(va, vf1, vroc, vprc)
                              val_metric_raw = val_metric
                              met_name = self.early_stopping_metric or 'valid_acc'
                              epoch_evaluating(epoch, time.time() - eval_st, val_metric_raw, met_name)
                              valid_result({"valid_acc": va, "valid_f1": vf1, "valid_auroc": vroc, "valid_auprc": vprc})

                        if has_valid:
                            improved = (val_metric > best_val_metric) if self._higher_is_better else (val_metric < best_val_metric)
                            if improved:
                                best_val_metric = val_metric
                                best_epoch = epoch
                                best_valid_metrics = {"valid_acc": va, "valid_f1": vf1, "valid_auroc": vroc, "valid_auprc": vprc}
                                cnt_wait = 0
                                if self.save_best and ckpt_path:
                                    ckpt = {'gnn': self.gnn.state_dict(), 'answering': self.answering.state_dict()}
                                    if self.prompt_type != 'None' and self.prompt is not None:
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
                                    train_info("-" * 50)
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

                        eval_fn = PromptRegistry.get_evaluator(self.prompt_type, 'NodeTask')
                        if eval_fn:
                              if self.prompt_type == 'MultiGprompt':
                                    prompt_feature = self.feature_prompt(self.features)
                                    test_acc, f1, roc, prc = eval_fn(loader=None, data=None, idx=idx_test, gnn=self.gnn, prompt=self.prompt, answering=self.answering, output_dim=self.output_dim, device=self.device,
                                                                     test_embs=test_embs, test_lbls=test_lbls, idx_test=idx_test, prompt_feature=prompt_feature, Preprompt=self.Preprompt, DownPrompt=self.DownPrompt, sp_adj=self.sp_adj)
                              elif test_loader is not None:
                                    test_acc, f1, roc, prc = eval_fn(loader=test_loader, data=self.data, idx=None, gnn=self.gnn, prompt=self.prompt, answering=self.answering, output_dim=self.output_dim, device=self.device, center=best_center)
                              else:
                                    test_acc, f1, roc, prc = eval_fn(loader=None, data=self.data, idx=idx_test, gnn=self.gnn, prompt=self.prompt, answering=self.answering, output_dim=self.output_dim, device=self.device)
                        else:
                              raise ValueError("Prompt type '{}' has no registered evaluator. Register in registry_config.py.".format(self.prompt_type))

                        finished_training(epoch)
                        if best_valid_metrics:
                            best_valid_ordered(best_valid_metrics)
                        test_result_ordered({"test_acc": test_acc, "test_f1": f1, "test_auroc": roc, "test_auprc": prc})
                        train_info("best_loss {}".format(batch_best_loss))     
                                    
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
            train_info("Acc List {}".format(test_accs))
            train_info("Final best | test Accuracy {:.4f}±{:.4f}(std)".format(mean_test_acc, std_test_acc))
            train_info("Final best | test F1 {:.4f}±{:.4f}(std)".format(mean_f1, std_f1))
            train_info("Final best | AUROC {:.4f}±{:.4f}(std)".format(mean_roc, std_roc))
            train_info("Final best | AUPRC {:.4f}±{:.4f}(std)".format(mean_prc, std_prc))
            train_info("{} {} {} Node Task completed".format(self.pre_train_type, self.gnn_type, self.prompt_type))
            mean_best = np.mean(batch_best_loss)

            return  mean_best, mean_test_acc, std_test_acc, mean_f1, std_f1, mean_roc, std_roc, mean_prc, std_prc

                  
            # elif self.prompt_type != 'MultiGprompt':
            #       # embeds, _ = self.Preprompt.embed(self.features, self.sp_adj, True, None, False)
            #       embeds, _ = self.Preprompt.embed(self.features, self.sp_adj, True, None, False)

                  
            #       test_lbls = torch.argmax(self.labels[0, self.idx_test], dim=1).cuda()
            #       tot = torch.zeros(1)
            #       tot = tot.cuda()
            #       accs = []
            #       patience = 20
            #       print('-' * 100)
            #       cnt_wait = 0
            #       for i in range(1,6):
            #             # idx_train = torch.load("./data/fewshot_cora/{}-shot_cora/{}/idx.pt".format(self.shot_num,i)).type(torch.long).cuda()
            #             # print('idx_train',idx_train)
            #             # train_lbls = torch.load("./data/fewshot_cora/{}-shot_cora/{}/labels.pt".format(self.shot_num,i)).type(torch.long).squeeze().cuda()
            #             # print("true",i,train_lbls)
            #             self.dataset_name ='Cora'
            #             idx_train = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/train_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).cuda()
            #             print('idx_train',idx_train)
            #             train_lbls = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/train_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().cuda()
            #             print("true",i,train_lbls)

            #             idx_test = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/test_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).cuda()
            #             test_lbls = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/test_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().cuda()
                        
            #             test_embs = embeds[0, idx_test]
            #             best = 1e9
            #             pat_steps = 0
            #             best_acc = torch.zeros(1)
            #             best_acc = best_acc.cuda()
            #             pretrain_embs = embeds[0, idx_train]
            #             for _ in range(50):
            #                   self.DownPrompt.train()
            #                   self.optimizer.zero_grad()
            #                   prompt_feature = self.feature_prompt(self.features)
            #                   # prompt_feature = self.feature_prompt(self.data.x)
            #                   # embeds1 = self.gnn(prompt_feature, self.data.edge_index)
            #                   embeds1= self.Preprompt.gcn(prompt_feature, self.sp_adj , True, False)
            #                   pretrain_embs1 = embeds1[0, idx_train]
            #                   logits = self.DownPrompt(pretrain_embs,pretrain_embs1, train_lbls,1).float().cuda()
            #                   loss = self.criterion(logits, train_lbls)
            #                   if loss < best:
            #                         best = loss
            #                         cnt_wait = 0
            #                   else:
            #                         cnt_wait += 1
            #                         if cnt_wait == patience:
            #                               print('Early stopping at '+str(_) +' eopch!')
            #                               break
                              
            #                   loss.backward(retain_graph=True)
            #                   self.optimizer.step()

            #             prompt_feature = self.feature_prompt(self.features)
            #             embeds1, _ = self.Preprompt.embed(prompt_feature, self.sp_adj, True, None, False)
            #             test_embs1 = embeds1[0, idx_test]
            #             print('idx_test', idx_test)
            #             logits = self.DownPrompt(test_embs, test_embs1, train_lbls)
            #             preds = torch.argmax(logits, dim=1)
            #             acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            #             accs.append(acc * 100)
            #             print('acc:[{:.4f}]'.format(acc))
            #             tot += acc

            #       print('-' * 100)
            #       print('Average accuracy:[{:.4f}]'.format(tot.item() / 10))
            #       accs = torch.stack(accs)
            #       print('Mean:[{:.4f}]'.format(accs.mean().item()))
            #       print('Std :[{:.4f}]'.format(accs.std().item()))
            #       print('-' * 100)
                  
            
            # print("Node Task completed")


            #       print('Mean:[{:.4f}]'.format(accs.mean().item()))
            #       print('Std :[{:.4f}]'.format(accs.std().item()))
            #       print('-' * 100)
                  
            
            # print("Node Task completed")


