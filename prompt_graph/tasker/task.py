import torch
from prompt_graph.model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer
from prompt_graph.utils.train_logger import train_info
from prompt_graph.prompt import GPF, GPF_plus, LightPrompt, HeavyPrompt, Gprompt, GPPTPrompt, DiffPoolPrompt, SAGPoolPrompt
from prompt_graph.prompt import featureprompt, downprompt
from prompt_graph.pretrain import GraphPrePrompt, NodePrePrompt
from prompt_graph.registry import PromptRegistry
from torch import nn, optim
from prompt_graph.data import load4node, load4graph
from prompt_graph.utils import Gprompt_tuning_loss
import numpy as np

class BaseTask:
    def __init__(self, pre_train_model_path='None', gnn_type='TransformerConv',
                 hid_dim = 128, num_layer = 2, dataset_name='Cora', prompt_type='None', epochs=100, shot_num=10, device : int = 5, lr =0.001, wd = 5e-4,
                 batch_size = 16, search = False, split_ratio=None, patience=20, save_best=True, checkpoint_dir='checkpoints/downstream',
                 eval_every=1, early_stopping_metric='valid_acc', log_dir='logs', log_file=None):
        
        self.pre_train_model_path = pre_train_model_path
        self.eval_every = eval_every
        self.early_stopping_metric = early_stopping_metric or 'valid_acc'
        self.log_dir = log_dir
        self.log_file = log_file
        # valid_acc/f1/auroc/auprc 均越大越好，早停仅看选定指标不看 loss
        self._higher_is_better = self.early_stopping_metric in ('valid_acc', 'valid_f1', 'valid_auroc', 'valid_auprc')
        self.pre_train_type = self.return_pre_train_type(pre_train_model_path)
        self.device = torch.device('cuda:' + str(device) if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim
        self.num_layer = num_layer
        self.dataset_name = dataset_name
        self.shot_num = shot_num
        self.gnn_type = gnn_type
        self.prompt_type = prompt_type
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size
        self.search = search
        self.split_ratio = split_ratio if split_ratio is not None else [0.8, 0.1, 0.1]
        self.patience = patience
        self.save_best = save_best
        self.checkpoint_dir = checkpoint_dir
        self.initialize_lossfn()

    def metric_from_eva(self, acc, f1, roc, prc):
        """从 Eva 返回的 (acc,f1,roc,prc) 中取出早停指标值"""
        m = self.early_stopping_metric or 'valid_acc'
        if m == 'valid_acc': return acc
        if m == 'valid_f1': return f1
        if m == 'valid_auroc': return roc
        if m == 'valid_auprc': return prc
        return acc

    def initialize_lossfn(self):
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.prompt_type == 'Gprompt':
            self.criterion = Gprompt_tuning_loss()

    def initialize_optimizer(self):
        if self.prompt_type == 'None':
            if self.pre_train_model_path == 'None':
                model_param_group = []
                model_param_group.append({"params": self.gnn.parameters()})
                model_param_group.append({"params": self.answering.parameters()})
                self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
            else:
                model_param_group = []
                model_param_group.append({"params": self.gnn.parameters()})
                model_param_group.append({"params": self.answering.parameters()})
                self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
                # self.optimizer = optim.Adam(self.answering.parameters(), lr=self.lr, weight_decay=self.wd)

        elif self.prompt_type == 'All-in-one':
            self.pg_opi = optim.Adam( self.prompt.parameters(), lr=1e-6, weight_decay= self.wd)
            self.answer_opi = optim.Adam( self.answering.parameters(), lr=self.lr, weight_decay= self.wd)
        elif self.prompt_type in ['GPF', 'GPF-plus']:
            model_param_group = []
            model_param_group.append({"params": self.prompt.parameters()})
            model_param_group.append({"params": self.answering.parameters()})
            self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
        elif self.prompt_type in ['Gprompt']:
            self.pg_opi = optim.Adam(self.prompt.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.prompt_type in ['GPPT']:
            self.pg_opi = optim.Adam(self.prompt.parameters(), lr=2e-3, weight_decay=5e-4)
        elif self.prompt_type == 'MultiGprompt':
            self.optimizer = optim.Adam([*self.DownPrompt.parameters(),*self.feature_prompt.parameters()], lr=self.lr)

    def initialize_prompt(self):
        # Try registry first (allows pluggable prompts without modifying this file)
        prompt_class = PromptRegistry.get_prompt_class(self.prompt_type)
        if prompt_class is not None:
            kwargs = PromptRegistry.get_prompt_init_kwargs(self.prompt_type, self)
            if kwargs:
                self.prompt = prompt_class(**kwargs).to(self.device)
            else:
                self.prompt = prompt_class(self.hid_dim).to(self.device)  # Gprompt-style fallback
            return

        if self.prompt_type == 'None':
            self.prompt = None
        elif self.prompt_type == 'GPPT':
            if(self.task_type=='NodeTask'):
                if self.dataset_name == 'Texas':
                    self.prompt = GPPTPrompt(self.hid_dim, 5, self.output_dim, device = self.device)
                else:
                    self.prompt = GPPTPrompt(self.hid_dim, self.output_dim, self.output_dim, device = self.device)
            elif(self.task_type=='GraphTask'):
                self.prompt = GPPTPrompt(self.hid_dim, self.output_dim, self.output_dim, device = self.device)                
        elif self.prompt_type =='All-in-one':
            if(self.task_type=='NodeTask'):
                self.prompt = HeavyPrompt(token_dim=self.input_dim, token_num=10, cross_prune=0.1, inner_prune=0.3).to(self.device)
            elif(self.task_type=='GraphTask'):
                self.prompt = HeavyPrompt(token_dim=self.input_dim, token_num=10, cross_prune=0.1, inner_prune=0.3).to(self.device)
        elif self.prompt_type == 'GPF':
            self.prompt = GPF(self.input_dim).to(self.device)
        elif self.prompt_type == 'GPF-plus':
            self.prompt = GPF_plus(self.input_dim, 20).to(self.device)
        elif self.prompt_type == 'Gprompt':
            self.prompt = Gprompt(self.hid_dim).to(self.device)
        elif self.prompt_type == 'MultiGprompt':
            nonlinearity = 'prelu'
            self.Preprompt = NodePrePrompt(self.dataset_name, self.hid_dim, nonlinearity, 0.9, 0.9, 0.1, 0.001, 1, 0.3).to(self.device)
            self.Preprompt.load_state_dict(torch.load(self.pre_train_model_path))
            self.Preprompt.eval()
            self.feature_prompt = featureprompt(self.Preprompt.dgiprompt.prompt,self.Preprompt.graphcledgeprompt.prompt,self.Preprompt.lpprompt.prompt).to(self.device)
            dgiprompt = self.Preprompt.dgi.prompt  
            graphcledgeprompt = self.Preprompt.graphcledge.prompt
            lpprompt = self.Preprompt.lp.prompt
            self.DownPrompt = downprompt(dgiprompt, graphcledgeprompt, lpprompt, 0.001, self.hid_dim, self.output_dim, self.device).to(self.device)
        else:
            raise KeyError(" We don't support this kind of prompt.")

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
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        self.gnn.to(self.device)

        if self.pre_train_model_path != 'None' and self.prompt_type != 'MultiGprompt':
            if self.gnn_type not in self.pre_train_model_path :
                raise ValueError(f"the Downstream gnn '{self.gnn_type}' does not match the pre-train model")
            if self.dataset_name not in self.pre_train_model_path :
                raise ValueError(f"the Downstream dataset '{self.dataset_name}' does not match the pre-train dataset")

            self.gnn.load_state_dict(torch.load(self.pre_train_model_path, map_location='cpu'))
            self.gnn.to(self.device)
            train_info("Pre-trained weights loaded from {}".format(self.pre_train_model_path))

    def return_pre_train_type(self, pre_train_model_path):
        names = ['None', 'DGI', 'GraphMAE','Edgepred_GPPT', 'Edgepred_Gprompt','GraphCL', 'SimGRACE']
        for name in names:
            if name  in  pre_train_model_path:
                return name


      
 
            
      
