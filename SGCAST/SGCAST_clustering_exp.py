import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import scanpy as sc
import math



class SGCAST(nn.Module):
    def __init__(self, nfeat, nhid, nemb):  
        super().__init__()
        self.weight_exp = nn.Parameter(torch.FloatTensor(nfeat, nhid))
        self.weight_spa = nn.Parameter(torch.FloatTensor(nhid, nemb))


        self.bias_spa_en = nn.Parameter(torch.FloatTensor(nemb))
        self.bias_exp_en = nn.Parameter(torch.FloatTensor(nhid))
        self.bias_spa_de = nn.Parameter(torch.FloatTensor(nhid))
        self.bias_exp_de = nn.Parameter(torch.FloatTensor(nfeat))
        self.act = nn.ELU(alpha=2.0) #nn.ReLU()
        self.nfeat = nfeat
        self.nhid = nhid
        self.initialize_weights()





    def initialize_weights(self):
        # initialization

        stdv = torch.tensor(1. / math.sqrt(self.nhid))
        stdvexp = torch.tensor(1. / math.sqrt(self.nfeat))
    
        self.weight_exp.data = self.weight_exp.data.uniform_(-stdvexp, stdvexp) #exp math.sqrt(stdv*stdvexp)
        self.weight_spa.data = self.weight_spa.data.uniform_(-stdv, stdv) 
        self.bias_spa_en.data = self.bias_spa_en.data.uniform_(-stdv, stdv) 
        self.bias_exp_en.data = self.bias_exp_en.data.uniform_(-stdv, stdv) 
        self.bias_spa_de.data = self.bias_spa_de.data.uniform_(-stdv, stdv) 
        self.bias_exp_de.data = self.bias_exp_de.data.uniform_(-stdvexp, stdvexp) 



    def forward_encoder(self, x, adj_exp, adj_spa):
        x = torch.Tensor(x).cuda()
        support = torch.mm(x, self.weight_exp) 
        output = self.act(torch.spmm(adj_exp, support) + self.bias_exp_en)
        return output

    def forward_decoder(self, x, adj_exp, adj_spa):
        iden3 = (torch.eye(adj_exp.shape[0])*torch.Tensor([3])).cuda()
        support = torch.mm(x, torch.transpose(self.weight_exp, 0, 1)) 
        output = self.act(torch.spmm((iden3-adj_exp), support) + self.bias_exp_de) 
        return output

    def loss_function(self, x,  pred): 
        loss_ae = torch.mean((pred - x) ** 2, dim=1)
        loss_ae = torch.mean(loss_ae)
        return loss_ae

    def forward(self, x, pdm_exp, pdm_spa, lexp, lspa):  
        adj_exp = torch.exp(-1 * (pdm_exp ** 2) / (2 * (lexp ** 2)))
        adj_spa = torch.exp(-1 * (pdm_spa ** 2) / (2 * (lspa ** 2)))
        y = self.forward_encoder(x, adj_exp, adj_spa) 
        pred = self.forward_decoder(y, adj_exp, adj_spa)
        loss = self.loss_function(x, pred) #p, q,
        return y, pred, loss



