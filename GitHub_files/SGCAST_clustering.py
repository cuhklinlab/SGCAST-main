import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import scanpy as sc
import math



class SGCAST(nn.Module):
    def __init__(self, nfeat, nhid, nemb):  #, n_clusters=None, alpha=0.2, res=0.2, n_neighbors=10
        super().__init__()
        # self.weight_exp_en= nn.Parameter(torch.FloatTensor(nfeat, nhid))
        # self.weight_spa_en = nn.Parameter(torch.FloatTensor(nhid, nhid))
        # self.weight_exp_de = nn.Parameter(torch.FloatTensor(nhid, nfeat))
        # self.weight_spa_de = nn.Parameter(torch.FloatTensor(nhid, nhid))
        # self.weight_exp = nn.Parameter(torch.FloatTensor(nfeat, nhid))
        # self.weight_prj = nn.Parameter(torch.FloatTensor(nfeat, nhid))
        self.weight_exp = nn.Parameter(torch.FloatTensor(nfeat, nhid))
        self.weight_spa = nn.Parameter(torch.FloatTensor(nhid, nemb))

        # self.bias_prj_en = nn.Parameter(torch.FloatTensor(nhid))
        self.bias_spa_en = nn.Parameter(torch.FloatTensor(nemb))
        self.bias_exp_en = nn.Parameter(torch.FloatTensor(nhid))
        self.bias_spa_de = nn.Parameter(torch.FloatTensor(nhid))
        self.bias_exp_de = nn.Parameter(torch.FloatTensor(nfeat))
        # self.bias_prj_de = nn.Parameter(torch.FloatTensor(nfeat))
        # self.ac = nn.ReLU()
        self.act = nn.ELU(alpha=2.0) #nn.ReLU()
        # self.lexp = 900# nn.Parameter(torch.FloatTensor(1)) 12000 10% nonzero/ 5%
        # self.lspa = 100# nn.Parameter(torch.FloatTensor(1))  ### 100
        self.nfeat = nfeat
        self.nhid = nhid
        # self.mu determined by the init method
        # self.alpha = alpha
        self.initialize_weights()
        # self.dropout = dropout
        # for m in self.encoder1.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data = nn.init.xavier_uniform(m.weight.data)




    def initialize_weights(self):
        '''
        权重初始化，方法同vit权重初始化方法
        :return:
        '''
        # initialization

        stdv = torch.tensor(1. / math.sqrt(self.nhid))
        stdvexp = torch.tensor(1. / math.sqrt(self.nfeat))
        # self.weight_exp_en.data = self.weight_exp_en.data.uniform_(-stdvexp, stdvexp)
        # self.weight_spa_en.data = self.weight_spa_en.data.uniform_(-stdv, stdv)
        # self.weight_exp_de.data = self.weight_exp_de.data.uniform_(-stdvexp, stdvexp)
        # self.weight_spa_de.data = self.weight_spa_de.data.uniform_(-stdv, stdv)
        # self.weight_prj.data = self.weight_prj.data.uniform_(-stdvexp, stdvexp)
        self.weight_exp.data = self.weight_exp.data.uniform_(-stdvexp, stdvexp) #exp math.sqrt(stdv*stdvexp)
        self.weight_spa.data = self.weight_spa.data.uniform_(-stdv, stdv) #dv
        # self.bias_prj_en.data = self.bias_prj_en.data.uniform_(-stdv, stdv)
        self.bias_spa_en.data = self.bias_spa_en.data.uniform_(-stdv, stdv) #dv
        self.bias_exp_en.data = self.bias_exp_en.data.uniform_(-stdv, stdv) #dv
        self.bias_spa_de.data = self.bias_spa_de.data.uniform_(-stdv, stdv) #dv
        self.bias_exp_de.data = self.bias_exp_de.data.uniform_(-stdvexp, stdvexp) #exp
        # self.bias_prj_de.data = self.bias_prj_de.data.uniform_(-stdvexp, stdvexp)
        # nn.init.constant_(self.lexp, 80.0)
        # nn.init.constant_(self.lspa, 70.0)

    # def mu_init(self, x, adj_exp, adj_spa):
    #     adata = sc.AnnData(x.data.cpu().numpy())
    #     if self.init == "kmeans":
    #         print("Initializing cluster centers with kmeans, n_clusters known")
    #         kmeans = KMeans(self.n_clusters, n_init=20)
    #         y_pred = kmeans.fit_predict(X.data.cpu().numpy())
    #         # if init_spa:
    #         #     # ------Kmeans use exp and spatial
    #         #     y_pred = kmeans.fit_predict(features.data.cpu().numpy())
    #         # else:
    #         #     # ------Kmeans only use exp info, no spatial
    #         #     y_pred = kmeans.fit_predict(X.data.cpu().numpy())  # Here we use X as numpy
    #     elif self.init == "louvain":
    #         print("Initializing cluster centers with louvain,  ")  # resolution = , res
    #         sc.pp.neighbors(adata, use_rep="X", n_neighbors=self.n_neighbors)
    #         sc.tl.louvain(adata, resolution=self.res)
    #         y_pred = adata.obs['louvain'].astype(int).to_numpy()
    #         self.n_clusters = len(np.unique(y_pred))
    #         self.mu = nn.Parameter(torch.Tensor(self.n_clusters, self.nhid).cuda())
    #         # *3
    #     self.mu = nn.Parameter(torch.Tensor(self.n_clusters, self.nhid).cuda())
    #     features = self.forward_encoder(x, adj_exp, adj_spa)
    #     feat = pd.DataFrame(features.data.cpu().numpy(), index=np.arange(0, features.data.cpu().numpy().shape[0]))
    #     Group = pd.Series(y_pred, index=np.arange(0, feat.shape[0]), name="Group")
    #     Mergefeature = pd.concat([feat, Group], axis=1)
    #     cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
    #     return cluster_centers


    def forward_encoder(self, x, adj_exp, adj_spa):
        x = torch.Tensor(x).cuda()
        # support = torch.mm(x, self.weight_prj) #self.weight_exp_en
        # output = self.act( support + self.bias_prj_en)
        support = torch.mm(x, self.weight_exp) #self.weight_exp_en  x
        output = self.act(torch.spmm(adj_exp, support) + self.bias_exp_en)
        support = torch.mm(output, self.weight_spa)  #self.weight_spa_en
        output = self.act(torch.spmm(adj_spa, support) + self.bias_spa_en)
        return output

    def forward_decoder(self, x, adj_exp, adj_spa):
        support = torch.mm(x, torch.transpose(self.weight_spa, 0, 1))# self.weight_spa_de
        iden3 = (torch.eye(adj_spa.shape[0])*torch.Tensor([3])).cuda()
        # iden2 = (torch.eye(adj_spa.shape[0])*torch.Tensor([2])).cuda()
        # print("iden3 device:", iden3.device)
        output = self.act(torch.spmm((iden3-adj_spa), support) + self.bias_spa_de)
        support = torch.mm(output, torch.transpose(self.weight_exp, 0, 1)) # self.weight_exp_de
        output = self.act(torch.spmm((iden3-adj_exp), support) + self.bias_exp_de) #11.9
        # support = torch.mm(output, torch.transpose(self.weight_prj, 0, 1)) # self.weight_exp_de
        # output = self.act( support + self.bias_prj_de)
        return output

    def loss_function(self, x,  pred): # y
        loss_ae = torch.mean((pred - x) ** 2, dim=1)
        # print("lossae shape:", loss_ae.shape)
        loss_ae = torch.mean(loss_ae)

        # loss_clust = torch.mean(torch.sum(p * torch.log((p) / (q)),dim=1))
        # loss = loss_ae + loss_clust
        # print("loss_ae:", loss_ae)
        return loss_ae

    def forward(self, x, pdm_exp, pdm_spa, lexp, lspa):  # ,adj3 ,p3
        adj_exp = torch.exp(-1 * (pdm_exp ** 2) / (2 * (lexp ** 2)))
        adj_spa = torch.exp(-1 * (pdm_spa ** 2) / (2 * (lspa ** 2)))
        # n = pdm_exp.shape[0]
        # res_exp = adj_exp.masked_select(~torch.eye(n, dtype=bool).cuda())
        # res_spa = adj_spa.masked_select(~torch.eye(n, dtype=bool).cuda())
        # q = torch.tensor([0.89,0.0,0.91,0.92,0.93, 0.94, 0.95, 0.96, 0.97, 0.98,0.99,1.00]).cuda()

        # avg_aft = torch.sum(adj_spa) / n
        # # # res_spa = adj_spa.masked_select(~torch.eye(n, dtype=bool).cuda())
        # print("exp quantile:", torch.quantile(res_exp, q,interpolation="nearest"))
        # print("spa quantile:", torch.quantile(res_spa, q,interpolation="nearest"))
        # print("avg exp value:", avg_aftexp)
        # print("avg  spa value:", avg_aft)
        ##########11.9adjust
        # print("pdm_exp max:", torch.max(pdm_exp))
        # print("adj_gen zero:",torch.sum(adj_exp==0))
        # # print("adj_gen", adj_exp )
        # print("adj_spa zero:",torch.sum(adj_spa==0))
        # # print("adj_spa", adj_exp )
        #########11.9adjust
        # mu = self.mu_init(x, adj_exp, adj_spa)
        # self.mu.data = torch.Tensor(mu).cuda()
        y = self.forward_encoder(x, adj_exp, adj_spa) # adj_spa,adj_exp
        pred = self.forward_decoder(y, adj_exp, adj_spa) # adj_spa, adj_exp
        # print("p shape:", p.shape)
        # print("p shape:", p[:3,:])
        # print("x device:", x.device)
        loss = self.loss_function(x, pred) #p, q,
        return y, pred, loss



    # def adjust_learning_rate(self, optimizer, epoch):
    #     lr = self.lr * (1.5 ** ((epoch - 0) // self.lr_decay_epoch))
    #     if lr > 0.3: lr = 0.3
    #     if (epoch - 0) % self.lr_decay_epoch == 0:
    #         print('LR is set to {}'.format(lr))
    #
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #     self.actual_lr = lr
    #
    # def target_distribution(self, q):
    #     # weight = q ** 2 / q.sum(0)
    #     # return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
    #     p = torch.pow(q, 2) / torch.sum(q, dim=0)  # 2 to 3
    #     p = p / torch.sum(p, dim=1, keepdim=True)
    #     return p



    # def fit(self, X, adj_gen, adj_spa, phis, pgen, lr=0.001, max_epochs=5000, update_interval=3, trajectory_interval=50,
    #         weight_decay=5e-4, init='kmeans', n_clusters=10, init_spa=True, n_neighbors=10, res=1.0,
    #         fresh_interval=2000,
    #         opt="sgd",
    #         tol=1e-3):  # adj_spa, pspa,init_spa=True,  n_neighbors=10,  n_clusters=10,
    #     self.lr = lr
    #     self.trajectory = []
    #     self.n_clusters = n_clusters
    #     self.n_neighbors = n_neighbors
    #     if opt == "sgd":
    #         optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
    #     elif opt == "admin":
    #         optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
    #     features = self.encoder1(X, adj_gen, adj_spa)
    #     # print("x1:",x1)
    #     # print("x2:",x2)
    #     # features = torch.squeeze(torch.mean(torch.cat((torch.unsqueeze(x1, 0), torch.unsqueeze(x2, 0)), 0), dim=0, keepdim=True), 0)
    #
    #     # ----------------------------------------------------------------
    #     if init == "kmeans":
    #         print("Initializing cluster centers with kmeans, n_clusters known")
    #         self.n_clusters = n_clusters
    #         kmeans = KMeans(self.n_clusters, n_init=20)
    #         if init_spa:
    #             # ------Kmeans use exp and spatial
    #             y_pred = kmeans.fit_predict(features.data.cpu().numpy())
    #         else:
    #             # ------Kmeans only use exp info, no spatial
    #             y_pred = kmeans.fit_predict(X.data.cpu().numpy())  # Here we use X as numpy
    #     elif init == "louvain":
    #         print("Initializing cluster centers with louvain,  ")  # resolution = , res
    #         if init_spa:
    #             adata = sc.AnnData(features.data.cpu().numpy())
    #         else:
    #             adata = sc.AnnData(X.data.cpu().numpy())
    #         if res is None:
    #             _, y_pred = find_res_at(adata, self.n_clusters, n_neighbors=self.n_neighbors)
    #         else:
    #             res_used = res
    #             sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    #             sc.tl.louvain(adata, resolution=res_used)
    #             y_pred = adata.obs['louvain'].astype(int).to_numpy()
    #             self.n_clusters = len(np.unique(y_pred))
    #         # res = find_res_at(adata, self.n_clusters,n_neighbors=self.neighbors)
    #         # sc.pp.neighbors(adata, n_neighbors=self.n_neighbors)
    #         # sc.tl.louvain(adata,resolution=res)
    #         # y_pred=adata.obs['louvain'].astype(int).to_numpy()
    #         print("number of initial clusters:", len(np.unique(y_pred)))
    #
    #     # ----------------------------------------------------------------
    #
    #     y_pred_last = torch.tensor(y_pred).long().cuda()
    #     self.mu = Parameter(torch.Tensor(self.n_clusters, self.nhid).cuda())  # *3
    #     self.trajectory.append(y_pred)
    #     feat = pd.DataFrame(features.data.cpu().numpy(), index=np.arange(0, features.data.cpu().numpy().shape[0]))
    #     Group = pd.Series(y_pred, index=np.arange(0, feat.shape[0]), name="Group")
    #     Mergefeature = pd.concat([feat, Group], axis=1)
    #     cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
    #
    #     self.mu.data = torch.Tensor(cluster_centers).cuda()
    #     # print("shape of mu:", self.mu.shape)
    #     del features
    #     self.train()
    #
    #     for epoch in range(max_epochs):
    #
    #         with torch.autograd.set_detect_anomaly(True):
    #             """
    #             if epoch > 0 and epoch % fresh_interval == 0:
    #                 del self.mu
    #                 adata = sc.AnnData(z.data.cpu().numpy())
    #                 if res is None:
    #                     _, y_pred2 = find_res_at(adata, self.n_clusters,n_neighbors=self.n_neighbors)
    #                 else:
    #                     res_used=res
    #                     sc.pp.neighbors(adata, n_neighbors=self.n_neighbors)
    #                     sc.tl.louvain(adata,resolution=res_used)
    #                     y_pred2=adata.obs['louvain'].astype(int).to_numpy()
    #                 print("number of freshed clusters:", len(np.unique(y_pred2)))
    #                 #y_pred_last =  torch.tensor(y_pred2).long().cuda()
    #                 feat=pd.DataFrame(z.data.cpu().numpy(),index=np.arange(0,z.data.cpu().numpy().shape[0]))
    #                 Group=pd.Series(y_pred2,index=np.arange(0,feat.shape[0]),name="Group")
    #                 Mergefeature=pd.concat([feat,Group],axis=1)
    #                 cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
    #                 self.mu = Parameter(torch.Tensor(len(np.unique(y_pred2)), self.nhid).cuda())
    #                 self.mu.data=torch.Tensor(cluster_centers).cuda()
    #             """
    #
    #             z, q = self.forward(X, adj_gen, adj_spa)
    #             if epoch % update_interval == 0:
    #                 # _, q = self.forward(X, adj_gen,adj_spa,adj_his)
    #                 if epoch >= fresh_interval:  # epoch > 0 and epoch % fresh_interval == 0:
    #                     p = self.target_distribution2(q)
    #                 else:
    #                     p = self.target_distribution(q)
    #                 p = p.detach()
    #                 # print("q:", q,flush=True)
    #             if epoch % 10 == 0:
    #                 print("Epoch ", epoch)
    #             optimizer.zero_grad()
    #             # z, q = self.forward(X, adj_his,adj_gen)  # X, adj_his,adj_spa,adj_gen, phis,pspa,pgen) #adj_his,
    #             # p = self.target_distribution(q) # this line is added
    #             # print("p:", p,flush=True)
    #             # print("q:", q,flush=True)
    #             loss = self.loss_function(p, q)  # nn.CrossEntropyLoss()
    #             print("loss:", loss.item())
    #             # print("mu:", self.mu)
    #             # loss.requires_grad = True
    #             loss.backward()  # retain_graph=True
    #             optimizer.step()
    #             if epoch % trajectory_interval == 0:
    #                 self.trajectory.append(torch.argmax(q, dim=1))
    #
    #             # Check stop criterion
    #             y_pred = torch.argmax(q, dim=1)
    #             delta_label = torch.sum(y_pred != y_pred_last).float() / X.shape[
    #                 0]  # np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
    #             y_pred_last = y_pred
    #             self.adjust_learning_rate(optimizer, epoch)
    #             self.writer.add_scalar('Loss/train', loss.item(), epoch)
    #             self.writer.add_scalar('lr', self.actual_lr, epoch)
    #
    #             if epoch > 0 and (epoch - 1) % update_interval == 0 and loss < tol:  # loss delta_label
    #                 print('delta_label ', delta_label, '< tol ', tol)
    #                 print("Reach tolerance threshold. Stopping training.")
    #                 print("Total epoch:", epoch)
    #                 break
    #
    # def predict(self, X, adj_gen, adj_spa, phis, pgen):  # X,  adj_his,adj_spa,adj_gen,phis,pspa,pgen): #
    #     self.eval()
    #     z, q = self.forward(X, adj_gen, adj_spa)  # adj_spa, pspa,
    #     return z, q


        # n = pdm_exp.shape[0]
        # res_exp = adj_exp.masked_select(~torch.eye(n, dtype=bool).cuda())
        # res_spa = adj_spa.masked_select(~torch.eye(n, dtype=bool).cuda())
        # q = torch.tensor([0.93, 0.94, 0.95, 0.96, 0.97, 0.98,0.99,1.00]).cuda()
        # # # max_spa = torch.quantile(res_spa, 1.0, interpolation="nearest")
        # # #
        # # # if max_spa>0.5:
        # # #     adj_spa = adj_spa * 0.2
        # # #
        # # # adj_spa = adj_spa.fill_diagonal_(1)
        # avg_aftexp = torch.sum(adj_exp) / n
        # avg = torch.sum(adj_spa) / n
        # avg = avg.detach()
        # if avg>2.2:
        #     adj_spa = adj_spa * 1.8/ avg.data.item()
        #     adj_spa = adj_spa.fill_diagonal_(1)
        # avg_aft = torch.sum(adj_spa) / n
        # # # res_spa = adj_spa.masked_select(~torch.eye(n, dtype=bool).cuda())
        # print("exp quantile:", torch.quantile(res_exp, q,interpolation="nearest"))
        # print("spa quantile:", torch.quantile(res_spa, q,interpolation="nearest"))
        # print("avg exp value:", avg_aftexp)
        # print("avg  spa value:", avg_aft)
        ##########11.9adjust
        # print("pdm_exp max:", torch.max(pdm_exp))
        # print("adj_gen zero:",torch.sum(adj_exp==0))
        # # print("adj_gen", adj_exp )
        # print("adj_spa zero:",torch.sum(adj_spa==0))
        # # print("adj_spa", adj_exp )