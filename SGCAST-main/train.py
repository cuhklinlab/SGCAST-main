import math

import torch
import torch.optim as optim


from dataloader import PrepareDataloader
from utils.utils import *

from SGCAST_clustering import SGCAST




class Training():
    def __init__(self, config):
        self.config = config
        # load data
        self.train_loader, self.test_loader,self.training_iters = PrepareDataloader(
            config).getloader()


        # initialize dataset
        self.model = SGCAST(config.nfeat, config.nhid, config.nemb).cuda() 
        self.actual_lr = 0.1

        # initialize optimizer 
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr_start,
                                           weight_decay=0.)


    def adjust_learning_rate(self, optimizer, epoch):
        # the function makes the learning rate 0.2 at first 80 epochs and decay to 0.1 at last 20 epochs
        if ((epoch - 0) // self.config.lr_decay_epoch) < self.config.lr_times:
            self.actual_lr = self.config.lr_start * (.5 ** ((epoch - 0) // self.config.lr_decay_epoch)) 
        else:
            self.actual_lr = self.actual_lr
        if (epoch - 0) % self.config.lr_decay_epoch == 0:
            print('LR is set to {}'.format(self.actual_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.actual_lr

    def load_checkpoint(self, args):
        if self.config.checkpoint is not None:
            if os.path.isfile(self.config.checkpoint):
                print("=> loading checkpoint '{}'".format(self.config.checkpoint))
                checkpoint = torch.load(self.config.checkpoint)
                self.model.load_state_dict(checkpoint['model_encoding_state_dict'])
            else:
                print("=> no resume checkpoint found at '{}'".format(self.config.checkpoint))
                
    # train function
    def train(self, epoch):
        self.model.train()
        total_loss = 0.
        self.adjust_learning_rate(self.optimizer, epoch)
        self.loss = 0.
        batch_idx = 0
        for data in self.train_loader:
            # load data for a mini-batch
            spot_data = data.float()
            coor = spot_data[:, -2:]
            
            # calculate pairwise distance matrix for spatial coordiantes and gene expression
            pdm_spa = torch.cdist(coor, coor).cuda()
            pdm_exp = torch.cdist(spot_data[:, :-2], spot_data[:, :-2]).cuda()

            n = pdm_exp.shape[0]
            conexp_ratio =self.config.train_conexp_ratio
            conspa_ratio = self.config.train_conspa_ratio
            
            # select non-diagonal elements 
            res_exp = pdm_exp.masked_select(~torch.eye(n, dtype=bool).cuda())
            
            # calculate mean and variance of distances
            resexp_mean = torch.mean(res_exp)
            resexp_var =  torch.var(res_exp)
            res_spa = pdm_spa.masked_select(~torch.eye(n, dtype=bool).cuda())
            resspa_mean = torch.mean(res_spa)
            resspa_var =  torch.var(res_spa)
            
            # rescale distances in gene expression
            pdm_exp = pdm_exp * math.sqrt(resspa_var/resexp_var)  + resspa_mean - resexp_mean
            res_exp = pdm_exp.masked_select(~torch.eye(n, dtype=bool).cuda())
            
            # find the quantile (conexp_ratio) of distances in expression layer
            key_exp = torch.quantile(res_exp,conexp_ratio,interpolation="higher") 
            
            # determine l in expression layer for this mini-batch
            lexp = key_exp/math.sqrt((2*104))  # torch.exp(-104) = 0 ; -104 is the largest integer.
            
            # find the quantile (conspa_ratio) of distances in spatial layer
            key_spa = torch.quantile(res_spa, conspa_ratio,interpolation="higher") 
            
            # determine l in spatial layer for this mini-batch
            lspa = key_spa /math.sqrt((2 * 104))
            
            # the distance on diagonal line is 0 (the distance to spot itself)
            pdm_exp = pdm_exp.fill_diagonal_(0)
            pdm_spa = pdm_spa.fill_diagonal_(0)
            
            # move data to GPU
            spot_data = spot_data.cuda()
            _, _, loss = self.model(spot_data[:, :-2], pdm_exp, pdm_spa, lexp, lspa)
            del pdm_spa; del pdm_exp; del spot_data
            self.optimizer.zero_grad()
            loss.backward() 
            
            # clip gradient by 1
            clipping_value = 1  
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipping_value)
            self.optimizer.step()
            total_loss += loss.data.item()


            progress_bar(batch_idx, self.training_iters,
                         'loss: %.3f ' % (total_loss / (batch_idx + 1)
                         ))
            batch_idx += 1
            self.loss = total_loss / (batch_idx + 1)
        # save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'model_encoding_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        })

    # write results after training
    def write_embeddings(self):
        self.model.eval()
        if not os.path.exists("output/"):
            os.makedirs("output/")
        data_name = os.path.basename(self.config.spot_paths).split('.')[0]
        print("data name:", data_name)

        fp_em = open('./output/' + data_name + '_embeddings.txt', 'w')
        print("ready to write embeddings")

        batch_idx = 0
        # same procedure as training but no backward loss to update parameter in the model.
        for data in self.test_loader:

            spot_data = data.float()
            coor = spot_data[:, -2:]
            pdm_spa = torch.cdist(coor, coor).cuda()
            pdm_exp = torch.cdist(spot_data[:, :-2], spot_data[:, :-2]).cuda()

            n = pdm_exp.shape[0]

            conexp_ratio = self.config.test_conexp_ratio 
            conspa_ratio = self.config.test_conspa_ratio 
            res_exp = pdm_exp.masked_select(~torch.eye(n, dtype=bool).cuda())
            resexp_mean = torch.mean(res_exp)
            resexp_var = torch.var(res_exp)
            res_spa = pdm_spa.masked_select(~torch.eye(n, dtype=bool).cuda())
            resspa_mean = torch.mean(res_spa)
            resspa_var = torch.var(res_spa)
            pdm_exp = pdm_exp * math.sqrt(resspa_var / resexp_var) + resspa_mean - resexp_mean
            res_exp = pdm_exp.masked_select(~torch.eye(n, dtype=bool).cuda())
            key_exp = torch.quantile(res_exp,conexp_ratio,interpolation="higher") 
            lexp = key_exp/math.sqrt((2*104))
            key_spa = torch.quantile(res_spa, conspa_ratio,interpolation="higher") 
            lspa = key_spa /math.sqrt((2 * 104))

            pdm_exp = pdm_exp.fill_diagonal_(0)
            pdm_spa = pdm_spa.fill_diagonal_(0)


            spot_data = spot_data.cuda()
            # model forward
            spot_embedding, _, _ = self.model(spot_data[:, :-2], pdm_exp, pdm_spa, lexp, lspa)# pdm)  #
            del pdm_spa; del pdm_exp; del spot_data
            
            # move results from GPU to CPU
            spot_embedding = torch.squeeze(spot_embedding, 0).data.cpu().numpy()

            # write embeddings
            test_num, embedding_size = spot_embedding.shape[0], spot_embedding.shape[1]
            # print("test num:", test_num)
            for print_i in range(test_num):
                fp_em.write(str(spot_embedding[print_i][0]))

                for print_j in range(1, embedding_size):
                    fp_em.write(' ' + str(spot_embedding[print_i][print_j]))

                fp_em.write('\n')


            progress_bar(batch_idx, len(self.test_loader),
                             'write embeddings for data:' + data_name)

        fp_em.close()






