import math

import torch
import torch.optim as optim


from dataloader_adjust import PrepareDataloader
from utils.utils import *

from SGCAST_clustering_spa import SGCAST 




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

    def train(self, epoch):
        self.model.train()
        total_loss = 0.
        self.adjust_learning_rate(self.optimizer, epoch)
        self.loss = 0.
        batch_idx = 0
        for data in self.train_loader:

            spot_data = data.float()
            coor = spot_data[:, -2:]
            pdm_spa = torch.cdist(coor, coor).cuda()
            pdm_exp = torch.cdist(spot_data[:, :-2], spot_data[:, :-2]).cuda()

            n = pdm_exp.shape[0]
            conexp_ratio =self.config.train_conexp_ratio
            conspa_ratio = self.config.train_conspa_ratio

            res_exp = pdm_exp.masked_select(~torch.eye(n, dtype=bool).cuda())
            resexp_mean = torch.mean(res_exp)
            resexp_var =  torch.var(res_exp)
            res_spa = pdm_spa.masked_select(~torch.eye(n, dtype=bool).cuda())
            resspa_mean = torch.mean(res_spa)
            resspa_var =  torch.var(res_spa)
            pdm_exp = pdm_exp  
            res_exp = pdm_exp.masked_select(~torch.eye(n, dtype=bool).cuda())
            key_exp = torch.quantile(res_exp,conexp_ratio,interpolation="nearest") 
            lexp = key_exp/math.sqrt((2*104))
            key_spa = torch.quantile(res_spa, conspa_ratio,interpolation="nearest") 
            lspa = key_spa /math.sqrt((2 * 104))

            pdm_exp = pdm_exp.fill_diagonal_(0)
            pdm_spa = pdm_spa.fill_diagonal_(0)

            spot_data = spot_data.cuda()
            _, _, loss = self.model(spot_data[:, :-2], pdm_exp, pdm_spa, lexp, lspa)
            del pdm_spa; del pdm_exp; del spot_data
            self.optimizer.zero_grad()
            loss.backward() 
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


    def write_embeddings(self):
        self.model.eval()
        if not os.path.exists("output_spa/"):
            os.makedirs("output_spa/")
        data_name = os.path.basename(self.config.spot_paths).split('.')[0]
        print("data name:", data_name)

        fp_em = open('./output_spa/' + data_name + '_embeddings_spa.txt', 'w')
        print("ready to write embeddings")

        batch_idx = 0
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
            pdm_exp = pdm_exp 
            res_exp = pdm_exp.masked_select(~torch.eye(n, dtype=bool).cuda())
            key_exp = torch.quantile(res_exp,conexp_ratio,interpolation="nearest") 
            lexp = key_exp/math.sqrt((2*104))
            key_spa = torch.quantile(res_spa, conspa_ratio,interpolation="nearest") 
            lspa = key_spa /math.sqrt((2 * 104))

            pdm_exp = pdm_exp.fill_diagonal_(0)
            pdm_spa = pdm_spa.fill_diagonal_(0)


            spot_data = spot_data.cuda()
            # model forward
            spot_embedding, _, _ = self.model(spot_data[:, :-2], pdm_exp, pdm_spa, lexp, lspa)# pdm)  
            del pdm_spa; del pdm_exp; del spot_data
            spot_embedding = torch.squeeze(spot_embedding, 0).data.cpu().numpy()

            # write embeddings
            test_num, embedding_size = spot_embedding.shape[0], spot_embedding.shape[1]
            for print_i in range(test_num):
                fp_em.write(str(spot_embedding[print_i][0]))

                for print_j in range(1, embedding_size):
                    fp_em.write(' ' + str(spot_embedding[print_i][print_j]))

                fp_em.write('\n')


            progress_bar(batch_idx, len(self.test_loader),
                             'write embeddings for data:' + data_name)

        fp_em.close()
