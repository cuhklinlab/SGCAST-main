import math

import torch
import torch.optim as optim


from dataloader_adjust import PrepareDataloader
from utils.utils import *

from SGCAST_clustering import SGCAST




class Training():
    def __init__(self, config):
        self.config = config
        # load data
        self.train_loader, self.test_loader,self.training_iters = PrepareDataloader(
            config).getloader()


        # initialize dataset
        self.model = SGCAST(config.nfeat, config.nhid, config.nemb).cuda() #torch.nn.DataParallel()
        self.actual_lr = 0.1
        # initialize criterion (loss)
        # self.criterion_cell = CellLoss()
        # self.criterion_encoding = EncodingLoss(dim=64, p=config.p)
        # self.l1_regular = L1regularization()

        # initialize optimizer (sgd/momemtum/weight decay)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr_start,
                                           # momentum=self.config.momentum,
                                           weight_decay=0.) #SGD Adam


    def adjust_learning_rate(self, optimizer, epoch):
        if ((epoch - 0) // self.config.lr_decay_epoch) < self.config.lr_times:
            self.actual_lr = self.config.lr_start * (.5 ** ((epoch - 0) // self.config.lr_decay_epoch)) #0.1 1.5
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

        # initialize iterator
        # iter_spot_loaders = []
        # for spot_loader in self.train_spot_loaders:
        #     iter_spot_loaders.append(cycle(iter(spot_loader)))
        #     print("iter_spot_loaders type:", type(iter_spot_loaders))

        ############### this part still gets problems!!!! loss in each iter_spot_loader?
        batch_idx = 0
        for data in self.train_loader:
            # spot forward

            # spot_cell_predictions = []
            # for iter_spot_loader in iter_spot_loaders:
            #     spot_data = next(iter_spot_loader)
            #     # print("spot_data type:", type(spot_data))
            #     # print("spot_data shape:", spot_data.shape)
            #     # print("spot_data dtype:", spot_data.dtype)

            spot_data = data.float()
            coor = spot_data[:, -2:]
            pdm_spa = torch.cdist(coor, coor).cuda()
            pdm_exp = torch.cdist(spot_data[:, :-2], spot_data[:, :-2]).cuda()

            n = pdm_exp.shape[0]
            # self.exp = n * 2 * 960 # 2000
            self.conexp_ratio = 0.07 #0.07 high resolution  0.02
            self.conspa_ratio = 0.07 #0.07  high resolution 0.02

            # self.spa = n * 2 * 63
            res_exp = pdm_exp.masked_select(~torch.eye(n, dtype=bool).cuda())
            resexp_mean = torch.mean(res_exp)
            resexp_var =  torch.var(res_exp)
            res_spa = pdm_spa.masked_select(~torch.eye(n, dtype=bool).cuda())
            resspa_mean = torch.mean(res_spa)
            resspa_var =  torch.var(res_spa)
            # print("mean df before:", resspa_mean - resexp_mean)
            pdm_exp = pdm_exp * math.sqrt(resspa_var/resexp_var)  + resspa_mean - resexp_mean#* math.sqrt(resspa_var/resexp_var) #11.9
            # print("rt_Sd:", math.sqrt(resspa_var/resexp_var))
            res_exp = pdm_exp.masked_select(~torch.eye(n, dtype=bool).cuda())
            # print("quantile:",self.exp/n/(n-1))
            key_exp = torch.quantile(res_exp,self.conexp_ratio,interpolation="nearest") #self.exp/n/(n-1)  'higher'
            lexp = key_exp/math.sqrt((2*104))
            # print("keyexp:", key_exp)
            key_spa = torch.quantile(res_spa, self.conspa_ratio,interpolation="nearest") #self.spa / n / (n - 1)  'higher'
            # print("keyspa:", key_spa)
            lspa = key_spa /math.sqrt((2 * 104))
            # print("lexp:", lexp)
            # print("lspa:", lspa)
            # print("var ratio:", math.sqrt(resspa_var/resexp_var))
            # res_exp = pdm_exp.masked_select(~torch.eye(n, dtype=bool).cuda())
            # resexp_mean = torch.mean(res_exp);
            # print("mean df after:", resspa_mean - resexp_mean)
            # pdm_exp = pdm_exp + resspa_mean - resexp_mean
            #
            pdm_exp = pdm_exp.fill_diagonal_(0)
            pdm_spa = pdm_spa.fill_diagonal_(0)
            # print("spot device:", spot_data.device)
            # print("x_pixel shape:", x_pixel.shape)
                # prepare data
                # spot_data = prepare_input([spot_data])

            spot_data = spot_data.cuda()
            # spot_data = torch.permute(spot_data, (1, 0, 2))
            # print("spot_data shape:", spot_data.shape)
            # print("spot_data device:", spot_data.device)
                # print("spot_data dtype:", spot_data.dtype)
                # model forward
            _, _, loss = self.model(spot_data[:, :-2], pdm_exp, pdm_spa, lexp, lspa)# pdm) #
            del pdm_spa; del pdm_exp; del spot_data
            # del coor; del pdm; del spot_data

            # update encoding weights
            self.optimizer.zero_grad()
            loss.backward() #retain_graph=True
            clipping_value = 1  # arbitrary value of your choosing
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipping_value)
            self.optimizer.step()
            # print log
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
        # if  (total_loss / (batch_idx + 1)) < 0.15:
        #     break
        # return (total_loss / batch_idx)

    def write_embeddings(self):
        self.model.eval()
        if not os.path.exists("output/"):
            os.makedirs("output/")

        # spot data
        # for i, spot_loader in enumerate(self.test_spot_loaders):
        #     data_name = "stereo" #os.path.basename(self.config.spot_paths[i]).split('.')[0]
        #     fp_em = open('./output/' + data_name + '_embeddings.txt', 'w')
        data_name = os.path.basename(self.config.spot_paths).split('.')[0]
        print("data name:", data_name)

        fp_em = open('./output/' + data_name + '_embeddings.txt', 'w')


        print("ready to write embeddings")
        # torch.save(x , 'x-file') ; x = torch.load('x-file')
        batch_idx = 0
        for data in self.test_loader:
            # spot_data = data.float().unsqueeze(0)

            # for batch_idx, spot_data in enumerate(spot_loader):
                # prepare data
                # print("spot_data type:", type(spot_data))
                # spot_data = prepare_input([spot_data])


            # coor = spot_data[:, :, -2:].squeeze(0)
            # pdm = torch.cdist(coor, coor).cuda()
            # x_pixel = (spot_data[:, :, -2].data.cpu().numpy() )*100
            # y_pixel = (spot_data[:, :, -1].data.cpu().numpy())*100
            # print("x_pixel shape:", x_pixel.shape)
            # prepare data
            # spot_data = prepare_input([spot_data])

            # spot_data = spot_data.cuda()
            spot_data = data.float()
            coor = spot_data[:, -2:]
            pdm_spa = torch.cdist(coor, coor).cuda()
            pdm_exp = torch.cdist(spot_data[:, :-2], spot_data[:, :-2]).cuda()

            n = pdm_exp.shape[0]
            # self.exp = n * 2 * 960 # 2000
            conexp_ratio = 0.07 #0.06 embryo  0.05 rest high-res
            conspa_ratio = 0.07 # 0.06  0.05
            # print("conspa_ratio:", conspa_ratio)
            # self.spa = n * 2 * 63 # 64,65,66,67
            res_exp = pdm_exp.masked_select(~torch.eye(n, dtype=bool).cuda())
            resexp_mean = torch.mean(res_exp)
            resexp_var = torch.var(res_exp)
            res_spa = pdm_spa.masked_select(~torch.eye(n, dtype=bool).cuda())
            resspa_mean = torch.mean(res_spa)
            resspa_var = torch.var(res_spa)
            pdm_exp = pdm_exp * math.sqrt(resspa_var / resexp_var) + resspa_mean - resexp_mean#* math.sqrt(resspa_var/resexp_var)
            res_exp = pdm_exp.masked_select(~torch.eye(n, dtype=bool).cuda())
            key_exp = torch.quantile(res_exp,conexp_ratio,interpolation="nearest") #self.exp/n/(n-1) higher
            lexp = key_exp/math.sqrt((2*104))
            # print("keyexp:", key_exp)
            key_spa = torch.quantile(res_spa, conspa_ratio,interpolation="nearest") #self.spa / n / (n - 1) higher
            # print("keyspa:", key_spa)
            lspa = key_spa /math.sqrt((2 * 104))
            # print("lspa:", lspa)


            # res_exp = pdm_exp.masked_select(~torch.eye(n, dtype=bool).cuda())
            # resexp_mean = torch.mean(res_exp);
            # print("mean df after:", resspa_mean - resexp_mean)
            # pdm_exp = pdm_exp + resspa_mean - resexp_mean
            # print("var ratio:", math.sqrt(resspa_var / resexp_var))
            # print("mean df:", resspa_mean - resexp_mean)
            pdm_exp = pdm_exp.fill_diagonal_(0)
            pdm_spa = pdm_spa.fill_diagonal_(0)

            # print("spot_data shape:", spot_data.shape)
            # print("spot_data device:", spot_data.device)
            # print("spot_data dtype:", spot_data.dtype)
            spot_data = spot_data.cuda()
            # model forward
            spot_embedding, _, _ = self.model(spot_data[:, :-2], pdm_exp, pdm_spa, lexp, lspa)# pdm)  #
            del pdm_spa; del pdm_exp; del spot_data
            # del coor; del pdm; del spot_data

            # x_pixel = spot_data[:, :, -2].data.cpu().numpy() - self.config.xstart
            # y_pixel = spot_data[:, :, -1].data.cpu().numpy() - self.config.ystart
            #     # prepare data
            #     # spot_data = prepare_input([spot_data])
            #
            # spot_data = spot_data.cuda()
            # # model forward
            # _, _, spot_embedding, _ = self.model(spot_data, x_pixel, y_pixel, mask_ratio=0)
            #
            # del x_pixel;del y_pixel;del spot_data

            spot_embedding = torch.squeeze(spot_embedding, 0).data.cpu().numpy()



            # normalization & softmax
            # spot_embedding = spot_embedding / norm(spot_embedding, axis=1, keepdims=True)


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






