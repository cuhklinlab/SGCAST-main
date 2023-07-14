# The file is used for simplified version of SGCAST with the layer that captures gene expression similarity only.
import torch
import os


class Config(object):
    def __init__(self):
        DB = 'DLPFC'
        self.use_cuda = True
        self.threads = 1
        self.device = torch.device('cuda:0')

        if DB == 'DLPFC':
            # DB info
            self.spot_paths = ["/datapath/151507.h5ad",
             "/datapath/151508.h5ad",
             "/datapath/151509.h5ad",
             "/datapath/151510.h5ad",
             "/datapath/151669.h5ad",
             "/datapath/151670.h5ad",
             "/datapath/151671.h5ad",
             "/datapath/151672.h5ad",
             "/datapath/151673.h5ad",
             "/datapath/151674.h5ad",
             "/datapath/151675.h5ad",
             "/datapath/151676.h5ad"]




            # Training config
            self.nfeat = 30 #  mclust has problem on high-dimension embedding of expression layer only so we decrease it from 50 to 30.
            self.nhid = 30
            self.nemb = 30
            self.batch_size = 2000 
            self.lr_start = 0.2 
            self.lr_times = 2
            self.lr_decay_epoch = 80 
            self.epochs_stage =100 
            self.seed = 2022
            self.checkpoint = ''
            self.train_conexp_ratio = 0.07 
            self.train_conspa_ratio = 0.07
            self.test_conexp_ratio = 0.07 
            self.test_conspa_ratio = 0.07 















