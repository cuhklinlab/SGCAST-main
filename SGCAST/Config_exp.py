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
            self.spot_paths = ["/lustre/project/Stat/s1155077016/spatial_data/151507/151507.h5ad",
             "/lustre/project/Stat/s1155077016/spatial_data/151508/151508.h5ad",
             "/lustre/project/Stat/s1155077016/spatial_data/151509/151509.h5ad",
             "/lustre/project/Stat/s1155077016/spatial_data/151510/151510.h5ad",
             "/lustre/project/Stat/s1155077016/spatial_data/151669/151669.h5ad",
             "/lustre/project/Stat/s1155077016/spatial_data/151670/151670.h5ad",
             "/lustre/project/Stat/s1155077016/spatial_data/151671/151671.h5ad",
             "/lustre/project/Stat/s1155077016/spatial_data/151672/151672.h5ad",
             "/lustre/project/Stat/s1155077016/spatial_data/151673/151673.h5ad",
             "/lustre/project/Stat/s1155077016/spatial_data/151674/151674.h5ad",
             "/lustre/project/Stat/s1155077016/spatial_data/151675/151675.h5ad",
             "/lustre/project/Stat/s1155077016/spatial_data/151676/151676.h5ad"]




            # Training config
            self.nfeat = 30 #  mclust has problem on high-dimension embedding of expression layer only so we decrease it to 30.
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















