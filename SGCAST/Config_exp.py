import torch
import os


class Config(object):
    def __init__(self):
        DB = 'stereo'
        self.use_cuda = True
        self.threads = 1

        if not self.use_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')

        if DB == 'stereo':
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

            # ["/lustre/project/Stat/s1155077016/spatial_data/Stereo-seq/Stereo-seq.h5ad",
            #  '/lustre/project/Stat/s1155077016/spatial_data/scope_colon/scope_colon.h5ad',
            #  '/lustre/project/Stat/s1155077016/spatial_data/scope_colon_2112_SW4X/scope_colon_2112_SW4X.h5ad',
            #  '/lustre/project/Stat/s1155077016/spatial_data/stereo_E16/E16_SS200000124BR_D1D2_bin50.h5ad']
            # ['/lustre/project/Stat/s1155077016/spatial_data/stereo_E16/E16_SS200000124BR_D1D2_bin50.h5ad']

            # ["/lustre/project/Stat/s1155077016/spatial_data/Stereo-seq/Stereo-seq.h5ad",
            # "/lustre/project/Stat/s1155077016/spatial_data/slidev2_mouse_OlfactoryBulb/slidev2_mouse_OlfactoryBulb.h5ad"]

            # ['/lustre/project/Stat/s1155077016/spatial_data/scope_colon/scope_colon.h5ad',
            #  '/lustre/project/Stat/s1155077016/spatial_data/scope_liverTD/scope_liverTD.h5ad',
            #  '/lustre/project/Stat/s1155077016/spatial_data/scope_liver/scope_liver.h5ad',
            #  '/lustre/project/Stat/s1155077016/spatial_data/scope_colon_2112_SW4X/scope_colon_2112_SW4X.h5ad']

            # ["/lustre/project/Stat/s1155077016/spatial_data/151507/151507.h5ad",
            #  "/lustre/project/Stat/s1155077016/spatial_data/151508/151508.h5ad",
            #  "/lustre/project/Stat/s1155077016/spatial_data/151509/151509.h5ad",
            #  "/lustre/project/Stat/s1155077016/spatial_data/151510/151510.h5ad",
            #  "/lustre/project/Stat/s1155077016/spatial_data/151669/151669.h5ad",
            #  "/lustre/project/Stat/s1155077016/spatial_data/151670/151670.h5ad",
            #  "/lustre/project/Stat/s1155077016/spatial_data/151671/151671.h5ad",
            #  "/lustre/project/Stat/s1155077016/spatial_data/151672/151672.h5ad",
            #  "/lustre/project/Stat/s1155077016/spatial_data/151673/151673.h5ad",
            #  "/lustre/project/Stat/s1155077016/spatial_data/151674/151674.h5ad",
            #  "/lustre/project/Stat/s1155077016/spatial_data/151675/151675.h5ad",
            #  "/lustre/project/Stat/s1155077016/spatial_data/151676/151676.h5ad"]
            # ['/lustre/project/Stat/s1155077016/spatial_data/Puck_200127_15/Puck_200127_15.h5ad',
            #  '/lustre/project/Stat/s1155077016/spatial_data/Puck_190921_21/Puck_190921_21.h5ad',
            #  '/lustre/project/Stat/s1155077016/spatial_data/Puck_191204_01/Puck_191204_01.h5ad',
            #  '/lustre/project/Stat/s1155077016/spatial_data/Puck_200115_08/Puck_200115_08.h5ad']
            # ['/lustre/project/Stat/s1155077016/spatial_data/stereo_E9/E9_DP8400015286BL_D2_bin50.h5ad',
            # '/lustre/project/Stat/s1155077016/spatial_data/stereo_E10/E10_FP200000587TR_E4_bin50.h5ad',
            # '/lustre/project/Stat/s1155077016/spatial_data/stereo_E11/E11_FP200000554BL_B2_bin50.h5ad',
            # '/lustre/project/Stat/s1155077016/spatial_data/stereo_E12/E12_FP200000587TR_D5_bin50.h5ad',
            # '/lustre/project/Stat/s1155077016/spatial_data/stereo_E13/E13_SS200000124BR_C3C4_bin50.h5ad',
            # '/lustre/project/Stat/s1155077016/spatial_data/stereo_E14/E14_SS200000108BR_C3C4_bin50.h5ad',
            #
            # '/lustre/project/Stat/s1155077016/spatial_data/stereo_E15/E15_SS200000108BR_E1E2_bin50.h5ad',
            # '/lustre/project/Stat/s1155077016/spatial_data/stereo_E16/E16_SS200000124BR_D1D2_bin50.h5ad']
            # '/lustre/project/Stat/s1155077016/spatial_data/stereo_E16/stereo_E16.h5ad'
            # '/lustre/project/Stat/s1155077016/spatial_data/Puck_200127_15/Puck_200127_15.h5ad'

            # '/lustre/project/Stat/s1155077016/spatial_data/seq-scope_colon/scope_colon.h5ad'
            # '/lustre/project/Stat/s1155077016/spatial_data/seq-scope_liverTD/scope_liverTD.h5ad'
            # '/lustre/project/Stat/s1155077016/spatial_data/seq-scope_liver/scope_liver.h5ad'
            # 'Puck_200127_15' 'Puck_190921_21' 'Puck_191204_01' 'Puck_200115_08'
            # '/lustre/project/Stat/s1155077016/spatial_data/Puck_200127_15/Puck_200127_15.h5ad'
            # "/lustre/project/Stat/s1155077016/spatial_data/151676/151676.h5ad"
             # "/lustre/project/Stat/s1155077016/spatial_data/Slide-seqV2_mouse_OlfactoryBulb/slidev2.h5ad"
            # '/lustre/project/Stat/s1155077016/spatial_data/seq-scope/scope.h5ad'
            # "/lustre/project/Stat/s1155077016/spatial_data/Stereo-seq/stereo.h5ad"
            # "/lustre/project/Stat/s1155077016/spatial_data/151674/151674.h5ad"


            # Training config
            self.nfeat = 30 #30
            self.nhid = 30
            self.nemb = 30
            # self.exp = self.batch_size*1 # 2000
            # self.spa = self.batch_size*2*6   #200000
            # self.init = "louvain"
            self.batch_size = 2000  #### 2000
            self.lr_start = 0.2 #0.12
            self.lr_times = 2
            self.lr_decay_epoch = 80 #500
            self.epochs_stage =100 #2000
            # self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 2022
            self.checkpoint = ''
            self.train_conexp_ratio = 0.07 #0.02 embryo
            self.train_conspa_ratio = 0.07#0.02 embryo 0.02 stereo 0.04 colon
            self.test_conexp_ratio = 0.07 #0.06 embryo
            self.test_conspa_ratio = 0.07 #0.06 embryo
            # self.exp_nei = 0.25 #0.05
            # self.spa_nei = 0.064 #0.05


            # lr_time =1
            # different lr schedule













