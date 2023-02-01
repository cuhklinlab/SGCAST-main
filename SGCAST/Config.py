import torch
import os


class Config(object):
    def __init__(self):
        DT = 'DFPLC'  # 'stereo_mouse_bulb', 'seq-scope', 'stereo_embryo' change Data type before running.
        self.threads = 1


        if self.use_cuda:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cuda:0')

        if DT == 'DFPLC':
            # DT info
            self.spot_paths = ["/datapath/151507/151507.h5ad",
             "/datapath/151508/151508.h5ad",
             "/datapath/151509/151509.h5ad",
             "/datapath/151510/151510.h5ad",
             "/datapath/151669/151669.h5ad",
             "/datapath/151670/151670.h5ad",
             "/datapath/151671/151671.h5ad",
             "/datapath/151672/151672.h5ad",
             "/datapath/151673/151673.h5ad",
             "/datapath/151674/151674.h5ad",
             "/datapath/151675/151675.h5ad",
             "/datapath/151676/151676.h5ad"]
            
            self.train_conexp_ratio = 0.07 
            self.train_conspa_ratio = 0.07
            self.test_conexp_ratio = 0.07 
            self.test_conspa_ratio = 0.07 
            
          if DT == 'stereo_mouse_bulb': 
            self.spot_paths = ["/lustre/project/Stat/s1155077016/spatial_data/Stereo-seq/Stereo-seq.h5ad"]
            
            self.train_conexp_ratio = 0.02 
            self.train_conspa_ratio = 0.02
            self.test_conexp_ratio = 0.07 
            self.test_conspa_ratio = 0.07 
            
          if DT == 'seq-scope': 
            self.spot_paths = ['/lustre/project/Stat/s1155077016/spatial_data/scope_colon/scope_colon.h5ad']
            
            self.train_conexp_ratio = 0.04
            self.train_conspa_ratio = 0.04
            self.test_conexp_ratio = 0.07 
            self.test_conspa_ratio = 0.07 
            
          if DT == 'stereo_embryo': 
            self.spot_paths = ['/datapath/stereo_E12/E12_FP200000587TR_D5_bin50.h5ad',
             '/datapath/stereo_E14/E14_SS200000108BR_C3C4_bin50.h5ad',
             '/datapath/stereo_E16/E16_SS200000124BR_D1D2_bin50.h5ad']
            
            self.train_conexp_ratio = 0.02 
            self.train_conspa_ratio = 0.02
            self.test_conexp_ratio = 0.06 
            self.test_conspa_ratio = 0.06
        # For low-resolution data, it is fine to set all $/tau$ to 0.07, which is default setting in SGCAST. 
        # For high-resolution data, when training, low $/tau$ is preferred for smaller loss and faster convergence.
        # when writing results, using common 0.07 is okay. For embryo data, especially E16.5, the spots number is too large, we pick a bit smaller $/tau$ 0.06 .
                     
        # Training config
        self.nfeat = 50 
        self.nhid = 50
        self.nemb = 50
        self.batch_size = 2000  
        self.lr_start = 0.2 
        self.lr_times = 2
        self.lr_decay_epoch = 80 
        self.epochs_stage =100 
        self.momentum = 0.9
        self.seed = 2022
        self.checkpoint = ''

















