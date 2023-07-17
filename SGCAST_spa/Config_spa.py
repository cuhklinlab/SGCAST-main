import torch
import os


class Config(object):
    def __init__(self):
        DT = 'DLPFC'  # 'DLPFC','stereo_mouse_bulb', 'seq-scope', 'stereo_embryo' change Data type before running.
        self.threads = 1
        self.device = torch.device('cuda:0')

        if DT == 'DLPFC':
            # DT info
            self.spot_paths = ["../data/DLPFC/151507/151507.h5ad",
             "../data/DLPFC/151508/151508.h5ad",
             "../data/DLPFC/151509/151509.h5ad",
             "../data/DLPFC/151510/151510.h5ad",
             "../data/DLPFC/151669/151669.h5ad",
             "../data/DLPFC/151670/151670.h5ad",
             "../data/DLPFC/151671/151671.h5ad",
             "../data/DLPFC/151672/151672.h5ad",
             "../data/DLPFC/151673/151673.h5ad",
             "../data/DLPFC/151674/151674.h5ad",
             "../data/DLPFC/151675/151675.h5ad",
             "../data/DLPFC/151676/151676.h5ad"]
            
            self.train_conexp_ratio = 0.07 
            self.train_conspa_ratio = 0.07
            self.test_conexp_ratio = 0.07 
            self.test_conspa_ratio = 0.07 
            
 
                     
        # Training config
        self.nfeat = 50 
        self.nhid = 50
        self.nemb = 50
        self.batch_size = 2000  
        self.lr_start = 0.2 
        self.lr_times = 2
        self.lr_decay_epoch = 80 
        self.epochs_stage =100 
        self.seed = 2022
        self.checkpoint = ''

















