import torch
import torch.utils.data as data
import numpy as np
import os
import os.path
import random
import scanpy as sc
from scipy.sparse import issparse
import math
from sklearn.decomposition import PCA
from Config import Config

def Anndata_reader(file_name,dim,seed):
    pca = PCA(n_components=dim, svd_solver='arpack',random_state=seed)
    adata = sc.read_h5ad(file_name)
    if  np.sum(adata.var['highly_variable'])<3000:
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)

    adata = adata[:, adata.var['highly_variable']]
    if issparse(adata.X):
        data = adata.X.A
    else:
        data = adata.X
    pca.fit(data)
    data = pca.transform(data)
    x_pixel = np.array([adata.obsm['spatial'][:, 0].tolist()]).T
    y_pixel = np.array([adata.obsm['spatial'][:, 1].tolist()]).T
    print('ST data shape:', data.shape)
    return data, data.shape[1], data.shape[0], x_pixel, y_pixel



def read_from_file(data_path,dim,seed):
    data = None
    input_size = 0
    data, input_size, sample_num, x_pixel, y_pixel = Anndata_reader(data_path,dim,seed)
    data = np.concatenate((data, x_pixel,y_pixel), 1)
    return input_size, sample_num,  data




class Dataloader(data.Dataset):
    def __init__(self, train=True, data_path=None, dim=50, seed=2022):
        self.train = train
        self.input_size, self.sample_num, self.data = read_from_file(
            data_path, dim, seed)
    def __getitem__(self, index):
        if self.train:
            sample = np.array(self.data[index])  
            return sample
        else:
            sample = np.array(self.data[index])
            return sample

    def __len__(self):
        return self.data.shape[0]


class PrepareDataloader():
    def __init__(self, config):
        self.config = config
        kwargs = {'num_workers': 0, 'pin_memory': False} 
        self.sample_num = 0

        trainset = Dataloader(True, config.spot_paths, config.nfeat, config.seed)
        self.sample_num += len(trainset)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=
            config.batch_size, shuffle=True,   **kwargs) 

        testset = Dataloader(False, config.spot_paths, config.nfeat, config.seed)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=
            config.batch_size, shuffle=False, **kwargs)

        self.train_loader = train_loader
        self.test_loader = test_loader
        print("sample_num :", self.sample_num)

    def getloader(self):
        return self.train_loader, self.test_loader,  math.ceil(self.sample_num/self.config.batch_size)



if __name__ == "__main__":
    config = Config()
    spot_data = Dataloader(True, config.spot_paths)
    print('spot data:', spot_data.input_size,  len(spot_data.data))


    train_loader, test_loader, num_iter = PrepareDataloader(config).getloader()

