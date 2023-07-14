# This is the dataloader, which generates mini-batch for training and writing results.

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
#     if  np.sum(adata.var['highly_variable'])<3000:
#         sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)
    # If the number of hvg in h5ad file is smaller than 3000, the above method is used to select top 3000 genes.
        

    adata = adata[:, adata.var['highly_variable']]
    if issparse(adata.X):
        data = adata.X.A
    else:
        data = adata.X
    pca.fit(data)
    data = pca.transform(data) # get principal components
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
            sample = np.array(self.data[index])  
            return sample

    def __len__(self):
        return self.data.shape[0]


class PrepareDataloader():
    def __init__(self, config):
        self.config = config
        # hardware constraint
        kwargs = {'num_workers': 0, 'pin_memory': False} 

        self.sample_num = 0
        self.batch_size = 0

        trainset = Dataloader(True, config.spot_paths, config.nfeat, config.seed)
        self.sample_num += len(trainset)
        # adjust the size of mini-batch in case that the size of the last batch is too small which may lead biased results.
        if (self.sample_num % config.batch_size) < (0.1*config.batch_size):
            self.batch_size = config.batch_size + math.ceil((self.sample_num % config.batch_size)/(self.sample_num//config.batch_size))
        else:
            self.batch_size = config.batch_size

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=
            self.batch_size, shuffle=True,   **kwargs) # mini-batches used for training are shuffled in this step.

        testset = Dataloader(False, config.spot_paths, config.nfeat, config.seed)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=
            self.batch_size, shuffle=False, **kwargs) # mini-batches used for writing results are not shuffled.



        self.train_loader = train_loader
        self.test_loader = test_loader
        print("sample_num :", self.sample_num)
        print("batch_size :", self.batch_size)

    def getloader(self):
        return self.train_loader, self.test_loader,  math.ceil(self.sample_num/self.batch_size)



if __name__ == "__main__":
    config = Config()
    spot_data = Dataloader(True, config.spot_paths)
    print('spot data:', spot_data.input_size,  len(spot_data.data))

    loader = PrepareDataloader(config)
    train_loader, test_loader, num_iter = loader.getloader()
    print('batch size:', loader.batch_size)
