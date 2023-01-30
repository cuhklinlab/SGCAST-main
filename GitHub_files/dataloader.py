# import glob
import torch
import torch.utils.data as data
import numpy as np
import os
import os.path
# import cv2
import random
# import csv
import scanpy as sc
from scipy.sparse import issparse
import math
from sklearn.decomposition import PCA
from Config import Config

# how to read from large matrix
# 1. use sparse matrix
# 2. use Anndata

# def sparse_mat_reader(file_name):
#     data = scipy.sparse.load_npz(file_name)
#     print('Read db:', file_name, ' shape:', data.shape)
#     return data, data.shape[1], data.shape[0]

#/lustre/project/Stat/s1155077016/spatial_data/Stereo-seq/adata.h5ad
def Anndata_reader(file_name,dim,seed):
    pca = PCA(n_components=dim, svd_solver='arpack',random_state=seed)
    #    random_state=2022 svd_solver='full'
    adata = sc.read_h5ad(file_name)
    if  np.sum(adata.var['highly_variable'])<3000:
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)

    # sc.pp.highly_variable_genes(adata, flavor="cell_ranger", n_top_genes=3000)
    adata = adata[:, adata.var['highly_variable']]
    if issparse(adata.X):
        data = adata.X.A
    else:
        data = adata.X
    pca.fit(data)
    data = pca.transform(data)
    # data = data-np.sum(data)/data.shape[1]/data.shape[0]### centerning
    x_pixel = np.array([adata.obsm['spatial'][:, 0].tolist()]).T
    y_pixel = np.array([adata.obsm['spatial'][:, 1].tolist()]).T
    # x_pixel = np.array([adata.obs["x_pixel"].values]).T
    # y_pixel = np.array([adata.obs["y_pixel"].values]).T
    print('ST data shape:', data.shape)
    return data, data.shape[1], data.shape[0], x_pixel, y_pixel


# def load_labels(label_file):  # please run parsing_label.py first to get the numerical label file (.txt)
#     return np.loadtxt(label_file)


def read_from_file(data_path,dim,seed):
    #data_path = os.path.join(os.path.realpath('.'), data_path)
    data = None
    input_size = 0
    data, input_size, sample_num, x_pixel, y_pixel = Anndata_reader(data_path,dim,seed)
    #coord = np .concatenate((np.array([x_pixel.values]).T,np.array([y_pixel.values]).T),1)
    data = np.concatenate((data, x_pixel,y_pixel), 1)
    return input_size, sample_num,  data




class Dataloader(data.Dataset):
    def __init__(self, train=True, data_path=None, dim=50, seed=2022):
        self.train = train
        self.input_size, self.sample_num, self.data = read_from_file(
            data_path, dim, seed)
############ 如果切块取点， 需要x_pixel, y_pixel 符合grid取值范围。 这样检查是否满足该条件会消耗大量时间，有无efficient算法？
    def __getitem__(self, index):
        if self.train:
            # get atac data
            # gap_x = 50 ; gap_y =50
            # min_x = min(x_pixel) ; max_x = max(x_pixel)
            # min_y = min(y_pixel) ; max_y = max(y_pixel)
            # xgrid = np.arange(min_x, max_x, 50)
            # ygrid = np.arange(min_y, max_y, 50)
            #
            # idx = self.x_pixel < min_x + index* gap_x & self.x_pixel < min_x + (index-1)* gap_x
            #     & self.y_pixel < min_y + index* gap_y & self.y_pixel < min_x + (indey-1)* gap_y
            # rand_idx = random.randint(0, self.sample_num - 1)
            sample = np.array(self.data[index])  #.todense() 从稀疏矩阵转到普通矩阵
            #in_data = (sample > 0).astype(np.float)  # binarize data
            # if self.proteins is not None:
            #     sample_protein = np.array(self.proteins[rand_idx].todense())
            #     in_data = np.concatenate((in_data, sample_protein), 1)
            # in_data = in_data.reshape((1, self.input_size))
            return sample

        else:

            #directly select in data_matrix
            sample = np.array(self.data[index])
            # in_data = (sample > 0).astype(np.float)  # binarize data
            # if self.proteins is not None:
            #     sample_protein = np.array(self.proteins[index].todense())
            #     in_data = np.concatenate((in_data, sample_protein), 1)

            # in_data = in_data.reshape((1, self.input_size))
            return sample

    def __len__(self):
        return self.data.shape[0]


class PrepareDataloader():
    def __init__(self, config):
        self.config = config
        # hardware constraint
        kwargs = {'num_workers': 0, 'pin_memory': False} # 1 True #, 'drop_last': True

        # load RNA

        self.sample_num = 0


        trainset = Dataloader(True, config.spot_paths, config.nfeat, config.seed)
        self.sample_num += len(trainset)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=
            config.batch_size, shuffle=True,   **kwargs) #drop_last=True,

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

