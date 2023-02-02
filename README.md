# SGCAST

SGCAST is a simple and efficient framework for identifying spatial domains by encoding spatial location and gene expression information to latent embeddings. SGCAST  can not only conduct accurate identification of spatial domains but also finish further extraction of
spatially expressed genes, which are critical for understanding tissue
organization and biological functions. SGCAST transforms information on gene expression and the position of spots into two adjacency matrices and further adopts a symmetric graph convolutional auto-encoder to integrate them.


## Tutorials

+ A data preprocessing tutorial for each data used in the paper is demonstrated here: [link](https://github.com/cuhklinlab/SGCAST/blob/main/SGCAST/data_preprocess.ipynb)
+ Tutorial for clustering: 
    + DLPFC dataset [link](https://github.com/cuhklinlab/SGCAST/blob/main/SGCAST/DLPFC_clustering.ipynb)
    + high-resolution data [link](https://github.com/cuhklinlab/SGCAST/blob/main/SGCAST/high-res_clustering.ipynb)


## Installation:
 
SGCAST is implemented in the pytorch framework. Please run SGCAST on CUDA. The following packages are required to be able to run everything in this repository (included are the versions we used):

```bash
anndata==0.8.0 
bokeh==2.4.2
h5py==3.6.0
imageio==2.9.0   
matplotlib==3.5.1 
numpy==1.21.5
pandas==1.4.2
python-igraph==0.9.6
python-louvain==0.15 
scanpy==1.9.1 
scikit-image==0.19.2
scikit-learn==1.1.3
scipy==1.7.3
seaborn==0.11.2
pytorch==1.12.1
torch-geometric==1.7.0 
torchvision==0.13.1
tqdm==4.64.0
umap-learn==0.5.3 
```
### Install package mclust 5.4.10 in R (used for the mclust clustering)

(**Recommended**) Using python virutal environment with conda（<https://anaconda.org/>）
```shell
conda create -n SGCAST python=3.9 pip
conda activate SGCAST
pip install -r requirements.txt
```



## Running SGCAST

Edit `Config.py` according to the data input (See Arguments section for more details).

In terminal, run

```
python main.py
```

The output will be saved in `./output` folder.


## Arguments

The script `config.py` indicate the arguments for scJoint, which needs to be modified according to the data.


### Training config

+ `use_cuda`: Whether GPU is used
+ `threads`: Number of threads used (set as 1 by default)

+ `batch_size`: Batch size (set as 256 by default)
+ `lr_start`: Initial learning rate 
+ `lr_stage3`: Learning rate for stage 3
+ `lr_decay_epoch`: Number of epochs learning rate decay
+ `nfeat`: Dimension of input of auto-encoder
+ `nhid`: Dimension of hidden layer of auto-encoder
+ `nemb`: Dimension of latent embedding of auto-encoder
+ `seed`: seed to be used

The configuration we used in our paper can be found in [link](https://github.com/cuhklinlab/SGCAST/blob/main/SGCAST/Config.py).



## Reference
