# SGCAST

SGCAST is a simple and efficient auto-encoder framework to identify spatial domains. SGCAST adopts a Symmetric Graph Convolutional Auto-encoder to learn aggregated latent embeddings via integrating the gene expression similarity and the proximity of the spatial spots. SGCAST employs a mini-batch training strategy, which makes SGCAST memory efficient and scalable to high-resolution spatial transcriptomic data with a large number of spots. The latent embeddings given by SGCAST can be used for clustering, data visualization, trajectory inference, and other downstream analyses.


## Tutorials

+ Data preprocessing tutorial and correcponding download address is demonstrated here: [link](https://github.com/cuhklinlab/SGCAST/blob/main/SGCAST/data_preprocess.ipynb)
+ Tutorial for clustering: 
    + DLPFC dataset [link](https://github.com/cuhklinlab/SGCAST/blob/main/SGCAST/DLPFC_clustering.ipynb)
    + high-resolution data [link](https://github.com/cuhklinlab/SGCAST/blob/main/SGCAST/high-res_clustering.ipynb)


## Installation:
 
SGCAST is implemented in the pytorch framework. Please run SGCAST on CUDA. SGCAST can be obtained by clonning the github repository:

```
git clone https://github.com/cuhklinlab/SGCAST.git
```

### Install package mclust 5.4.10 in R (used for the mclust clustering)

## Versions of the software:
- System: Anaconda
- gcc: 7.5.0
- Python: 3.9.12
- Python packages: anndata=0.8.0, joblib=1.1.0,matplotlib=3.5.1,numpy=1.21.5,pandas=1.4.2,rpy2=3.5.3,scanpy=1.9.1,scikit-learn=1.1.3,scipy=1.7.3,torch=1.12.1



## Running SGCAST

Edit `Config.py` according to the data input (See Arguments section for more details).

In terminal, run

```
python main.py
```

The output will be saved in `./output` folder.


## Arguments

The script `Config.py` indicate the arguments for SGCAST, which needs to be modified according to the data.


### Training config

+ `threads`: Number of threads used (set as 1 by default)
+ `spot_paths`: paths of input data (can be multiple paths, which will be trained under the same configuration.)
+ `batch_size`: Batch size (set as 2000 by default)
+ `lr_start`: Initial learning rate (set as 0.2 by default)
+ `lr_decay_epoch`: Number of epochs learning rate decay
+ `nfeat`: Dimension of input of auto-encoder
+ `nhid`: Dimension of hidden layer of auto-encoder
+ `nemb`: Dimension of latent embedding of auto-encoder
+ `seed`: seed to be used
+ `train_conexp_ratio`: tau for expression layer in training 
+ `train_conspa_ratio`: tau for spatial layer in training 
+ `train_conexp_ratio`: tau for expression layer when writing results
+ `train_conexp_ratio`: tau for spatial layer when writing results


The configuration we used in our paper can be found in [link](https://github.com/cuhklinlab/SGCAST/blob/main/SGCAST/Config.py).


## Reference
