# SGCAST

## Overview
SGCAST is a simple and efficient auto-encoder framework to identify spatial domains. SGCAST adopts a Symmetric Graph Convolutional Auto-encoder to learn aggregated latent embeddings via integrating the gene expression similarity and the proximity of the spatial spots. SGCAST employs a mini-batch training strategy, which makes SGCAST memory efficient and scalable to high-resolution spatial transcriptomic data with a large number of spots. The latent embeddings given by SGCAST can be used for clustering, data visualization, trajectory inference, and other downstream analyses.

![Workflow](./Figure/Workflow.png)

## Tutorials

+ Data preprocessing tutorial is demonstrated here: [link](https://github.com/cuhklinlab/SGCAST/blob/main/data_preprocess.ipynb)
+ Tutorial for clustering: 
    + DLPFC dataset [link](https://github.com/cuhklinlab/SGCAST/blob/main/DLPFC_clustering.ipynb)
    + high-resolution data [link](https://github.com/cuhklinlab/SGCAST/blob/main/high-res_clustering.ipynb)


SGCAST is implemented in the pytorch framework. Please run SGCAST on CUDA. SGCAST can be obtained by clonning the github repository:

### Installation
#### Start by grabbing source codes:
```bash
git clone https://github.com/cuhklinlab/SGCAST.git
cd SGCAST
```

(Recommended) Using python virtual environment with [`conda`](https://anaconda.org/)

```bash
conda env create -f environment.yaml
conda activate sgcast_env
```

### Install package mclust 5.4.10 in R (used for the mclust clustering)



## Running SGCAST on DLPFC from 10x Visium.
``` cd /home/.../SGCAST/SGCAST```
```python
import os 
from main import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc

data_path = "../data/DLPFC" #### to your path
data_name = '151673' #### project name
save_path = "../Results" #### save path
n_domains = 7 ###### the number of spatial domains.
```

Edit `Config.py` according to the data input (See Arguments section for more details).

In terminal, run

```
python main.py
```

The output will be saved in `./output` folder.

+ #### SGCAST on DLPFC from 10x Visium.
First, ``` cd /home/.../SGCAST/SGCAST```
```python
import os 
from main import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc

data_path = "../data/DLPFC" #### to your path
data_name = '151673' #### project name
save_path = "../Results" #### save path
n_domains = 7 ###### the number of spatial domains.

deepen = run(save_path = save_path,
	task = "Identify_Domain", #### DeepST includes two tasks, one is "Identify_Domain" and the other is "Integration"
	pre_epochs = 800, ####  choose the number of training
	epochs = 1000, #### choose the number of training
	use_gpu = True)
###### Read in 10x Visium data, or user can read in themselves.
adata = deepen._get_adata(platform="Visium", data_path=data_path, data_name=data_name)




###### Build graphs. "distType" includes "KDTree", "BallTree", "kneighbors_graph", "Radius", etc., see adj.py


###### Enhanced data preprocessing


###### Training models

###### SGCAST outputs
adata.obsm["DeepST_embed"] = deepst_embed

###### Define the number of space domains, and the model can also be customized. If it is a model custom priori = False.
adata = deepen._get_cluster_data(adata, n_domains=n_domains, priori = True)

###### Spatial localization map of the spatial domain
sc.pl.spatial(adata, color='DeepST_refine_domain', frameon = False, spot_size=150)
plt.savefig(os.path.join(save_path, f'{data_name}_domains.pdf'), bbox_inches='tight', dpi=300)
```


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
