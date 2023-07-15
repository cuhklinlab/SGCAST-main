# This file is used to do clustering on the embedding from DLPFC datasets (stored in '/SGCAST_path/output') and calculate ARI score afterwards.

import sys
sys.path.append('/home/.../SGCAST-main/SGCAST-main')
from utils.utils import refine
import os
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score

ARIset=[]
base_path = '/home/.../SGCAST-main/SGCAST-main/output'
IDs = ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676']
for ID in IDs:
    file_name = "/data_path/"+ID+".h5ad" 
    adata = sc.read_h5ad(file_name)

    spots_embeddings = np.loadtxt(os.path.join(base_path, ID+'_embeddings.txt'))
    adata.obsm['embedding'] = np.float32(spots_embeddings)
    random_seed=2022
    np.random.seed(random_seed)
    os.environ['R_HOME'] = "/Rpath/lib/R"
    os.environ['R_USER'] = '/pythonpath/python3.9/site-packages/rpy2'
    import rpy2.robjects as robjects

    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    # run mclust
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    num_cluster = 7; modelNames="EEE" 
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm['embedding']), num_cluster, modelNames) 
    mclust_res = np.array(res[-2]) 
    

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')


    obs_df = adata.obs.dropna()
    # calculate ARI score
    ARI = adjusted_rand_score(obs_df['mclust'], obs_df["Ground Truth"])
    ARI
    
    # refine clustering result
    from scipy.spatial.distance import cdist
    xarr=np.array([adata.obs['array_row'].tolist()]).T
    yarr=np.array([adata.obs['array_col'].tolist()]).T
    am = np.concatenate((xarr,yarr), 1)
    arr = cdist(am, am)
    
    refined_pred=refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["mclust"].tolist(), dis=arr, shape="hexagon")
    adata.obs["refined_pred"]=refined_pred
    adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')

    adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
    obs_df = adata.obs.dropna()
    # calculate ARI score for refined clustering result
    ARI_ref = adjusted_rand_score(obs_df['refined_pred'], obs_df["Ground Truth"])
    ARI_ref

    ARIset.append(ARI_ref)
    
    # save results, one example is displayed in the tutorial.
    plt.rcParams["figure.figsize"] = (6, 3)
   
    sc.pl.spatial(adata, color=["refined_pred", "Ground Truth"], title=['SGCAST(ARI=%.2f)' % ARI_ref,
                                                                        "Manual annotation"])  # adata.obs["true"] = adata_processed.obs["true"]
    plt.savefig(os.path.join(base_path, ID+'_results.png'), bbox_inches="tight", dpi=600)  # pred.png avremb
    plt.axis('off')
    plt.close()
"""
    sc.pp.neighbors(adata, use_rep='embedding')
    sc.tl.umap(adata)

    used_adata = adata[np.isin(adata.obs_names.values,obs_df.index.values),] #used_adata = adata[adata.obs['Ground Truth'] != 'nan',]
    used_adata
    sc.tl.paga(used_adata, groups='Ground Truth')
    plt.rcParams["figure.figsize"] = (4, 3)
    sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=False, size=20,
                       title=ID + '_SGCAST', legend_fontoutline=2, show=False, save=ID + '_com.png')
"""

print('ARI SET: ', ARIset)
