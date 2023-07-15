# This file is used to do clustering, refinement and testing of differentially expressed genes given embeddings from high-resolution data.

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/home/.../SGCAST-main/SGCAST-main')
from utils.utils import refine_high
import gseapy
base_path = '/home/.../SGCAST-main/SGCAST-main/output'
ID ='Stereo-seq'

file_name = "/datapath/"+ID+"/"+ID+".h5ad"
spots_embeddings = np.loadtxt(os.path.join(base_path, ID + '_embeddings.txt'))
stage = ID
adata = sc.read_h5ad(file_name)
adata.obsm['embedding'] = np.float32(spots_embeddings) # save memory
n_neighbors = 30 # common setting: 9, 15, 25, 30 
sc.pp.neighbors(adata, use_rep='embedding', n_neighbors=n_neighbors) 
resolution = 1.2  
sc.tl.louvain(adata,  resolution=resolution)  
size = 6
figsize = (3, 3) 
plt.rcParams["figure.figsize"] = figsize
sc.pl.embedding(adata, basis="spatial", color="louvain", s=size, show=False,
                title='SGCAST')  
###########refine
sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=n_neighbors, key_added="pixel")
arr = adata.obsp['pixel_connectivities']
arr = arr.astype('int32') # save memory
pred = adata.obs["louvain"].astype('int32')
refined_pred = refine_high(pred=np.array(pred.tolist()), dis=arr, option=False)
adata.obs["refined_pred"] = refined_pred
adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')
plt.rcParams["figure.figsize"] = figsize
sc.pl.embedding(adata, basis="spatial", color="refined_pred", s=size, show=False,
                title='SGCAST')  
plt.savefig(os.path.join(base_path, stage  + "_plt" + str(int(figsize[0])) + "_" + str(
    int(figsize[1])) + "_50pc.png"), bbox_inches="tight",dpi=600)  
plt.axis('off')
plt.close()


num_cluster = len(np.unique(adata.obs["refined_pred"]))

# find differentially expressed genes given above clusters.
if 'log1p' in list(adata.uns):
    del adata.uns['log1p']


adata.obs["refined_pred"]=adata.obs["refined_pred"].astype(str)
sc.tl.rank_genes_groups(adata, 'refined_pred', method='wilcoxon', key_added="wilcoxon")  

# generate dotplot
sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, key="wilcoxon", groupby='refined_pred',save=ID+'dotplot.png')

DEs = []
df = sc.get.rank_genes_groups_df(adata, key='wilcoxon', group=None)
groups = df['group'].unique()

# select top 2 genes for each identified cluster
for group in groups:
    dfg = df.loc[df['group'] == group]
    names = dfg['names'][:2].tolist()
    for i in names:
        DEs.append(i)
        
print(ID + '_DEs are:', DEs)

# flip and rotate the plot
adata.obsm["spatial"] = adata.obsm["spatial"][:, [1, 0]]
adata.obsm["spatial"][:, 0] = adata.obsm["spatial"][:, 0]*(-1)

adata.obsm["spatial"] = adata.obsm["spatial"][:, [1, 0]]

# plot the expression of each DE
for DE in DEs:
    plot_gene = DE
    spot_size = 20
    plt.rcParams["figure.figsize"] = (3,3)
    sc.pl.spatial(adata, color=plot_gene, img_key=None, spot_size=spot_size, size=1.5,
                  title='plot_' + plot_gene, vmin='p10', vmax='p99.2')  #vmin=0 'p99.2'
    plt.savefig(ID+"marker_gene_processed" + plot_gene + '_size_' + str(int(spot_size))+'p100_refine', bbox_inches="tight",
                dpi=600)  # pred.pdf avremb
    plt.axis('off')
    plt.close()

print('Finished')
