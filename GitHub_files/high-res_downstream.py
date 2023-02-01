import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
sys.path.append('/lustre/project/Stat/s1155077016/SGCAST/SGCAST/SGCAST')
from utils.utils import refine_high
import gseapy
np.random.seed(2022)
base_path = '/lustre/project/Stat/s1155077016/SGCAST/SGCAST/SGCAST/output'
ID ='Stereo-seq'

file_name = "/lustre/project/Stat/s1155077016/spatial_data/"+ID+"/"+ID+".h5ad"
spots_embeddings = np.loadtxt(os.path.join(base_path, ID + '_embeddings.txt'))
stage = ID
adata = sc.read_h5ad(file_name)
adata.obsm['embedding'] = np.float32(spots_embeddings)
n_neighbors = 30  # 10 70 140 neighbors
sc.pp.neighbors(adata, use_rep='embedding', #key_added='embedding',
                n_neighbors=n_neighbors)  # use_rep="X", ,method='gauss'
resolution = 1.2  # iver/TD 4 colon_all/SW4X 6,8,9
sc.tl.louvain(adata,  resolution=resolution)  # random_state=0,obsp='embedding_connectivities',
# del spots_embeddings
size = 6
figsize = (3, 3)  # sizedict[ID]  # liver 18,3 liver TD 12,3 colon 18,6 E16 6,4 colon_2112_SW4X 3,3
plt.rcParams["figure.figsize"] = figsize
sc.pl.embedding(adata, basis="spatial", color="louvain", s=size, show=False,
                title='SGCAST')  # legend_loc='on data',legend_fontsize=12,
# sc.pl.embedding(adata, basis="spatial", color="leiden",s=6, show=False, title='SGCAST')
# plt.savefig(
#     os.path.join(base_path,
#                  stage + "_bin50_res_" + str(int(resolution * 10)) + "_" + str(n_neighbors) + "neigh_size" + str(
#                      size) + "_plt" + str(int(figsize[0])) + "_" + str(int(figsize[1])) + "_50pc_7_4_2000_v100.png"),
#     bbox_inches="tight",
#     dpi=600)  # pred.png avremb
# plt.axis('off')
# plt.close()
###########refine
sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=n_neighbors, key_added="pixel")
arr = adata.obsp['pixel_connectivities']
arr = arr.astype('int32')
ref_num = 6
pred = adata.obs["louvain"].astype('int32')
refined_pred = refine_high(pred=np.array(pred.tolist()), dis=arr, num=ref_num, option=False)
adata.obs["refined_pred"] = refined_pred
adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')
plt.rcParams["figure.figsize"] = figsize
sc.pl.embedding(adata, basis="spatial", color="refined_pred", s=size, show=False,
                title='SGCAST')  # legend_loc='on data',legend_fontsize=12,
plt.savefig(os.path.join(base_path, "refine_" + stage + "_bin50_res_" + str(int(resolution * 10)) + "_"
                         + str(n_neighbors) + "neigh_size" + str(size) + "_plt" + str(int(figsize[0])) + "_" + str(
    int(figsize[1])) +
                         "_50pc.png"), bbox_inches="tight",
            dpi=600)  # pred.png avremb _7_4_2000_v100
plt.axis('off')
plt.close()


num_cluster = len(np.unique(adata.obs["refined_pred"]))

if 'log1p' in list(adata.uns):
    del adata.uns['log1p']


adata.obs["refined_pred"]=adata.obs["refined_pred"].astype(str)
sc.tl.rank_genes_groups(adata, 'refined_pred', method='wilcoxon', key_added="wilcoxon")  # 'refined_pred'

sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, key="wilcoxon", groupby='refined_pred',save=ID+'dotplot.png')
DEs = []
df = sc.get.rank_genes_groups_df(adata, key='wilcoxon', group=None)
groups = df['group'].unique()

for group in groups:
    dfg = df.loc[df['group'] == group]
    names = dfg['names'][:5].tolist()
    for i in names:
        DEs.append(i)
        
print(ID + '_DEs are:', DEs)

adata.obsm["spatial"] = adata.obsm["spatial"][:, [1, 0]]
#adata.obsm["spatial"] = adata.obsm["spatial"]*(-1)
adata.obsm["spatial"][:, 0] = adata.obsm["spatial"][:, 0]*(-1)

adata.obsm["spatial"] = adata.obsm["spatial"][:, [1, 0]]

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
"""
for i in range(num_cluster):
    glist = sc.get.rank_genes_groups_df(adata, group=i,
                                        key='wilcoxon', log2fc_min=0.25,
                                        pval_cutoff=0.01)['names'].squeeze().str.strip().tolist()
    enr_res = gseapy.enrichr(gene_list=glist,
                             organism='Human',
                             gene_sets='GO_Biological_Process_2018',
                             cutoff=0.5)
    gseapy.barplot(enr_res.res2d, figsize=(3, 5), legend=True)  # ,title='GO_Biological_Process_2018'
    plt.savefig(ID + '_' + group + '_' + 'gobarplot.png', bbox_inches="tight", dpi=600)  # pred.png avremb
    plt.axis('off')
    plt.close()
"""
print('Finished')
