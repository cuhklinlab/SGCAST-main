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
np.random.seed(2022)
base_path = '/lustre/project/Stat/s1155077016/SGCAST/SGCAST/SGCAST/output'
IDs = ['Stereo-seq'] #'scope_colon','scope_colon_2112_SW4X','Puck_200127_15', 'Puck_200115_08',   'Puck_191204_01', 'Puck_190921_21','Stereo-seq','slidev2_mouse_OlfactoryBulb'
sizedict= {'scope_colon_2112_SW4X':(3, 3),'scope_liverTD':(12, 3),'scope_liver':(18, 3),'Stereo-seq':(3, 3),'scope_colon':(18, 6)}
sizedict_clu= {'Stereo-seq':'stereo_mbulb_cluster','slidev2_mouse_OlfactoryBulb':'slidev2_mbulb_cluster','scope_colon': 'colon_cluster'}
ress = [ 1.2 ] #0.8, , 0.5,0.6,0.7,0.8,0.9,1.0,1.1, 1.3, 1.4,1.5,1.6,1.7 0.9,1.0,1.1, 1.2
neighbors = [30] #,40  12,15,20,25,30 9,10,12,15,20
for ID in IDs:
    for resolution in ress:
        for n_neighbors in neighbors:
            file_name = "/lustre/project/Stat/s1155077016/spatial_data/" + ID + "/" + ID + ".h5ad"
            spots_embeddings = np.loadtxt(os.path.join(base_path, ID + '_embeddings.txt'))
            stage = ID
            adata = sc.read_h5ad(file_name)
            adata.obsm['embedding'] = np.float32(spots_embeddings)
            # n_neighbors = 15  # 10 70 140 neighbors
            sc.pp.neighbors(adata, use_rep='embedding',  # key_added='embedding',
                            n_neighbors=n_neighbors)  # use_rep="X", ,method='gauss'
            # resolution = 1.2  # iver/TD 4 colon_all/SW4X 6,8,9
            sc.tl.louvain(adata, resolution=resolution)  # random_state=0, obsp='embedding_connectivities',
            # # del spots_embeddings
            size = 6
            figsize = sizedict[ID] #  (3, 3)  # liver 18,3 liver TD 12,3 colon 18,6 E16 6,4 colon_2112_SW4X 3,3
            # plt.rcParams["figure.figsize"] = figsize
            # sc.pl.embedding(adata, basis="spatial", color="louvain", s=size, show=False,
            #                 title='SGCAST')  # legend_loc='on data',legend_fontsize=12,
            # # sc.pl.embedding(adata, basis="spatial", color="leiden",s=6, show=False, title='SGCAST')
            # plt.savefig(
            #     os.path.join(base_path, stage + "_bin50_res_" + str(int(resolution * 10)) + "_" + str(
            #         n_neighbors) + "neigh_size" + str(
            #         size) + "_plt" + str(int(figsize[0])) + "_" + str(int(figsize[1])) + "_50pc_2000_v100.png"),
            #     bbox_inches="tight",
            #     dpi=600)  # pred.png avremb _7_4
            # plt.axis('off')
            # plt.close()
            ###########refine
            sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=n_neighbors, key_added="pixel")
            arr = adata.obsp['pixel_connectivities']
            arr = arr.astype('int32')
            ref_num = 6
            RAGI = 0.07628
            pred = adata.obs["louvain"].astype('int32')
            refined_pred = refine_high(pred=np.array(pred.tolist()), dis=arr, num=ref_num, option=False)
            adata.obs["refined_pred"] = refined_pred
            print(adata.obs["refined_pred"])
            di = {0: "EPL",1: "GCL", 2: "GL",3: "IPL", 4: "ONL",5: "MCL", 6: "RMS"}
            adata.obs=adata.obs.replace({"refined_pred": di})
            print(adata.obs["refined_pred"])
            adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')
            plt.rcParams["figure.figsize"] = figsize
            sc.pl.embedding(adata, basis="spatial", color="refined_pred", s=size, show=False,
                            title=['SGCAST(RAGI=%.3f)' % RAGI])  # legend_loc='on data',legend_fontsize=12,  'SGCAST'
            plt.savefig(os.path.join(base_path, "refine_" + stage + "_bin50_res_" + str(int(resolution * 10)) + "_"
                                     + str(n_neighbors) + "neigh_size" + str(size) + "_plt" + str(
                int(figsize[0])) + "_" + str(int(figsize[1])) +
                                     "_50pc_2000_v100.png"), bbox_inches="tight",
                        dpi=600)  # pred.png avremb _7_4
            plt.axis('off')
            plt.close()
            if 'log1p' in list(adata.uns):
                del adata.uns['log1p']

            adata.obs["refined_pred"]=adata.obs["refined_pred"].astype(str)
            sc.tl.rank_genes_groups(adata, 'refined_pred', method='wilcoxon', key_added="wilcoxon")  # 'refined_pred'

            sc.pl.rank_genes_groups_dotplot(adata, n_genes=2, key="wilcoxon",dendrogram=False, groups=[ "IPL", "MCL","ONL","RMS"],swap_axes=True,save=ID+'dotplot.png')
#groupby='refined_pred'
            output_path = '/lustre/project/Stat/s1155077016/SGCAST/SGCAST/SGCAST/'+sizedict_clu[ID]
            adata.obs['refined_pred'].to_csv(os.path.join(output_path, 'sgcast' + ID + '_.tsv'), index=0)
            sc.tl.umap(adata)
            sc.pl.umap(adata, color='refined_pred',save=stage +'.png', title='SGCAST')
            sc.tl.paga(adata, groups='refined_pred')
            sc.pl.paga(adata,save=stage +'.png', title='SGCAST')
            sc.pl.paga_compare(adata,save=stage +'com.png', title='SGCAST')




    #SC_score = silhouette_score(adata.obsm['embedding'], adata.obs['refined_pred'],  metric='euclidean') #euclidean
    #DB_score = davies_bouldin_score(adata.obsm['embedding'], adata.obs['refined_pred'])
    #print('SC score res'+stage+':', SC_score)
    #print('DB score res'+stage+':',DB_score)


    #sc.tl.umap(adata) #,neighbors_key='embedding'
    #sc.pl.umap(adata, color='refined_pred',save=stage +'.png', title='SGCAST')

    #sc.tl.paga(adata, groups='refined_pred')
    #sc.pl.paga(adata,save=stage +'.png')
    #sc.pl.paga_compare(adata,save=stage +'com.png')
"""
            output_path = '/lustre/project/Stat/s1155077016/SpaGCN-1.2.2/SpaGCN-1.2.2-attentionGPU_go/SpaGCN122/'+sizedict_clu[ID]
            adata.obs['refined_pred'].to_csv(os.path.join(output_path, 'sgcast' + ID + "res" + str(int(resolution * 10)) +
                                                      str(n_neighbors) + "neigh"+'_.tsv'), index=0)
"""


print('Finished')