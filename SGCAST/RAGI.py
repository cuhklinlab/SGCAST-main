from scipy import stats
import scanpy as sc
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/SGCAST_path')

np.random.seed(2022)
base_path = '/SGCAST_path/output'
file_name= "/datapath/Stereo-seq.h5ad"

adata = sc.read_h5ad(file_name)
all_genes = list(adata.var_names)
all_genescap = [i.upper() for i in all_genes]
housekeeping_genes=['RRN18S','GAPDH','ARBP','HPRT1','PGK1','PPIA','RPL13A','RPLP0','B2M','YWHAZ','SDHA','TFRC',
                    'Actb','Puf60','Psmd4','Chmp2a','Eif3f','Heatr3']
# one more step to check wether all house genes in highly variable genes and remove those not in hvgs.
housekeeping_genes = [i.upper() for i in housekeeping_genes]
house_filter=[]
for i in housekeeping_genes:
    house_filter.append(i in all_genescap)
housekeeping_genes = list(np.array(housekeeping_genes)[house_filter])
housekeeping_genes = list(np.unique(housekeeping_genes))


marker_genes=['F3', 'C3', 'Omp', 'Gnal','Cnga2',  'Adcy3','Olfr78','Neurod1',
'Omp','Neurog1','Ascl1','Gng8','Gap43','Ncam1','Omp','Ano2','Cngb1','Neurog1','Cnga4','Cnga2','Adcy3','Gnal','Krt5','Olfr50','Olfr632',
'Hes5','Sox5','Wif1','Tubb3','Frzb','Ptn','Ptprz1','Sox10','Sema6a','Plp1','Nkd2','Nell2','Adgrg1','Tubb3','Dpysl3','Lypd6','Plp1']
# from CellMarker

# one more step to check wether all marker genes in highly variable genes and remove those not in hvgs.
marker_genes = [i.upper() for i in marker_genes]
marker_filter=[]
for i in marker_genes:
    marker_filter.append(i in all_genescap)
marker_genes = list(np.array(marker_genes)[marker_filter])
marker_genes= list(np.unique(marker_genes))

folder_clusters = '/folder that stores clusters (csv file) identified by each method for Stereo-seq mouse olfactory bulb'

# get index of housekeeping genes and marker genes
house_pos=[]
for i in housekeeping_genes:
    house_pos.append(all_genescap.index(i))

marker_pos=[]
for i in marker_genes:
    marker_pos.append(all_genescap.index(i))
    
# Subset from the main matrix the housekeeping genes and marker genes
mat1 = np.transpose(adata.X[:,house_pos])
df_matrix_housekeeping=pd.DataFrame(mat1, index = housekeeping_genes)
mat2 = np.transpose(adata.X[:,marker_pos])
df_matrix_marker=pd.DataFrame(mat2, index = marker_genes)


def residual_average_gini_index(df_matrix_housekeeping, df_matrix_marker,folder_clusters,
                                min_cells_per_cluster=10):
    
    # Define a function to compute the Gini score
    def gini(list_of_values):
        sorted_list = sorted(list_of_values)
        height, area = 0, 0
        for value in sorted_list:
            height += value
            area += height - value / 2.
            fair_area = height * len(list_of_values) / 2.
        return (fair_area - area) / fair_area
    # Function to calculate Gini value for all the genes
    def calculate_gini(df_matrix, gene_name, clustering_info):
        return gini(get_avg_per_cluster(df_matrix, gene_name, clustering_info, use_log2=False))
    # Function to calculate Gini value for all the genes
    def calculate_gini_values(df_matrix, clustering_info):
        gini_values = []
        for gene_name in df_matrix.index:
            gini_values.append(calculate_gini(df_matrix, gene_name, clustering_info))
        return gini_values
    # Write a function to compute delta difference of the average accessibility in Marker vs Housekeeping and Kolmogorov Smirnov test
    def score_clustering_solution(df_matrix_marker, df_matrix_housekeeping, clustering_info):
        gini_values_housekeeping = calculate_gini_values(df_matrix_housekeeping, clustering_info)
        gini_values_marker = calculate_gini_values(df_matrix_marker, clustering_info)
        statistic, p_value = stats.ks_2samp(gini_values_marker, gini_values_housekeeping)
        return np.mean(gini_values_marker), np.mean(gini_values_housekeeping), np.mean(gini_values_marker) - np.mean(
            gini_values_housekeeping), statistic, p_value
    # Function to compute the average accessibility value per cluster
    def get_avg_per_cluster(df_matrix, gene_name, clustering_info, use_log2=False):
        N_clusters = len(clustering_info.index.unique())
        avg_per_cluster = np.zeros(N_clusters)
        for idx, idx_cluster in enumerate(sorted(np.unique(clustering_info.index.unique()))):
            if use_log2:
                values_cluster = df_matrix.loc[gene_name, clustering_info.loc[idx_cluster, :].values.flatten()].apply(
                    lambda x: np.log2(x + 1))
            else:
                values_cluster = df_matrix.loc[gene_name, clustering_info.loc[idx_cluster, :].values.flatten()]
            avg_per_cluster[idx] = values_cluster.mean()
            if avg_per_cluster[idx] > 0:
                avg_per_cluster[idx] = avg_per_cluster[idx]  # /values_cluster.std()
        return avg_per_cluster
    # Run the method for all the clustering solutions
    df_metrics = pd.DataFrame(
        columns=['Method', 'Clustering', 'Gini_Marker_Genes', 'Gini_Housekeeping_Genes', 'Difference', 'KS_statistics',
                 'p-value'])
    for clusters_filename in os.listdir(folder_clusters):
        method = '_'.join(clusters_filename.split('_')[:-1])
        print(method)
        df_clusters = pd.read_csv(os.path.join(folder_clusters, clusters_filename), sep=',') #\t , index_col=0
        for clustering_method in df_clusters.columns:
            clustering_info = pd.DataFrame(df_clusters[clustering_method])
            clustering_info['Barcode'] = clustering_info.index
            clustering_info = clustering_info.set_index(clustering_method)
            # REMOVE CLUSTERS WITH FEW CELLS
            cluster_sizes = pd.value_counts(clustering_info.index)
            clustering_info = clustering_info.loc[cluster_sizes[cluster_sizes > min_cells_per_cluster].index.values, :]
            mean_gini_marker, mean_gini_housekeeping, mean_gini_difference, statistics, p_value = score_clustering_solution(
                df_matrix_marker, df_matrix_housekeeping, clustering_info)
            df_metrics = df_metrics.append({'Method': method, 'Clustering': clustering_method,
                                            'Gini_Marker_Genes': mean_gini_marker,
                                            'Gini_Housekeeping_Genes': mean_gini_housekeeping,
                                            'Difference': mean_gini_difference, 'KS_statistics': statistics,
                                            'p-value': p_value},
                                           ignore_index=True)
    return df_metrics

# run the function and save results in df_metrics
df_metrics = residual_average_gini_index(df_matrix_housekeeping, df_matrix_marker,folder_clusters,
                                min_cells_per_cluster=10)


