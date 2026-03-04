import sys
import warnings
import numpy as np
import polars as pl
import scanpy as sc
import matplotlib.pyplot as plt
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info

warnings.filterwarnings('ignore')

DATA_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH_TIME = sys.argv[3]
OUTPUT_PATH_EMBEDDING = sys.argv[4]
OUTPUT_PATH_PCS = sys.argv[5]
OUTPUT_PATH_NEIGHBORS = sys.argv[6]

system_info()
print('--- Params ---')
print('scanpy basic')
print(f'{DATA_PATH=}')

if __name__ == '__main__':

    timers = MemoryTimer(silent=False)

    with timers('Load data'):
        data = sc.read_h5ad(DATA_PATH)

    with timers('Quality control'):
        data.var['mt'] = data.var_names.str.startswith('MT-')
        data.var['malat1'] = data.var_names == 'MALAT1'
        sc.pp.calculate_qc_metrics(
            data, qc_vars=['mt', 'malat1'], inplace=True)
        keep = ((data.obs['n_genes_by_counts'].values >= 100) &
                (data.obs['pct_counts_mt'].values <= 5) &
                (data.obs['pct_counts_malat1'].values > 0))
        data = data[keep].copy()

    with timers('Normalization'):
        sc.pp.normalize_total(data)
        sc.pp.log1p(data)

    with timers('Feature selection'):
        sc.pp.highly_variable_genes(
            data, n_top_genes=2000, batch_key='sample')

    with timers('PCA'):
        sc.tl.pca(data)

    with timers('Nearest neighbors'):
        sc.pp.neighbors(data)

    with timers('Embedding'):
        sc.tl.umap(data)

    with timers('Clustering'):
        for res in [0.5, 1.0, 2.0]:
            sc.tl.leiden(
                data,
                resolution=res,
                flavor='igraph', # future default
                n_iterations=2, # future default
                key_added=f'leiden_res_{res:4.2f}')

    with timers('Plot embedding'):
        sc.pl.umap(
            data, color='cell_type')
        plt.savefig(
            f'sc-benchmarking/figures/scanpy_embedding_{DATA_NAME}.png',
            bbox_inches='tight',
            dpi=300)

    with timers('Find markers'):
        sc.tl.rank_genes_groups(data, groupby='cell_type', method='wilcoxon')

    # save pcs
    pcs = data.obsm['X_pca']
    pc_df = pl.DataFrame({
        f'PC_{i+1}': pcs[:, i] for i in range(pcs.shape[1])
    })
    pc_df.write_csv(OUTPUT_PATH_PCS)

    # save neighbors
    dist = data.obsp['distances']
    n_neighbors = dist.indptr[1] - dist.indptr[0] - 1
    neighbors = dist.indices.reshape(data.n_obs, -1)[:, 1:].astype(np.uint32)
    neighbors_df = pl.DataFrame({
        f'neighbor_{i+1}': neighbors[:, i]
        for i in range(n_neighbors)
    })
    neighbors_df.write_csv(OUTPUT_PATH_NEIGHBORS)

    # save embeddings
    embedding_df = pl.DataFrame({
        'cell_id': data.obs_names.tolist(),
        'embed_1': data.obsm['X_umap'][:, 0],
        'embed_2': data.obsm['X_umap'][:, 1],
        'cell_type': data.obs['cell_type'].tolist()
    })
    embedding_df.write_csv(OUTPUT_PATH_EMBEDDING)

    # save timings 
    timers_df = timers\
        .to_dataframe(sort=False, unit='s')\
        .with_columns(
            pl.lit('scanpy').alias('library'),
            pl.lit('basic').alias('test'),
            pl.lit(DATA_NAME).alias('dataset'))
    timers_df.write_csv(OUTPUT_PATH_TIME)

    timers.print_summary(unit='s')

    if not any(timers_df['aborted']):
        print('--- Completed successfully ---')

    print('\n--- Session Info ---')
    sc.logging.print_header()
