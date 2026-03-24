import sys
import numpy as np
import polars as pl
import scanpy as sc
import rapids_singlecell as rsc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info

DATA_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH_TIME = sys.argv[3]
OUTPUT_PATH_EMBEDDING = sys.argv[4]
OUTPUT_PATH_PCS = sys.argv[5]
OUTPUT_PATH_NEIGHBORS = sys.argv[6]

if __name__ == '__main__':

    system_info()
    print('--- Params ---')
    print('rapids basic')
    print(f'{DATA_PATH=}')

    timers = MemoryTimer(
        silent=False, csv_path=OUTPUT_PATH_TIME,
        csv_columns={'library': 'rapids', 'test': 'basic',
                     'dataset': DATA_NAME})

    # Load and QC on CPU, then transfer filtered data to GPU
    # (raw data too large for single GPU VRAM; driver 535 breaks managed memory)
    with timers('Load data'):
        data = sc.read_h5ad(DATA_PATH)

    with timers('Quality control'):
        data.var['mt'] = data.var_names.str.upper().str.startswith('MT-')
        data.var['malat1'] = data.var_names.str.upper() == 'MALAT1'
        sc.pp.calculate_qc_metrics(
            data, qc_vars=['mt', 'malat1'], inplace=True)
        keep = ((data.obs['n_genes_by_counts'].values >= 100) &
                (data.obs['pct_counts_mt'].values <= 5) &
                (data.obs['pct_counts_malat1'].values > 0))
        data = data[keep].copy()
        rsc.get.anndata_to_GPU(data)

    with timers('Normalization'):
        rsc.pp.normalize_total(data)
        rsc.pp.log1p(data)

    with timers('Feature selection'):
        rsc.pp.highly_variable_genes(
            data, n_top_genes=2000, batch_key='donor')

    with timers('PCA'):
        rsc.tl.pca(data)

    with timers('Nearest neighbors'):
        rsc.pp.neighbors(data)

    with timers('Embedding'):
        rsc.tl.umap(data)

    with timers('Clustering'):
        for res in [0.25, 0.5, 1.0, 1.5, 2.0]:
            rsc.tl.leiden(
                data,
                resolution=res,
                key_added=f'leiden_res_{res:4.2f}')

    with timers('Plot embedding'):
        rsc.get.anndata_to_CPU(data)
        sc.pl.umap(
            data, color='cell_type')
        plt.savefig(
            f'sc-benchmarking/figures/rapids_embedding_{DATA_NAME}.png',
            bbox_inches='tight', dpi=300)
        rsc.get.anndata_to_GPU(data)

    # GPU-native logreg; scanpy uses t-test
    with timers('Find markers'):
        rsc.tl.rank_genes_groups_logreg(data, groupby='cell_type')

    # Move back to CPU for output
    rsc.get.anndata_to_CPU(data)

    # Save PCs
    pcs = data.obsm['X_pca']
    pc_df = pl.DataFrame({
        f'PC_{i+1}': pcs[:, i] for i in range(pcs.shape[1])
    })
    pc_df.write_csv(OUTPUT_PATH_PCS)

    # Save neighbors
    dist = data.obsp['distances']
    n_neighbors = dist.indptr[1] - dist.indptr[0] - 1
    neighbors = dist.indices.reshape(data.n_obs, -1)[:, 1:].astype(np.uint32)
    neighbors_df = pl.DataFrame({
        f'neighbor_{i+1}': neighbors[:, i]
        for i in range(n_neighbors)
    })
    neighbors_df.write_csv(OUTPUT_PATH_NEIGHBORS)

    # Save embeddings
    embedding_df = pl.DataFrame({
        'cell_id': data.obs_names.tolist(),
        'embed_1': data.obsm['X_umap'][:, 0],
        'embed_2': data.obsm['X_umap'][:, 1],
        'cell_type': data.obs['cell_type'].tolist(),
        'cell_type_broad': data.obs['cell_type_broad'].tolist(),
        'cluster_res_0.25': data.obs['leiden_res_0.25'].tolist(),
        'cluster_res_0.5': data.obs['leiden_res_0.50'].tolist(),
        'cluster_res_1.0': data.obs['leiden_res_1.00'].tolist(),
        'cluster_res_1.5': data.obs['leiden_res_1.50'].tolist(),
        'cluster_res_2.0': data.obs['leiden_res_2.00'].tolist(),
    })
    embedding_df.write_csv(OUTPUT_PATH_EMBEDDING)

    timers.shutdown()
    print('--- Completed successfully ---')

    print('\n--- Session Info ---')
    sc.logging.print_header()
