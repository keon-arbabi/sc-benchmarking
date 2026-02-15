import gc
import sys
import warnings
import polars as pl
import scanpy as sc
import matplotlib.pyplot as plt
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info

warnings.filterwarnings("ignore")

DATASET_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH_TIME = sys.argv[3]
OUTPUT_PATH_EMBEDDING = sys.argv[4]
OUTPUT_PATH_DOUBLET = sys.argv[5]

if __name__ == '__main__':

    system_info()
    print('--- Params ---')
    print('scanpy basic')
    print(f'{DATASET_NAME=}')

    timers = MemoryTimer(silent=False)

    with timers('Load data'):
        data = sc.read_h5ad(DATA_PATH)

    with timers('Quality control'):
        data.var['mt'] = data.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(
            data, qc_vars=['mt'], inplace=True, log1p=True)
        sc.pp.filter_cells(data, min_genes=100)
        sc.pp.filter_genes(data, min_cells=3)

    with timers('Doublet detection'):
        sc.pp.scrublet(data, batch_key='sample')

    pl.DataFrame({
        'cell_id': data.obs_names.tolist(),
        'doublet_score': data.obs['doublet_score'].tolist(),
        'is_doublet': data.obs['predicted_doublet'].tolist()
    }).write_csv(OUTPUT_PATH_DOUBLET)

    with timers('Quality control'):
        data = data[data.obs['predicted_doublet'] == False].copy()

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

    with timers('Clustering (3 res.)'):
        for res in [0.5, 1.0, 2.0]:
            sc.tl.leiden(
                data,
                resolution=res,
                flavor='igraph',
                n_iterations=2,
                key_added=f'leiden_res_{res:4.2f}')

    embedding_df = pl.DataFrame({
        'cell_id': data.obs_names.tolist(),
        'embed_1': data.obsm['X_umap'][:, 0],
        'embed_2': data.obsm['X_umap'][:, 1],
        'cell_type': data.obs['cell_type'].tolist(),
        'clusters_0.5': data.obs['leiden_res_0.50'].tolist(),
        'clusters_1.0': data.obs['leiden_res_1.00'].tolist(),
        'clusters_2.0': data.obs['leiden_res_2.00'].tolist(),
    })
    embedding_df.write_csv(OUTPUT_PATH_EMBEDDING)

    with timers('Plot embedding'):
        sc.pl.umap(data, color='cell_type')
        plt.savefig(
            f'sc-benchmarking/figures/scanpy_embedding_{DATASET_NAME}.png',
            dpi=300,
            bbox_inches='tight')

    with timers('Find markers'):
        sc.tl.rank_genes_groups(data, groupby='cell_type', method='wilcoxon')

    timers.print_summary(unit='s')

    timers_df = timers.to_dataframe(sort=False, unit='s').with_columns(
        pl.lit('scanpy').alias('library'),
        pl.lit('basic').alias('test'),
        pl.lit(DATASET_NAME).alias('dataset'),)
    timers_df.write_csv(OUTPUT_PATH_TIME)

    if not any(timers_df['aborted']):
        print('--- Completed successfully ---')

    print('\n--- Session Info ---')
    sc.logging.print_header()

    del timers, timers_df, data
    gc.collect()
