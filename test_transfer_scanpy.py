import gc
import sys
import polars as pl
import scanpy as sc
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info, transfer_accuracy

DATA_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH_TIME = sys.argv[3]
OUTPUT_PATH_ACC = sys.argv[4]

if __name__ == '__main__':

    system_info()
    print('--- Params ---')
    print('scanpy transfer')
    print(f'{DATA_PATH=}')

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

    with timers('Split data'):
        data_ref = data[data.obs['is_ref'] == 1].copy()
        data_query = data[data.obs['is_ref'] == 0].copy()

    del data; gc.collect()

    with timers('Normalization'):
        sc.pp.normalize_total(data_ref)
        sc.pp.log1p(data_ref)
        sc.pp.normalize_total(data_query)
        sc.pp.log1p(data_query)

    with timers('Feature selection'):
        sc.pp.highly_variable_genes(
            data_ref, n_top_genes=2000, batch_key='donor')
        hvg_genes = data_ref.var[data_ref.var['highly_variable']].index
        data_ref = data_ref[:, hvg_genes].copy()
        data_query = data_query[:, hvg_genes].copy()

    with timers('PCA'):
        sc.pp.pca(data_ref)

    data_query.obs['cell_type_orig'] = data_query.obs['cell_type']

    with timers('Transfer labels'):
        sc.pp.neighbors(data_ref)
        sc.tl.ingest(
            data_query,
            data_ref,
            obs='cell_type',
            embedding_method='pca')

    print('--- Transfer Accuracy ---')
    transfer_accuracy(
        data_query.obs, 'cell_type_orig', 'cell_type')\
    .write_csv(OUTPUT_PATH_ACC)

    timers.print_summary(unit='s')

    timers_df = timers.to_dataframe(sort=False, unit='s')\
        .with_columns(
            pl.lit('scanpy').alias('library'),
            pl.lit('transfer').alias('test'),
            pl.lit(DATA_NAME).alias('dataset'))
    timers_df.write_csv(OUTPUT_PATH_TIME)

    if not any(timers_df['aborted']):
        print('--- Completed successfully ---')

    print('\n--- Session Info ---')
    sc.logging.print_header()
