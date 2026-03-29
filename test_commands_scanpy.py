import gc
import sys
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_local import MemoryTimer, system_info

DATA_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH_TIME = sys.argv[3]

if __name__ == '__main__':

    system_info()
    print('--- Params ---')
    print('scanpy manipulation')
    print(f'{DATA_PATH=}')

    timers = MemoryTimer(
        silent=False, csv_path=OUTPUT_PATH_TIME, summary_unit='ms',
        csv_columns={'library': 'scanpy', 'test': 'manipulation',
                     'dataset': DATA_NAME})

    # Setup
    data = sc.read_h5ad(DATA_PATH)
    sc.pp.highly_variable_genes(
        data, n_top_genes=2000, batch_key='donor', flavor='seurat_v3')

    cell_name = data.obs_names[0]
    gene_name = data.var_names[0]
    cell_type_select = data.obs['cell_type'].iloc[0]
    donors = sorted(data.obs['donor'].unique())
    donor_df = pd.DataFrame({
        'donor': donors,
        'donor_index': range(len(donors))
    }).set_index('donor')

    with timers('Get expression by cell'):
        data[cell_name, :].X.toarray().ravel()

    with timers('Get expression by gene'):
        data[:, gene_name].X.toarray().ravel()

    with timers('Subset cells'):
        data[data.obs['cell_type'] == cell_type_select].copy()

    with timers('Subset genes'):
        data[:, data.var['highly_variable']].copy()

    with timers('Subsample cells'):
        sc.pp.sample(data, n=10_000, copy=True)

    with timers('Select obs columns'):
        data.obs.select_dtypes(exclude='number')

    with timers('Add metadata column'):
        data.obs['cell_type_enrichment'] = (
            data.obs.groupby(['donor', 'cell_type'], observed=True)['donor']
            .transform('size') /
            data.obs.groupby('donor', observed=True)['donor']
            .transform('size')) / (
            data.obs.groupby('cell_type', observed=True)['donor']
            .transform('size') / len(data))

    with timers('Cast obs column'):
        data.obs['cell_type'] = data.obs['cell_type'].astype(str)

    with timers('Rename obs column'):
        data.obs = data.obs.rename(
            columns={'cell_type_enrichment': 'ct_enrichment'})

    with timers('Remove metadata column'):
        data.obs = data.obs.drop(columns=['ct_enrichment'])

    with timers('Join obs metadata'):
        data.obs = data.obs.join(donor_df, on='donor')

    with timers('Rename cells'):
        data.obs_names = 'prefix_' + data.obs_names

    with timers('Split by obs column'):
        data_split = [data[data.obs['cell_type_broad'] == ct].copy()
                      for ct in data.obs['cell_type_broad'].unique()]

    with timers('Concatenate objects'):
        data = ad.concat(data_split)

    del data_split; gc.collect()

    with timers('Copy object'):
        data_copy = data.copy()

    timers.shutdown()
    print('--- Completed successfully ---')
