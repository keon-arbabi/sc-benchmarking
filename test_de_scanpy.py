import gc
import sys
import polars as pl
import scanpy as sc 
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info

DATASET_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH_TIME = sys.argv[3]

system_info()

print('--- Params ---')
print('scanpy de')
print(f'{DATASET_NAME=}')

timers = MemoryTimer(silent=False)

with timers('Load data'):
    data = sc.read_h5ad(DATA_PATH)

with timers('Quality control'):
    data.var['mt'] = data.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(data, qc_vars=['mt'], inplace=True, log1p=True)
    sc.pp.filter_cells(data, min_genes=100)
    sc.pp.filter_genes(data, min_cells=3)

with timers('Doublet detection'):
    sc.pp.scrublet(data, batch_key='sample')

with timers('Quality control'):
    data = data[data.obs['predicted_doublet'] == False].copy()

with timers('Data transformation (pseudobulk / normalization)'):
    sc.pp.normalize_total(data)
    sc.pp.log1p(data)

with timers('Differential expression'):
    de = {}
    if DATASET_NAME == 'SEAAD':
        data.obs['group'] = data.obs['cell_type']\
            .astype(str) + '_' + data.obs['cond'].astype(str)

        for cell_type in data.obs['cell_type'].unique():
            adata_sub = data[data.obs['cell_type'] == cell_type].copy()
            sc.tl.rank_genes_groups(
                adata_sub, 
                groupby='group', 
                method='wilcoxon',
                key_added=f'de_{cell_type}'
            )
            de[cell_type] = sc.get.rank_genes_groups_df(
                adata_sub,
                group=cell_type + '_1',
                key=f'de_{cell_type}'
            )
            
    elif DATASET_NAME == 'PBMC':
        for cell_type in data.obs['cell_type'].unique():
            adata_sub = data[data.obs['cell_type'] == cell_type].copy()
            sc.tl.rank_genes_groups(
                adata_sub,
                groupby='cytokine',
                groups=['IFN_gamma'],
                reference='PBS',
                method='wilcoxon',
                key_added=f'de_{cell_type}'
            )
            de[cell_type] = sc.get.rank_genes_groups_df(
                adata_sub,
                group='IFN_gamma',
                key=f'de_{cell_type}'
            )

timers.print_summary(unit='s')

timers_df = timers.to_dataframe(sort=False, unit='s').with_columns(
    pl.lit('scanpy').alias('library'),
    pl.lit('de').alias('test'),
    pl.lit(DATASET_NAME).alias('dataset'),)
timers_df.write_csv(OUTPUT_PATH_TIME)

if not any(timers_df['aborted']):
    print('--- Completed successfully ---')

print('\n--- Session Info ---')
sc.logging.print_header()

del data, de, timers, timers_df
gc.collect()

