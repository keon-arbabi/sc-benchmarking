import gc
import sys
import polars as pl  
import scanpy as sc 
import warnings
warnings.filterwarnings('ignore')
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info, transfer_accuracy, print_df

DATASET_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH_TIME = sys.argv[3]
OUTPUT_PATH_ACC = sys.argv[4]

system_info()

print('--- Params ---')
print('scanpy transfer')
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

with timers('Split data'):
    if DATASET_NAME == 'SEAAD':
        data_ref = data[data.obs['cond'] == 0].copy()
        data_query = data[data.obs['cond'] == 1].copy()
    elif DATASET_NAME == 'PBMC':
        data_ref = data[data.obs['cond'] == 'PBS'].copy()
        data_query = data[data.obs['cond'] == 'cytokine'].copy()
    
del data; gc.collect()

with timers('Normalization'):
    sc.pp.normalize_total(data_ref)
    sc.pp.log1p(data_ref)    
    sc.pp.normalize_total(data_query)
    sc.pp.log1p(data_query)

with timers('Feature selection'):
    sc.pp.highly_variable_genes(data_ref, n_top_genes=2000, batch_key='sample')
    hvg_genes = data_ref.var[data_ref.var['highly_variable']].index
    data_ref = data_ref[:, hvg_genes].copy()
    data_query = data_query[:, hvg_genes].copy()

with timers('PCA'):
    sc.pp.scale(data_ref)
    sc.pp.scale(data_query)
    sc.pp.pca(data_ref)

data_query.obs['cell_type_orig'] = data_query.obs['cell_type'].copy()

with timers('Transfer labels'):
    sc.pp.neighbors(data_ref)
    sc.tl.ingest(data_query, data_ref, obs='cell_type', embedding_method='pca')

accuracy_df = transfer_accuracy(
    data_query.obs, 'cell_type_orig', 'cell_type')
accuracy_df.write_csv(OUTPUT_PATH_ACC)

timers.print_summary(unit='s')

timers_df = timers.to_dataframe(sort=False, unit='s')\
    .with_columns(
        pl.lit('scanpy').alias('library'),
        pl.lit('transfer').alias('test'),
        pl.lit(DATASET_NAME).alias('dataset'))
timers_df.write_csv(OUTPUT_PATH_TIME)

if not any(timers_df['aborted']):
    print('--- Completed successfully ---')

del timers, timers_df, data_query, data_ref
gc.collect()
