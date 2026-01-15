import gc
import sys
import warnings
import polars as pl  
import scanpy as sc 
warnings.filterwarnings('ignore')
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info, transfer_accuracy

# DATASET_NAME = sys.argv[1]
# DATA_PATH = sys.argv[2]
# REF_PATH = sys.argv[3]
# OUTPUT_PATH = sys.argv[4]

DATASET_NAME = 'SEAAD'
DATA_PATH = 'single-cell/SEAAD/SEAAD_raw.h5ad'
REF_PATH = 'single-cell/SEAAD/SEAAD_ref.h5ad'
OUTPUT_PATH = 'sc-benchmarking/output/test_transfer_scanpy_SEAAD.csv'

system_info()

print('--- Params ---')
print('scanpy transfer')
print(f'{sys.version=}')
print(f'{DATASET_NAME=}')

timers = MemoryTimer(silent=False)

with timers('Load data (query)'):
    data_query = sc.read_h5ad(DATA_PATH)

# Not timed 
data_query.obs['cell_type_orig'] = data_query.obs['cell_type']
data_query.obs = data_query.obs.drop(columns=['cell_type'])

with timers('Load data (ref)'):
    data_ref = sc.read_h5ad(REF_PATH)

with timers('Quality control'):
    data_query.var['mt'] = data_query.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(
        data_query, qc_vars=['mt'], log1p=True, inplace=True)
    sc.pp.filter_cells(data_query, min_genes=100, copy=False)
    sc.pp.filter_genes(data_query, min_cells=3, copy=False)
    
with timers('Doublet detection'):
    sc.pp.scrublet(data_query, batch_key='sample', copy=False)

with timers('Quality control'):
    data_query = data_query[
        data_query.obs['predicted_doublet'] == False].copy()

#%% Normalization
with timers('Normalization'):
    sc.pp.normalize_total(data_ref)
    sc.pp.log1p(data_ref)    
    sc.pp.normalize_total(data_query)
    sc.pp.log1p(data_query)

# Note: Highly variable gene selection is not explicitly done in the 
# scanpy vignette, but the exemplar data are pre-filtered.

#%% Feature selection
with timers('Feature selection'):
    var_names = data_ref.var_names.intersection(data_query.var_names)
    data_ref = data_ref[:, var_names].copy()
    data_query = data_query[:, var_names].copy()

    sc.pp.highly_variable_genes(data_ref, n_top_genes=2000, batch_key='sample')
    hvg_genes = data_ref.var[data_ref.var['highly_variable']].index
    data_ref = data_ref[:, hvg_genes].copy()
    data_query = data_query[:, hvg_genes].copy()

# Note: The exemplar data used in the scanpy vignette are scaled

with timers('PCA'):
    sc.pp.scale(data_ref)
    sc.pp.scale(data_query)
    sc.pp.pca(data_ref)

with timers('Transfer labels'):
    sc.pp.neighbors(data_ref)
    sc.tl.ingest(data_query, data_ref, obs='cell_type', embedding_method='pca')

with pl.Config(tbl_rows=-1):
    print('--- Transfer Accuracy ---')
    obs_df = pl.from_pandas(
        data_query.obs[['cell_type_orig', 'cell_type']].reset_index(drop=True))
    print(transfer_accuracy(obs_df, 'cell_type_orig', 'cell_type'))

timers.print_summary(unit='s')

timers_df = timers.to_dataframe(sort=False, unit='s')\
    .with_columns(
        pl.lit('scanpy').alias('library'),
        pl.lit('transfer').alias('test'),
        pl.lit(DATASET_NAME).alias('dataset'))
timers_df.write_csv(OUTPUT_PATH)

if not any(timers_df['aborted']):
    print('--- Completed successfully ---')

del timers, timers_df, data_query, data_ref, obs_df
gc.collect()
