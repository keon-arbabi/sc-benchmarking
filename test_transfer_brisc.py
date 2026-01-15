import gc
import sys
import polars as pl  
sys.path.append('/home/karbabi') 
from single_cell import SingleCell  
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info, transfer_accuracy

# DATASET_NAME = sys.argv[1]
# DATA_PATH = sys.argv[2]
# REF_PATH = sys.argv[3]
# NUM_THREADS = int(sys.argv[4])
# OUTPUT_PATH = sys.argv[5]

# DATASET_NAME = 'SEAAD'
# DATA_PATH = 'single-cell/SEAAD/SEAAD_raw.h5ad'
# REF_PATH = 'single-cell/SEAAD/SEAAD_ref.h5ad'
# NUM_THREADS = -1
# OUTPUT_PATH = 'sc-benchmarking/output/test_transfer_brisc_SEAAD_-1.csv'

DATASET_NAME = 'PBMC'
DATA_PATH = 'single-cell/PBMC/Parse_PBMC_raw.h5ad'
REF_PATH = 'single-cell/PBMC/ScaleBio_PBMC_ref.h5ad'
NUM_THREADS = -1
OUTPUT_PATH = 'sc-benchmarking/output/test_transfer_brisc_PBMC_-1.csv'

system_info()

print('--- Params ---')
print('brisc transfer')
print(f'{sys.version=}')
print(f'{DATASET_NAME=}')
print(f'{NUM_THREADS=}')

timers = MemoryTimer(silent=False)

with timers('Load data (query)'):
    data_query = SingleCell(DATA_PATH, num_threads=NUM_THREADS)

with timers('Load data (ref)'):
    data_ref = SingleCell(REF_PATH, num_threads=NUM_THREADS)
    
with timers('Quality control'):
    data_query = data_query.qc(
        subset=False,
        remove_doublets=False,
        allow_float=True,
        verbose=False)

with timers('Doublet detection'):
    data_query = data_query.find_doublets(batch_column='sample')

with timers('Quality control'):
    data_query = data_query.filter_obs(
        pl.col('doublet').not_() & pl.col('passed_QC'))

data_query = data_query.drop_obs('passed_QC')
    
with timers('Feature selection'):
    data_ref, data_query = data_ref.hvg(data_query)

with timers('Normalization'):
    data_ref = data_ref.normalize()
    data_query = data_query.normalize()

with timers('PCA'):
    data_ref, data_query = data_ref.PCA(data_query)

with timers('Transfer labels'):
    data_ref, data_query = data_ref.harmonize(data_query)
    data_query = data_query.label_transfer_from(
        data_ref, 'cell_type', cell_type_column='cell_type_transferred')

with pl.Config(tbl_rows=-1):
    print('--- Transfer Accuracy ---')
    print(transfer_accuracy(
        data_query.obs, 'cell_type', 'cell_type_transferred'))

timers.print_summary(unit='s')

timers_df = timers.to_dataframe(sort=False, unit='s')\
    .with_columns(
        pl.lit('brisc').alias('library'),
        pl.lit('transfer').alias('test'),
        pl.lit(DATASET_NAME).alias('dataset'),
        pl.lit('single-threaded' if NUM_THREADS == 1 else 'multi-threaded')
        .alias('num_threads'))
timers_df.write_csv(OUTPUT_PATH)

if not any(timers_df['aborted']):
    print('--- Completed successfully ---')

del timers, timers_df, data_query, data_ref
gc.collect()

