import gc
import sys
import polars as pl  
sys.path.append('/home/karbabi') 
from single_cell import SingleCell  
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info, transfer_accuracy

DATASET_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
NUM_THREADS = int(sys.argv[3])
OUTPUT_PATH_TIME = sys.argv[4]
OUTPUT_PATH_ACC = sys.argv[5]

system_info()

print('--- Params ---')
print('brisc transfer')
print(f'{DATASET_NAME=}')
print(f'{NUM_THREADS=}')

timers = MemoryTimer(silent=False)

with timers('Load data'):
    data = SingleCell(DATA_PATH, num_threads=NUM_THREADS)

with timers('Quality control'):
    data = data.qc(
        subset=False,
        remove_doublets=False,
        allow_float=True,
        verbose=False)

with timers('Doublet detection'):
    data = data.find_doublets(batch_column='sample')

with timers('Quality control'):
    data = data.filter_obs(
        pl.col('doublet').not_() & pl.col('passed_QC'))

data = data.drop_obs('passed_QC')

with timers('Split data'):
    data = dict(data.split_by_obs('cond'))
    if DATASET_NAME == 'SEAAD':
        data_ref = data.pop(0)
        data_query = data.pop(1)
    elif DATASET_NAME == 'PBMC':
        data_ref = data.pop('PBS')
        data_query = data.pop('cytokine')

del data; gc.collect()
    
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

print('--- Transfer Accuracy ---')
accuracy_df = transfer_accuracy(
    data_query.obs, 'cell_type', 'cell_type_transferred')
accuracy_df.write_csv(OUTPUT_PATH_ACC)

timers.print_summary(unit='s')

timers_df = timers.to_dataframe(sort=False, unit='s')\
    .with_columns(
        pl.lit('brisc').alias('library'),
        pl.lit('transfer').alias('test'),
        pl.lit(DATASET_NAME).alias('dataset'),
        pl.lit('single-threaded' if NUM_THREADS == 1 else 'multi-threaded')
        .alias('num_threads'))
timers_df.write_csv(OUTPUT_PATH_TIME)

if not any(timers_df['aborted']):
    print('--- Completed successfully ---')

del timers, timers_df, data_query, data_ref
gc.collect()

