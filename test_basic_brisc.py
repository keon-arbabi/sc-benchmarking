import gc
import sys
import polars as pl  
sys.path.append('/home/karbabi') 
from single_cell import SingleCell  
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info

DATASET_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
NUM_THREADS = int(sys.argv[3])
OUTPUT_PATH_TIME = sys.argv[4]

system_info()

print('--- Params ---')
print('brisc basic')
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
    data = data.filter_obs(pl.col('doublet').not_() & pl.col('passed_QC'))

data = data.drop_obs('passed_QC')

with timers('Feature selection'):
    data = data.hvg()

with timers('Normalization'):
    data = data.normalize()

with timers('PCA'):
    data = data.PCA()

with timers('Nearest neighbors'):
    data = data.neighbors().shared_neighbors()  

with timers('Clustering (3 res.)'):
    data = data.cluster(resolution=[0.5, 1.0, 2.0])

with timers('Embedding'):
    data = data.embed(QC_column=None)

with timers('Plot embedding'):
    data.plot_embedding(    
        'cell_type', 
        f'sc-benchmarking/figures/brisc_embedding_{DATASET_NAME}.png', 
        cells_to_plot_column=None)

with timers('Find markers'):
    markers = data.find_markers('cell_type')

timers.print_summary(unit='s')

timers_df = timers.to_dataframe(sort=False, unit='s')\
    .with_columns(
        pl.lit('brisc').alias('library'),
        pl.lit('basic').alias('test'),
        pl.lit(DATASET_NAME).alias('dataset'),
        pl.lit('single-threaded' if NUM_THREADS == 1 else 'multi-threaded')
        .alias('num_threads'))
timers_df.write_csv(OUTPUT_PATH_TIME)

if not any(timers_df['aborted']):
    print('--- Completed successfully ---')

del data, markers, timers, timers_df
gc.collect()