import gc
import sys
import polars as pl  
sys.path.append('/home/karbabi') 
from single_cell_keon import SingleCell  
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info

# DATASET_NAME = sys.argv[1]
# DATA_PATH = sys.argv[2]
# NUM_THREADS = int(sys.argv[3])
# OUTPUT_PATH = sys.argv[4]

# DATASET_NAME = 'PBMC'
# DATA_PATH = 'single-cell/PBMC/Parse_PBMC_raw.h5ad'
# NUM_THREADS = -1
# OUTPUT_PATH = 'sc-benchmarking/output/test_basic_brisc_PBMC_-1.csv'

DATASET_NAME = 'SEAAD'
DATA_PATH = 'single-cell/SEAAD/SEAAD_raw.h5ad'
NUM_THREADS = 1
OUTPUT_PATH = 'sc-benchmarking/output/test_basic_brisc_SEAAD_1.csv'

system_info()

print('--- Params ---')
print('brisc basic')
print(f'{sys.version=}')
print(f'{DATASET_NAME=}')
print(f'{NUM_THREADS=}')

timers = MemoryTimer(silent=True)

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
    data = data.filter_obs(pl.col('doublet').not_())

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
    data = data.embed()

with timers('Plot embedding'):
    data.plot_embedding(    
        'cell_type', 
        f'sc-benchmarking/figures/brisc_embedding_{DATASET_NAME}.png')

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
timers_df.write_csv(OUTPUT_PATH)

if not all(timers_df['aborted']):
    print('--- Completed successfully ---')

del timers, timers_df, data, markers
gc.collect()