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
OUTPUT_PATH_DOUBLET = sys.argv[5]

DATASET_NAME = 'SEAAD'
DATA_PATH = 'single-cell/SEAAD/SEAAD_raw.h5ad'
NUM_THREADS = -1

if __name__ == '__main__':

    system_info()
    print('--- Params ---')
    print('brisc qc')
    print(f'{DATA_PATH=}')
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
        data = data\
            .find_doublets(batch_column='sample')\
            .with_columns_obs(pl.col('passed_QC') & pl.col('doublet').not_())

    doublet_df = pl.DataFrame({
        'cell_id': data.obs['cell_id'],
        'doublet_score': data.obs['doublet_score'],
        'is_doublet': data.obs['doublet']
    })
    doublet_df.write_csv(OUTPUT_PATH_DOUBLET)

    timers.print_summary(unit='s')

    timers_df = timers.to_dataframe(sort=False, unit='s')\
        .with_columns(
            pl.lit('brisc').alias('library'),
            pl.lit('qc').alias('test'),
            pl.lit(DATASET_NAME).alias('dataset'),
            pl.lit('single-threaded' if NUM_THREADS == 1 else 'multi-threaded')
            .alias('num_threads'))
    timers_df.write_csv(OUTPUT_PATH_TIME)

    if not any(timers_df['aborted']):
        print('--- Completed successfully ---')

    del data, timers, timers_df
    gc.collect()
