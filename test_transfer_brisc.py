import gc
import sys
import polars as pl
sys.path.append('/home/karbabi')
from single_cell import SingleCell
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info, transfer_accuracy

DATA_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
NUM_THREADS = int(sys.argv[3])
OUTPUT_PATH_TIME = sys.argv[4]
OUTPUT_PATH_ACC = sys.argv[5]

if __name__ == '__main__':

    system_info()
    print('--- Params ---')
    print('brisc transfer')
    print(f'{DATA_PATH=}')
    print(f'{NUM_THREADS=}')

    timers = MemoryTimer(silent=False)

    with timers('Load data'):
        data = SingleCell(DATA_PATH, num_threads=NUM_THREADS)

    with timers('Quality control'):
        data = data.qc(subset=False, allow_float=True)

    with timers('Split data'):
        data_query, data_ref = data.split_by_obs('is_ref').values()

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
            data_ref, 'cell_type',
            cell_type_column='cell_type_transferred')

    print('--- Transfer Accuracy ---')
    transfer_accuracy(
        data_query.obs, 'cell_type', 'cell_type_transferred')\
    .write_csv(OUTPUT_PATH_ACC)

    timers.print_summary(unit='s')

    timers_df = timers.to_dataframe(sort=False, unit='s')\
        .with_columns(
            pl.lit('brisc').alias('library'),
            pl.lit('transfer').alias('test'),
            pl.lit(DATA_NAME).alias('dataset'),
            pl.lit(NUM_THREADS).alias('num_threads'))
    timers_df.write_csv(OUTPUT_PATH_TIME)

    if not any(timers_df['aborted']):
        print('--- Completed successfully ---')
