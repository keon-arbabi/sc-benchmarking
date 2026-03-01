import sys
import polars as pl
sys.path.append('/home/karbabi')
from single_cell import SingleCell
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info

DATA_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
NUM_THREADS = int(sys.argv[3])
OUTPUT_PATH_TIME = sys.argv[4]
OUTPUT_PATH_EMBEDDING = sys.argv[5]

system_info()
print('--- Params ---')
print('brisc basic')
print(f'{DATA_PATH=}')
print(f'{NUM_THREADS=}')

if __name__ == '__main__':

    timers = MemoryTimer(silent=False)

    with timers('Load data'):
        data = SingleCell(DATA_PATH, num_threads=NUM_THREADS)

    with timers('Quality control'):
        data = data.qc(subset=False, allow_float=True)

    with timers('Feature selection'):
        data = data.hvg()

    with timers('Normalization'):
        data = data.normalize()

    with timers('PCA'):
        data = data.PCA()

    with timers('Nearest neighbors'):
        data = data\
            .neighbors()\
            .shared_neighbors()

    with timers('Clustering'):
        data = data.cluster(resolution=[0.5, 1.0, 2.0])

    with timers('Embedding'):
        data = data.embed()

    with timers('Plot embedding'):
        data.plot_embedding(
            'cell_type',
            f'sc-benchmarking/figures/brisc_embedding_{DATA_NAME}.png')

    with timers('Find markers'):
        markers = data.find_markers('cell_type')

    embedding_df = pl.DataFrame({
        'cell_id': data.obs['cell_id'],
        'embed_1': data.obsm['LocalMAP'][:, 0],
        'embed_2': data.obsm['LocalMAP'][:, 1],
        'cell_type': data.obs['cell_type']
    })
    embedding_df.write_csv(OUTPUT_PATH_EMBEDDING)

    timers.print_summary(unit='s')

    timers_df = timers\
        .to_dataframe(sort=False, unit='s')\
        .with_columns(
            pl.lit('brisc').alias('library'),
            pl.lit('basic').alias('test'),
            pl.lit(DATA_NAME).alias('dataset'),
            pl.lit(NUM_THREADS).alias('num_threads'))
    timers_df.write_csv(OUTPUT_PATH_TIME)

    if not any(timers_df['aborted']):
        print('--- Completed successfully ---')
