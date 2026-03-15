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
OUTPUT_PATH_PCS = sys.argv[6]
OUTPUT_PATH_NEIGHBORS = sys.argv[7]

if __name__ == '__main__':

    system_info()
    print('--- Params ---')
    print('brisc basic')
    print(f'{DATA_PATH=}')
    print(f'{NUM_THREADS=}')

    timers = MemoryTimer(silent=False)

    with timers('Load data'):
        data = SingleCell(DATA_PATH, num_threads=NUM_THREADS)

    with timers('Quality control'):
        data = data.qc(subset=False, allow_float=True)

    with timers('Feature selection'):
        data = data.hvg(batch_column='donor')

    with timers('Normalization'):
        data = data.normalize()

    with timers('PCA'):
        data = data.PCA()

    with timers('Nearest neighbors'):
        data = data.neighbors()
        data = data.shared_neighbors()

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

    data = data.filter_obs(pl.col('passed_QC'))

    # Save PCs
    pcs = data.obsm['PCs']
    pc_df = pl.DataFrame({
        f'PC_{i+1}': pcs[:, i] for i in range(pcs.shape[1])
    })
    pc_df.write_csv(OUTPUT_PATH_PCS)

    # Save neighbors
    neighbors = data.obsm['neighbors']
    neighbors_df = pl.DataFrame({
        f'neighbor_{i+1}': neighbors[:, i]
        for i in range(neighbors.shape[1])
    })
    neighbors_df.write_csv(OUTPUT_PATH_NEIGHBORS)

    # Save embeddings
    embedding_df = pl.DataFrame({
        'cell_id': data.obs['cell_id'],
        'embed_1': data.obsm['LocalMAP'][:, 0],
        'embed_2': data.obsm['LocalMAP'][:, 1],
        'cell_type': data.obs['cell_type'],
        'cell_type_broad': data.obs['cell_type_broad'],
        'cluster_res_0.5': data.obs['cluster_0'],
        'cluster_res_1.0': data.obs['cluster_1'],
        'cluster_res_2.0': data.obs['cluster_2'],
    })
    embedding_df.write_csv(OUTPUT_PATH_EMBEDDING)

    # Save timings
    timers_df = timers\
        .to_dataframe(sort=False, unit='s')\
        .with_columns(
            pl.lit('brisc').alias('library'),
            pl.lit('basic').alias('test'),
            pl.lit(DATA_NAME).alias('dataset'),
            pl.lit(NUM_THREADS).alias('num_threads'))
    timers_df.write_csv(OUTPUT_PATH_TIME)

    timers.print_summary(unit='s')

    if not any(timers_df['aborted']):
        print('--- Completed successfully ---')
