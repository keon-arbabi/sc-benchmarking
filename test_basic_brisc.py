import sys
from pathlib import Path
import polars as pl

_PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_DIR))
sys.path.insert(0, str(_PROJECT_DIR.parent))
from single_cell import SingleCell
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

    timers = MemoryTimer(
        silent=False, csv_path=OUTPUT_PATH_TIME,
        csv_columns={'library': 'brisc', 'test': 'basic',
                     'dataset': DATA_NAME, 'num_threads': NUM_THREADS})

    with timers('Load data'):
        data = SingleCell(DATA_PATH, num_threads=NUM_THREADS)

    with timers('Quality control'):
        data = data.qc(subset=False, allow_float=True)

    with timers('Feature selection'):
        data = data.hvg(batch_column='donor')

    with timers('Normalization'):
        data = data.normalize()

    with timers('PCA'):
        data = data.pca()

    with timers('Nearest neighbors'):
        data = data.neighbors()
        data = data.shared_neighbors()

    with timers('Clustering'):
        data = data.cluster(resolution=[0.25, 0.5, 1.0, 1.5, 2.0])

    with timers('Embedding'):
        data = data.pacmap()

    with timers('Find markers'):
        markers = data.find_markers('cell_type')

    data = data.filter_obs(pl.col('passed_QC'))

    # Save PCs
    pcs = data.obsm['pca']
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
        'embed_1': data.obsm['pacmap'][:, 0],
        'embed_2': data.obsm['pacmap'][:, 1],
        'cell_type': data.obs['cell_type'],
        'cell_type_broad': data.obs['cell_type_broad'],
        'cluster_res_0.25': data.obs['cluster_0'],
        'cluster_res_0.5': data.obs['cluster_1'],
        'cluster_res_1.0': data.obs['cluster_2'],
        'cluster_res_1.5': data.obs['cluster_3'],
        'cluster_res_2.0': data.obs['cluster_4'],
    })
    embedding_df.write_csv(OUTPUT_PATH_EMBEDDING)

    timers.shutdown()
    print('--- Completed successfully ---')
