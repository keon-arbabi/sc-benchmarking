import os
import sys
from pathlib import Path
import polars as pl
sys.path.append(f'{Path.home()}')
sys.path.append(f'{Path.home()}/sc-benchmarking')
from single_cell import SingleCell
from utils_local import MemoryTimer, system_info

GPU = os.uname().nodename.startswith('trig')
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
        data = data.hvg() if GPU else data.hvg(batch_column='donor')

    with timers('Normalization'):
        data = data.normalize()

    with timers('PCA'):
        data = data.pca(match_parallel=(NUM_THREADS == 1))

    with timers('Nearest neighbors'):
        data = data.neighbors().shared_neighbors()

    with timers('Clustering'):
        data = data.cluster(resolution=[0.25, 0.5, 1.0, 1.5, 2.0])

    with timers('Embedding (PaCMAP)'):
        data = data.pacmap()

    if not GPU:
        with timers('Embedding (LocalMAP)', exclude=True):
            data = data.localmap()

    if not GPU and NUM_THREADS != 1:
        with timers('Embedding (UMAP)', exclude=True):
            data = data.umap()

        with timers('Embedding (UMAP hogwild)', exclude=True):
            data = data.umap(
                hogwild=True, embedding_key='umap_hogwild')

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
        'cell_type': data.obs['cell_type'],
        'cell_type_broad': data.obs['cell_type_broad'],
        'cluster_res_0.25': data.obs['cluster_0'],
        'cluster_res_0.5': data.obs['cluster_1'],
        'cluster_res_1.0': data.obs['cluster_2'],
        'cluster_res_1.5': data.obs['cluster_3'],
        'cluster_res_2.0': data.obs['cluster_4'],
        'pacmap_1': data.obsm['pacmap'][:, 0],
        'pacmap_2': data.obsm['pacmap'][:, 1],
        'localmap_1': data.obsm['localmap'][:, 0],
        'localmap_2': data.obsm['localmap'][:, 1],
        **({'umap_1': data.obsm['umap'][:, 0],
            'umap_2': data.obsm['umap'][:, 1]}
           if 'umap' in data.obsm else {}),
        **({'umap_hogwild_1': data.obsm['umap_hogwild'][:, 0],
            'umap_hogwild_2': data.obsm['umap_hogwild'][:, 1]}
           if 'umap_hogwild' in data.obsm else {}),
    })
    embedding_df.write_csv(OUTPUT_PATH_EMBEDDING)

    timers.shutdown()
    print('--- Completed successfully ---')
