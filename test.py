import sys
import polars as pl
sys.path.append('/home/karbabi')
from single_cell import SingleCell, Timer

# sys.path.append('data-benchmarking')
# from utils_local import MemoryTimer
# timers = MemoryTimer(silent=False)

# DATA_PATH = '/scratch/karbabi/single-cell/SEAAD/SEAAD_raw.h5ad'
DATA_PATH = '/scratch/karbabi/single-cell/PBMC/Parse_PBMC_raw.h5ad'

for NUM_THREADS in [-1, 1]:
    data = SingleCell(DATA_PATH)

    with Timer(f'Quality control subset=True, num_threads={NUM_THREADS}'):
        data = data.qc(
            subset=True, allow_float=True, verbose=False,
            num_threads=NUM_THREADS)

    del data
    import gc; gc.collect()

    data = SingleCell(DATA_PATH)

    with Timer(f'Quality control'):
        data = data.qc(
            subset=False, allow_float=True, verbose=False,
            num_threads=NUM_THREADS)

# for NUM_THREADS in [-1, 1]:
#     with Timer('Load data'):
#         data = SingleCell(DATA_PATH, num_threads=NUM_THREADS)

#     data = data\
#         .filter_obs(pl.col('_passed_QC'))\
#         .skip_qc()

#     with Timer('Feature selection'):
#         data = data.hvg()

#     with Timer('Normalization'):
#         data = data.normalize()

#     with Timer('PCA'):
#         data = data.PCA()

#     with Timer('Nearest neighbors'):
#         data = data.neighbors(
#             min_clusters_searched=100,
#             num_kmeans_iterations=2)

#     with Timer('Shared neighbors'):
#         data = data.shared_neighbors()

#     with Timer('Embedding'):
#         data = data.embed()

#     with Timer('Plot embedding'):
#         data.plot_embedding('cell_type', 'scratch/localmap.png')

#     with Timer('Clustering (3 res.)'):
#         data = data.cluster(resolution=[0.5, 1, 2])

#     with Timer('Find markers'):
#         markers = data.find_markers('cell_type')

