from single_cell import SingleCell
from utils import Timer
import sys
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer

timers = MemoryTimer(silent=False)

# DATA_PATH = '/scratch/karbabi/single-cell/PBMC/Parse_PBMC_raw.h5ad'
DATA_PATH = '/scratch/karbabi/single-cell/SEAAD/SEAAD_raw.h5ad'
NUM_THREADS = -1

with Timer('Load data'):
    sc = SingleCell(DATA_PATH, num_threads=NUM_THREADS)

with Timer('Quality control'):
    sc = sc.qc(subset=False, remove_doublets=False, allow_float=True, verbose=False)

with Timer('Doublet detection'):
    sc = sc.find_doublets(batch_column='sample')

with Timer('Feature selection'):
    sc = sc.hvg()

with Timer('Normalization'):
    sc = sc.normalize()

with Timer('PCA'):
    sc = sc.PCA()

with Timer('Nearest neighbors'):
    sc = sc.neighbors()

with Timer('Shared neighbors'):
    sc = sc.shared_neighbors()

with Timer('Embedding'):
    sc = sc.embed()

with Timer('Plot embedding'):
    sc.plot_embedding('cell_type', 'scratch/localmap.png')

with Timer('Clustering (3 res.)'):
    sc = sc.cluster(resolution=[0.5, 1, 2])

with Timer('Find markers'):
    markers = sc.find_markers('cell_type')

