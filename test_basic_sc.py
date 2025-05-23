import os
import gc
import sys
import polars as pl
import itertools
from single_cell import SingleCell

work_dir = 'projects/rrg-wainberg/karbabi/sc-benchmarking'
data_dir = 'single-cell/SEAAD/subsampled'

os.makedirs(f'{work_dir}/output', exist_ok=True)
os.makedirs(f'{work_dir}/figures', exist_ok=True)
sys.path.append(work_dir)

from utils_local import TimerCollection, system_info

system_info()

num_threads_options = [-1, 1]
subset_options = [True, False]
drop_X_options = [True, False]
size_options = ['20K', '400K', '1M']

all_timers = []

params = itertools.product(
    size_options, num_threads_options, subset_options, drop_X_options
)

for size, num_threads, subset, drop_X in params:
    timers = TimerCollection(silent=True)

    # Note: Loading is much slower from $scratch disk
    with timers('Load data'):
        data = SingleCell(
            f'{data_dir}/SEAAD_raw_{size}.h5ad',
            num_threads=num_threads)

    print(f'X num_threads: {data.X._num_threads}')

    # Note: QC filters are matched across libraries for timing, then
    # standardized by filtering to single_cell.py qc cells
    with timers('Quality control'):
        data.qc(
            subset=subset,
            max_mito_fraction=0.05,
            min_genes=100,
            nonzero_MALAT1=False,
            remove_doublets=False,
            allow_float=True,
            verbose=False,
            num_threads=num_threads)

    # Not timed
    if subset:
        data = data\
            .filter_obs(pl.col('tmp_passed_QC'))\
            .with_uns(QCed=True)
    else:
        data = data\
            .rename_obs({'tmp_passed_QC': 'passed_QC'})\
            .with_uns(QCed=True)

    with timers('Doublet detection'):
        data = data.find_doublets(
            batch_column='sample',
            num_threads=num_threads)

    print(f'cells: {data.shape[0]}, genes: {data.shape[1]}')

    with timers('Feature selection'):
        data = data.hvg(
            allow_float=True,
            num_threads=num_threads)

    with timers('Normalization'):
        data = data.normalize(
            allow_float=True,
            num_threads=num_threads)

    with timers('PCA'):
        data = data.PCA(num_threads=num_threads)

    # Not timed
    if not subset:
        data = data.filter_obs(pl.col('passed_QC'))
    if drop_X:
        X_copy = data.X.copy()
        data = data.drop_X()

    with timers('Neighbor graph'):
        with timers('KNN'):
            data = data.neighbors(num_threads=num_threads)
        with timers('SNN'):
            data = data.shared_neighbors(num_threads=num_threads)

    # TODO: The number of clusters needs to match across libraries

    with timers('Clustering (3 resolutions)'):
        data = data.cluster(
            resolution=[1, 0.5, 2],
            num_threads=num_threads)

    print(f'cluster_0: {len(data.obs["cluster_0"].unique())}')
    print(f'cluster_1: {len(data.obs["cluster_1"].unique())}')
    print(f'cluster_2: {len(data.obs["cluster_2"].unique())}')

    # TODO: Swap neighbor graphs with scanpy to assess
    # embedding differences

    with timers('Embedding'):
        data = data.embed(
            num_threads=num_threads)

    with timers('Plot embeddings'):
        data.plot_embedding(
            'cluster_0',
            f'{work_dir}/figures/sc_embedding_cluster_{size}.png')

    # Not timed
    if drop_X:
        data.X = X_copy

    with timers('Find markers'):
        markers = data.find_markers(
            'cluster_0',
            num_threads=num_threads)

    with timers('Save data'):
        data.save(
            f'{data_dir}/test_write.h5ad',
            overwrite=True)

    # Not timed
    os.remove(f'{data_dir}/test_write.h5ad')

    print('--- Params ---')
    print(f'{size=}, {num_threads=}, {subset=}, {drop_X=}')
    timers.print_summary(sort=False)

    df = timers\
        .to_dataframe(sort=False, unit='s')\
        .with_columns(
            pl.lit('basic').alias('vignette'),
            pl.lit(size).alias('size'),
            pl.lit(num_threads).alias('num_threads'),
            pl.lit(subset).alias('subset'),
            pl.lit(drop_X).alias('drop_X'))

    all_timers.append(df)
    del data, timers, df; gc.collect()

timers_df = pl.concat(all_timers)
timers_df.write_csv(f'{work_dir}/output/test_basic_sc_all.csv')

'''
--- System Information ---
Node: nl10603.narval.calcul.quebec
CPU: 64 physical cores, 64 logical cores
Memory: 1987.9 GB available / 2015.4 GB total

--- Params ---
size='20K', num_threads=-1, subset=True, drop_X=True

--- Timing Summary ---
Load data took 332ms 514µs (0.8%)
Quality control took 57ms 488µs (0.1%)
Doublet detection took 3s 837ms (8.7%)
Feature selection took 188ms 536µs (0.4%)
Normalization took 11ms 741µs (0.0%)
PCA took 33s 491ms (75.9%)
KNN took 929ms 249µs (2.1%)
SNN took 85ms 344µs (0.2%)
Neighbor graph took 1s 14ms (2.3%)
Clustering (3 resolutions) took 121ms 646µs (0.3%)
Embedding took 1s 112ms (2.5%)
Plot embeddings took 2s 36ms (4.6%)
Find markers took 75ms 189µs (0.2%)
Save data took 810ms 135µs (1.8%)

Total time: 44s 104ms


'''

