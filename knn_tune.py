import gc
import os
import sys
import time
import numpy as np
import polars as pl
sys.path.append('/home/karbabi')
from single_cell import SingleCell
sys.path.append('/home/karbabi/sc-benchmarking')
from utils_local import print_df

DATASETS = ['SEAAD', 'Parse', 'PanSci']
DATA_PATH = 'single-cell/{name}/{name}_raw.h5ad'
OUTPUT_DIR = 'sc-benchmarking/output'
FIGURES_DIR = 'sc-benchmarking/figures'

NUM_CLUSTERS_MULT = [4, 8]
NUM_CLUSTERS_SEARCHED = [32, 64, 128]
KMEANS_ITERS = [1, 2]
NUM_THREADS = -1

def log(msg):
    print(msg, flush=True)

def recall_at_k(gt, approx):
    n, k = gt.shape
    assert approx.shape == (n, k)
    gt_s = np.sort(gt, axis=1)
    ap_s = np.sort(approx, axis=1)
    hits = np.zeros(n, dtype=np.int32)
    for j in range(k):
        hits += np.any(gt_s == ap_s[:, j:j + 1], axis=1)
    return hits.astype(np.float32) / k

results_csv = f'{OUTPUT_DIR}/knn_tune_clusters.csv'
recall_summary = pl.read_csv(f'{OUTPUT_DIR}/knn_recall_summary.csv')

all_rows = (pl.read_csv(results_csv).to_dicts()
    if os.path.exists(results_csv) else [])
done_bl = {(r['dataset'], r['method'])
    for r in all_rows if r['method'] != 'brisc'}
done = {(r['dataset'], r['ncs'], r['num_clusters_mult'])
    for r in all_rows if r['method'] == 'brisc'}

for dataset_name in DATASETS:

    log(f'[{dataset_name}] Preprocessing...')
    data_sc = SingleCell(DATA_PATH.format(name=dataset_name))\
        .qc(subset=True, allow_float=True)\
        .hvg(batch_column='donor').normalize().pca()
    num_cells = len(data_sc.obsm['pca'])

    gt_cache = f'{OUTPUT_DIR}/knn_exact_brisc_{dataset_name}.npz'
    cached = np.load(gt_cache, allow_pickle=True)
    gt_neighbors, query_idx = cached['gt_neighbors'], cached['query_idx']
    if query_idx.ndim == 0:
        query_idx = None

    for method in ('scanpy', 'seurat', 'rapids'):
        if (dataset_name, method) in done_bl:
            log(f'  {method}: cached')
            continue
        suffix = '_gpu' if method == 'rapids' else ''
        timer_path = (
            f'{OUTPUT_DIR}/basic_{method}_{dataset_name}{suffix}_timer.csv')
        if not os.path.exists(timer_path):
            log(f'  {method}: timer not found, skipping')
            continue
        recall_filtered = recall_summary.filter(
            (pl.col('method') == method) &
            (pl.col('dataset') == dataset_name))
        if recall_filtered.is_empty():
            log(f'  {method}: recall not found, skipping')
            continue
        time_s = float(pl.read_csv(timer_path)
            .filter(pl.col('operation') == 'Nearest neighbors')['duration'][0])
        recall = float(recall_filtered['mean'][0])
        log(f'  {method}: recall={recall:.4f} time={time_s:.1f}s')
        all_rows.append(dict(method=method, recall=recall, time_s=time_s,
            ncs=None, num_clusters_mult=None, num_clusters=None,
            dataset=dataset_name))
        pl.DataFrame(all_rows).write_csv(results_csv)

    for ncs in NUM_CLUSTERS_SEARCHED:
        for nc_mult in NUM_CLUSTERS_MULT:
            num_clusters = int(np.ceil(nc_mult * np.sqrt(num_cells)))
            if ncs > num_clusters:
                continue
            if (dataset_name, ncs, nc_mult) in done:
                log(f'  brisc nc={nc_mult}x ncs={ncs}... cached')
                continue
            t0 = time.time()
            sc_nb = data_sc.neighbors(
                num_clusters=num_clusters,
                num_clusters_searched=ncs,
                num_kmeans_iterations=KMEANS_ITERS,
                num_threads=NUM_THREADS,
                overwrite=True,
                verbose=False)
            elapsed = time.time() - t0
            nb = sc_nb.obsm['neighbors']
            if query_idx is not None:
                nb = nb[query_idx]
            rc = recall_at_k(gt_neighbors, nb).mean()
            log(f'  brisc nc={nc_mult}x ncs={ncs}...'
                f' recall={rc:.4f} time={elapsed:.1f}s')
            all_rows.append(dict(method='brisc', recall=float(rc),
                time_s=elapsed, ncs=ncs,
                num_clusters_mult=nc_mult, num_clusters=num_clusters,
                dataset=dataset_name))
            pl.DataFrame(all_rows).write_csv(results_csv)

    log(f'\n=== {dataset_name} ===')
    print_df(pl.DataFrame(all_rows).filter(pl.col('dataset') == dataset_name))

    del data_sc, gt_neighbors, query_idx
    gc.collect()
