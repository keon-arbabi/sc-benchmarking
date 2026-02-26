import os
import sys
import time
import faiss
import numpy as np
import polars as pl
import scanpy as sc
from ryp import r, to_r, to_py
sys.path.append('/home/karbabi')
from single_cell import SingleCell

OUTPUT_DIR = 'sc-benchmarking/output'
DATASETS = {
    'SEAAD': 'single-cell/SEAAD/SEAAD_raw.h5ad',
    'PBMC': 'single-cell/PBMC/Parse_PBMC_raw.h5ad',
}
K = 20
BATCH_SIZE = 100_000
MAX_QUERIES = 300_000

def log(msg):
    print(msg, flush=True)

def exact_knn(PCs, k=20, batch_size=100_000, max_queries=None, seed=0):
    n, d = PCs.shape
    assert PCs.dtype == np.float32 and PCs.flags['C_CONTIGUOUS']
    faiss.omp_set_num_threads(os.cpu_count())
    index = faiss.IndexFlatL2(d)
    index.add(PCs)

    if max_queries is not None and n > max_queries:
        rng = np.random.default_rng(seed)
        query_idx = np.sort(
            rng.choice(n, size=max_queries, replace=False))
        queries = PCs[query_idx]
        nq = max_queries
    else:
        query_idx = None
        queries = PCs
        nq = n

    all_I = np.empty((nq, k + 1), dtype=np.int64)
    all_D = np.empty((nq, k + 1), dtype=np.float32)
    t0 = time.time()

    for b in range((nq + batch_size - 1) // batch_size):
        s = b * batch_size
        e = min(s + batch_size, nq)
        all_D[s:e], all_I[s:e] = index.search(queries[s:e], k + 1)
        log(f'  {e:,}/{nq:,} ({time.time() - t0:.0f}s)')

    expected = (query_idx if query_idx is not None else np.arange(nq))
    assert np.all(all_I[:, 0] == expected)
    return (all_I[:, 1:].astype(np.uint32),
            all_D[:, 1:].astype(np.float32), query_idx)

def recall_at_k(gt, approx):
    n, k = gt.shape
    assert approx.shape == (n, k)
    gt_s = np.sort(gt, axis=1)
    ap_s = np.sort(approx, axis=1)
    hits = np.zeros(n, dtype=np.int32)
    for j in range(k):
        hits += np.any(gt_s == ap_s[:, j:j + 1], axis=1)
    return hits.astype(np.float32) / k

all_summary_rows = []

for dataset_name, data_path in DATASETS.items():

    log(f'[{dataset_name}] Preprocessing...')
    data_sc = SingleCell(data_path, num_threads=-1)\
        .skip_qc()\
        .filter_obs(pl.col('_passed_QC'))\
        .hvg()\
        .normalize()\
        .PCA(num_PCs=50)
    PCs = data_sc.obsm['PCs']

    gt_cache = f'{OUTPUT_DIR}/knn_exact_{dataset_name}.npz'
    if os.path.exists(gt_cache):
        log(f'[{dataset_name}] Load exact KNN...')
        cached = np.load(gt_cache, allow_pickle=True)
        gt_neighbors = cached['gt_neighbors']
        query_idx = cached['query_idx']
        if query_idx.ndim == 0:
            query_idx = None
    else:
        log(f'[{dataset_name}] Calculate exact KNN...')
        gt_neighbors, gt_distances, query_idx = exact_knn(
            PCs, k=K, batch_size=BATCH_SIZE, max_queries=MAX_QUERIES)
        np.savez(
            gt_cache,
            gt_neighbors=gt_neighbors,
            gt_distances=gt_distances,
            query_idx=query_idx if query_idx is not None
            else np.array(None))

    log(f'[{dataset_name}] Neighbors scanpy...')
    data_ad = data_sc.to_scanpy()
    sc.pp.neighbors(data_ad, use_rep='PCs', n_pcs=50, n_neighbors=K)
    scanpy_neighbors = data_ad.obsp['distances'].indices\
        .reshape(data_ad.n_obs, K + 1)[:, 1:].astype(np.uint32)

    log(f'[{dataset_name}] Neighbors BPCells...')
    to_r(PCs, 'pca_mat')
    r('library(BPCells)')
    r(f'knn <- knn_hnsw(pca_mat, k = {K + 1})')
    bpcells_idx = to_py('knn$idx').to_numpy()
    n_cells = bpcells_idx.shape[0]
    self_found = bpcells_idx[:, 0] == np.arange(1, n_cells + 1)
    bpcells_neighbors = (
        np.where(self_found[:, None],
        bpcells_idx[:, 1:], bpcells_idx[:, :K]) - 1).astype(np.uint32)

    log(f'[{dataset_name}] Neighbors brisc...')
    data_sc = data_sc.neighbors(
        min_clusters_searched=100,
        num_kmeans_iterations=2,
        verbose=False)
    brisc_neighbors = data_sc.obsm['neighbors']

    if query_idx is not None:
        brisc_neighbors = brisc_neighbors[query_idx]
        scanpy_neighbors = scanpy_neighbors[query_idx]
        bpcells_neighbors = bpcells_neighbors[query_idx]

    brisc_rc = recall_at_k(gt_neighbors, brisc_neighbors)
    scanpy_rc = recall_at_k(gt_neighbors, scanpy_neighbors)
    bpcells_rc = recall_at_k(gt_neighbors, bpcells_neighbors)

    for name, rc in [('brisc', brisc_rc), ('scanpy', scanpy_rc),
                     ('BPCells', bpcells_rc)]:
        all_summary_rows.append(dict(
            method=name,
            dataset=dataset_name,
            mean=float(rc.mean()),
            se=float(rc.std() / np.sqrt(len(rc))),
            median=float(np.median(rc)),
            min=float(rc.min()),
            p5=float(np.quantile(rc, 0.05)),
            p25=float(np.quantile(rc, 0.25)),
            p75=float(np.quantile(rc, 0.75)),
            p95=float(np.quantile(rc, 0.95))))

    pl.DataFrame({
        'brisc': brisc_rc, 'scanpy': scanpy_rc, 'BPCells': bpcells_rc})\
    .write_parquet(
        f'{OUTPUT_DIR}/knn_recall_{dataset_name}.parquet')

summary_df = pl.DataFrame(all_summary_rows)\
    .write_csv(f'{OUTPUT_DIR}/knn_recall_summary.csv')
