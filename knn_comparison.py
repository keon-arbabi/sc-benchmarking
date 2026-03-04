import os
import time
import faiss
import numpy as np
import polars as pl

OUTPUT_DIR = 'sc-benchmarking/output'
DATASETS = ['PBMC', 'SEAAD']
LIBRARIES = {
    'brisc': {
        'pcs': 'basic_brisc_{dataset}_-1_pcs.csv',
        'neighbors': 'basic_brisc_{dataset}_-1_neighbors.csv',
    },
    'scanpy': {
        'pcs': 'basic_scanpy_{dataset}_pcs.csv',
        'neighbors': 'basic_scanpy_{dataset}_neighbors.csv',
    },
    'seurat': {
        'pcs': 'basic_seurat_{dataset}_pcs.csv',
        'neighbors': 'basic_seurat_{dataset}_neighbors.csv',
    },
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

for dataset_name in DATASETS:
    for lib_name, paths in LIBRARIES.items():
        pc_path = os.path.join(
            OUTPUT_DIR, paths['pcs'].format(dataset=dataset_name))
        nn_path = os.path.join(
            OUTPUT_DIR, paths['neighbors'].format(dataset=dataset_name))

        if not os.path.exists(pc_path):
            log(f'[{dataset_name}/{lib_name}] PCs not found: {pc_path}')
            continue
        if not os.path.exists(nn_path):
            log(f'[{dataset_name}/{lib_name}] Neighbors not found: {nn_path}')
            continue

        # Load approximate neighbors
        log(f'[{dataset_name}/{lib_name}] Loading approximate neighbors...')
        approx_neighbors = pl.read_csv(nn_path).to_numpy().astype(np.uint32)
        k_lib = approx_neighbors.shape[1]

        # Compute or load exact KNN
        gt_cache = os.path.join(
            OUTPUT_DIR, f'knn_exact_{lib_name}_{dataset_name}.npz')
        if os.path.exists(gt_cache):
            log(f'[{dataset_name}/{lib_name}] Loading cached exact KNN...')
            cached = np.load(gt_cache, allow_pickle=True)
            gt_neighbors = cached['gt_neighbors']
            query_idx = cached['query_idx']
            if query_idx.ndim == 0:
                query_idx = None
        else:
            log(f'[{dataset_name}/{lib_name}] Loading PCs...')
            PCs = pl.read_csv(pc_path).to_numpy()
            PCs = np.ascontiguousarray(PCs, dtype=np.float32)

            log(f'[{dataset_name}/{lib_name}] Computing exact KNN '
                f'({PCs.shape[0]:,} cells x {PCs.shape[1]} PCs)...')
            gt_neighbors, gt_distances, query_idx = exact_knn(
                PCs, k=K, batch_size=BATCH_SIZE, max_queries=MAX_QUERIES)

            np.savez(
                gt_cache,
                gt_neighbors=gt_neighbors,
                gt_distances=gt_distances,
                query_idx=query_idx if query_idx is not None
                else np.array(None))

        # Truncate exact KNN to match library's K if needed
        gt_k = gt_neighbors[:, :k_lib]

        # Subsample to query cells if dataset was too large
        if query_idx is not None:
            approx_neighbors = approx_neighbors[query_idx]

        # Compute recall
        log(f'[{dataset_name}/{lib_name}] Computing recall@{k_lib}...')
        rc = recall_at_k(gt_k, approx_neighbors)

        all_summary_rows.append(dict(
            method=lib_name,
            dataset=dataset_name,
            k=k_lib,
            mean=float(rc.mean()),
            se=float(rc.std() / np.sqrt(len(rc))),
            median=float(np.median(rc)),
            min=float(rc.min()),
            p5=float(np.quantile(rc, 0.05)),
            p25=float(np.quantile(rc, 0.25)),
            p75=float(np.quantile(rc, 0.75)),
            p95=float(np.quantile(rc, 0.95))))

        log(f'[{dataset_name}/{lib_name}] Recall@{k_lib}: '
            f'mean={rc.mean():.4f}, median={np.median(rc):.4f}')

if all_summary_rows:
    pl.DataFrame(all_summary_rows)\
        .write_csv(f'{OUTPUT_DIR}/knn_recall_summary.csv')
    log(f'Saved summary to {OUTPUT_DIR}/knn_recall_summary.csv')
