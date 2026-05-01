import os
import time
import faiss
import numpy as np
import polars as pl
from pathlib import Path

OUTPUT_DIR = f'{Path.home()}/sc-benchmarking/output'
DATASETS = ['SEAAD', 'Parse', 'PanSci']
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
    'rapids': {
        'pcs': 'basic_rapids_{dataset}_gpu_pcs.csv',
        'neighbors': 'basic_rapids_{dataset}_gpu_neighbors.csv',
    },
}

K = 20
BATCH_SIZE = 500_000
MAX_QUERIES = None

def log(msg):
    print(msg, flush=True)

def exact_knn(PCs, k=20, batch_size=100_000, max_queries=None, seed=0):
    n, d = PCs.shape
    assert PCs.dtype == np.float32 and PCs.flags['C_CONTIGUOUS']
    if hasattr(faiss, 'omp_set_num_threads'):
        faiss.omp_set_num_threads(os.cpu_count())
    res = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatL2(d)
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    index.add(PCs)

    if max_queries is not None and n > max_queries:
        rng = np.random.default_rng(seed)
        query_idx = np.sort(rng.choice(n, size=max_queries, replace=False))
        queries = PCs[query_idx]
        nq = max_queries
    else:
        query_idx = None
        queries = PCs
        nq = n

    all_I = np.empty((nq, k + 1), dtype=np.int64)
    t0 = time.time()

    for b in range((nq + batch_size - 1) // batch_size):
        s = b * batch_size
        e = min(s + batch_size, nq)
        _, all_I[s:e] = index.search(queries[s:e], k + 1)
        log(f'  {e:,}/{nq:,} ({time.time() - t0:.0f}s)')

    expected = query_idx if query_idx is not None \
        else np.arange(nq, dtype=np.int64)
    self_mask = all_I == expected[:, None]
    valid = self_mask.any(axis=1)
    n_missing = int((~valid).sum())
    if n_missing:
        log(f'  WARN: {n_missing:,}/{nq:,} queries missing self in top-{k+1} '
            f'(excluded from recall)')
    drop = self_mask.argmax(axis=1)
    keep = np.ones((nq, k + 1), dtype=bool)
    keep[np.arange(nq), drop] = False
    return all_I[keep].reshape(nq, k).astype(np.uint32), query_idx, valid

def recall_at_k(gt, approx):
    n, k = gt.shape
    assert approx.shape == (n, k)
    gt_s = np.sort(gt, axis=1)
    ap_s = np.sort(approx, axis=1)
    is_first = np.empty_like(ap_s, dtype=bool)
    is_first[:, 0] = True
    is_first[:, 1:] = ap_s[:, 1:] != ap_s[:, :-1]
    n_dup_rows = int((~is_first).any(axis=1).sum())
    if n_dup_rows:
        log(f'  WARN: {n_dup_rows:,}/{n:,} rows have duplicate approx '
            f'indices (deduplicated for recall)')
    hits = np.zeros(n, dtype=np.int32)
    for j in range(k):
        hits += np.any(gt_s == ap_s[:, j:j + 1], axis=1) & is_first[:, j]
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

        log(f'[{dataset_name}/{lib_name}] Loading approximate neighbors...')
        approx_neighbors = pl.read_csv(nn_path).to_numpy().astype(np.uint32)
        k_lib = approx_neighbors.shape[1]

        gt_cache = os.path.join(
            OUTPUT_DIR, f'knn_exact_{lib_name}_{dataset_name}.npz')
        if os.path.exists(gt_cache):
            log(f'[{dataset_name}/{lib_name}] Loading cached exact KNN...')
            cached = np.load(gt_cache)
            gt_neighbors = cached['gt_neighbors']
            query_idx = cached['query_idx'] \
                if 'query_idx' in cached.files else None
            valid = cached['valid'] if 'valid' in cached.files \
                else np.ones(gt_neighbors.shape[0], dtype=bool)
        else:
            log(f'[{dataset_name}/{lib_name}] Loading PCs...')
            PCs = pl.read_csv(pc_path).to_numpy()
            PCs = np.ascontiguousarray(PCs, dtype=np.float32)

            log(f'[{dataset_name}/{lib_name}] Computing exact KNN '
                f'({PCs.shape[0]:,} cells x {PCs.shape[1]} PCs)...')
            gt_neighbors, query_idx, valid = exact_knn(
                PCs, k=K, batch_size=BATCH_SIZE, max_queries=MAX_QUERIES)

            saved = {'gt_neighbors': gt_neighbors, 'valid': valid}
            if query_idx is not None:
                saved['query_idx'] = query_idx
            np.savez(gt_cache, **saved)

        gt_k = gt_neighbors[:, :k_lib]
        if query_idx is not None:
            approx_neighbors = approx_neighbors[query_idx]

        n_invalid = int((~valid).sum())
        if n_invalid:
            log(f'[{dataset_name}/{lib_name}] Excluding {n_invalid:,}/'
                f'{valid.shape[0]:,} rows with missing self from recall')
            gt_k = gt_k[valid]
            approx_neighbors = approx_neighbors[valid]

        log(f'[{dataset_name}/{lib_name}] Computing recall@{k_lib}...')
        rc = recall_at_k(gt_k, approx_neighbors)

        rc_path = os.path.join(
            OUTPUT_DIR, f'knn_recall_{lib_name}_{dataset_name}.csv')
        pl.DataFrame({'recall': rc}).write_csv(rc_path)
        log(f'[{dataset_name}/{lib_name}] Saved per-cell recall to {rc_path}')

        all_summary_rows.append(dict(
            method=lib_name,
            dataset=dataset_name,
            k=k_lib,
            mean=float(rc.mean()),
            min=float(rc.min()),
            p5=float(np.quantile(rc, 0.05)),
            p25=float(np.quantile(rc, 0.25))))

        log(f'[{dataset_name}/{lib_name}] Recall@{k_lib}: '
            f'mean={rc.mean():.4f}, min={rc.min():.4f}')

if all_summary_rows:
    pl.DataFrame(all_summary_rows)\
        .write_csv(f'{OUTPUT_DIR}/knn_recall_summary.csv')
    log(f'Saved summary to {OUTPUT_DIR}/knn_recall_summary.csv')
