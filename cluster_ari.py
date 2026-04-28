import os
import numpy as np
import polars as pl
from pathlib import Path

OUTPUT_DIR = f'{Path.home()}/sc-benchmarking/output'
DATASETS = ['SEAAD', 'Parse', 'PanSci']
LIBRARIES = {
    'brisc': 'basic_brisc_{dataset}_-1_embedding.csv',
    'brisc_st': 'basic_brisc_{dataset}_1_embedding.csv',
    'scanpy': 'basic_scanpy_{dataset}_embedding.csv',
    'seurat': 'basic_seurat_{dataset}_embedding.csv',
    'rapids': 'basic_rapids_{dataset}_gpu_embedding.csv',
}
LABEL_COLS = ['cell_type', 'cell_type_broad']
RESOLUTIONS = ['0.25', '0.5', '1.0', '1.5', '2.0']
N_ITER = 10
SEED = 0

def log(msg):
    print(msg, flush=True)

def encode(arr):
    _, inv = np.unique(arr, return_inverse=True)
    return inv.astype(np.int32)

def fast_ari(a, b):
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    n = a.shape[0]
    flat = a.astype(np.int64) * nb + b.astype(np.int64)
    c = np.bincount(flat, minlength=na * nb)\
        .reshape(na, nb).astype(np.float64)
    ai, bj = c.sum(1), c.sum(0)
    sum_c = (c * (c - 1)).sum() / 2
    sum_a = (ai * (ai - 1)).sum() / 2
    sum_b = (bj * (bj - 1)).sum() / 2
    n_pairs = n * (n - 1) / 2
    expected = sum_a * sum_b / n_pairs
    max_idx = (sum_a + sum_b) / 2
    return float((sum_c - expected) / (max_idx - expected)) \
        if max_idx != expected else 0.0

def balanced_ari(it, ip, n_iter=10, seed=0):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(it, return_counts=True)
    n_per = int(counts.min())
    sort_idx = np.argsort(it, kind='stable')
    starts = np.zeros(len(classes) + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])

    scores = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        chunks = [
            rng.choice(sort_idx[starts[k]:starts[k + 1]],
                       size=n_per, replace=False)
            for k in range(len(classes))
        ]
        idx = np.concatenate(chunks)
        scores[i] = fast_ari(it[idx], ip[idx])
    return float(scores.mean()), float(scores.std()), n_per

all_summary_rows = []

for dataset_name in DATASETS:
    for lib_name, fname in LIBRARIES.items():
        path = os.path.join(
            OUTPUT_DIR, fname.format(dataset=dataset_name))
        if not os.path.exists(path):
            log(f'[{dataset_name}/{lib_name}] Embedding not found: {path}')
            continue

        cols = LABEL_COLS + [f'cluster_res_{r}' for r in RESOLUTIONS]
        log(f'[{dataset_name}/{lib_name}] Loading embedding...')
        df = pl.read_csv(path, columns=cols)

        for label_col in LABEL_COLS:
            it = encode(df[label_col].to_numpy())
            n_classes = int(it.max()) + 1
            for res in RESOLUTIONS:
                ip = encode(df[f'cluster_res_{res}'].to_numpy())
                n_clusters = int(ip.max()) + 1
                ari = fast_ari(it, ip)
                bari, bari_std, n_per = balanced_ari(
                    it, ip, n_iter=N_ITER, seed=SEED)

                all_summary_rows.append(dict(
                    method=lib_name,
                    dataset=dataset_name,
                    label=label_col,
                    resolution=float(res),
                    n_classes=n_classes,
                    n_clusters=n_clusters,
                    n_per_class=n_per,
                    ari=ari,
                    balanced_ari=bari,
                    balanced_ari_std=bari_std))

                log(f'[{dataset_name}/{lib_name}] {label_col} '
                    f'res={res}: ARI={ari:.4f}, '
                    f'balanced ARI={bari:.4f}+/-{bari_std:.4f} '
                    f'(n_classes={n_classes}, n_clusters={n_clusters}, '
                    f'n_per_class={n_per})')

if all_summary_rows:
    out_path = f'{OUTPUT_DIR}/cluster_ari_summary.csv'
    pl.DataFrame(all_summary_rows).write_csv(out_path)
    log(f'Saved summary to {out_path}')
