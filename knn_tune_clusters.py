import gc
import os
import sys
import time
import numpy as np
import polars as pl
import scanpy as sc
from ryp import r, to_r, to_py
sys.path.append('/home/karbabi')
from single_cell import SingleCell
sys.path.append('/home/karbabi/sc-benchmarking')
from utils_local import print_df

DATASETS = {
    'SEAAD': 'single-cell/SEAAD/SEAAD_raw.h5ad',
    'PBMC': 'single-cell/PBMC/Parse_PBMC_raw.h5ad',
}
OUTPUT_DIR = 'sc-benchmarking/output'
FIGURES_DIR = 'sc-benchmarking/figures'
K = 20

# num_clusters 1,2,4,8,16 * sq root n

MIN_CLUSTERS_SEARCHED = [10, 20, 50, 100]
KMEANS_ITERS = [1, 2, 4]
NUM_THREADS = [1, -1]

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

all_rows = []

for dataset_name, data_path in DATASETS.items():

    log(f'[{dataset_name}] Preprocessing...')
    data_sc = SingleCell(data_path, num_threads=-1)\
        .filter_obs(pl.col('_passed_QC'))\
        .skip_qc()\
        .hvg()\
        .normalize()\
        .PCA(num_PCs=50)
    PCs = data_sc.obsm['PCs']

    gt_cache = f'{OUTPUT_DIR}/knn_exact_{dataset_name}.npz'
    assert os.path.exists(gt_cache), \
        f'Run knn_comparison.py first to generate {gt_cache}'
    cached = np.load(gt_cache, allow_pickle=True)
    gt_neighbors, query_idx = (
        cached['gt_neighbors'], cached['query_idx'])
    if query_idx.ndim == 0:
        query_idx = None

    log(f'[{dataset_name}] scanpy neighbors...')
    t0 = time.time()
    data_ad = data_sc.to_scanpy()
    sc.pp.neighbors(
        data_ad, use_rep='PCs', n_pcs=50, n_neighbors=K)
    scanpy_neighbors = (data_ad.obsp['distances'].indices
        .reshape(data_ad.n_obs, K + 1)[:, 1:]
        .astype(np.uint32))
    scanpy_time = time.time() - t0
    log(f'  scanpy: {scanpy_time:.1f}s')
    del data_ad
    if query_idx is not None:
        scanpy_neighbors = scanpy_neighbors[query_idx]
    scanpy_recall = recall_at_k(
        gt_neighbors, scanpy_neighbors).mean()
    del scanpy_neighbors
    all_rows.append(dict(
        method='scanpy', recall=float(scanpy_recall),
        time_s=scanpy_time, kmeans_iters=None,
        min_clusters=None, num_threads=None,
        dataset=dataset_name))

    log(f'[{dataset_name}] BPCells neighbors...')
    t0 = time.time()
    to_r(PCs, 'pca_mat')
    r('library(BPCells)')
    r(f'knn <- knn_hnsw(pca_mat, k = {K + 1})')
    bpcells_idx = to_py('knn$idx').to_numpy()
    self_found = (bpcells_idx[:, 0] ==
        np.arange(1, bpcells_idx.shape[0] + 1))
    bpcells_neighbors = (np.where(
        self_found[:, None], bpcells_idx[:, 1:],
        bpcells_idx[:, :K]) - 1).astype(np.uint32)
    bpcells_time = time.time() - t0
    log(f'  BPCells: {bpcells_time:.1f}s')
    del bpcells_idx, self_found
    r('rm(pca_mat, knn); gc()')
    if query_idx is not None:
        bpcells_neighbors = bpcells_neighbors[query_idx]
    bpcells_recall = recall_at_k(
        gt_neighbors, bpcells_neighbors).mean()
    del bpcells_neighbors
    all_rows.append(dict(
        method='BPCells', recall=float(bpcells_recall),
        time_s=bpcells_time, kmeans_iters=None,
        min_clusters=None, num_threads=None,
        dataset=dataset_name))

    for nt in NUM_THREADS:
        nt_label = 'MT' if nt == -1 else 'ST'
        for n_iter in KMEANS_ITERS:
            for mcs in MIN_CLUSTERS_SEARCHED:
                t0 = time.time()
                sc_tmp = data_sc.neighbors(
                    num_neighbors=K,
                    min_clusters_searched=mcs,
                    num_kmeans_iterations=n_iter,
                    num_threads=nt,
                    overwrite=True, verbose=False)
                elapsed = time.time() - t0
                nb = sc_tmp.obsm['neighbors']
                if query_idx is not None:
                    nb = nb[query_idx]
                rc = recall_at_k(gt_neighbors, nb).mean()
                log(f'  brisc {nt_label} iters={n_iter} '
                    f'mcs={mcs}... recall={rc:.4f} '
                    f'time={elapsed:.1f}s')
                all_rows.append(dict(
                    method='brisc', recall=float(rc),
                    time_s=elapsed, kmeans_iters=n_iter,
                    min_clusters=mcs, num_threads=nt,
                    dataset=dataset_name))

    log(f'\n=== {dataset_name} ===')
    print_df(pl.DataFrame(all_rows).filter(
        pl.col('dataset') == dataset_name))

    del data_sc, PCs, gt_neighbors, query_idx
    gc.collect()

results_csv = f'{OUTPUT_DIR}/knn_tune_clusters.csv'
pl.DataFrame(all_rows).write_csv(results_csv)

to_r(results_csv, 'csv_path')
to_r(FIGURES_DIR, 'fig_dir')
r('''
suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
})

pdf(NULL)
df <- read_csv(csv_path, show_col_types = FALSE)
brisc <- df %>%
  filter(method == "brisc") %>%
  mutate(threading = ifelse(num_threads == -1, "MT", "ST"),
         group = paste0("brisc (iters=", kmeans_iters, ")"))
bl <- df %>%
  filter(method != "brisc") %>%
  cross_join(tibble(threading = c("ST", "MT")))

ds_lab <- c(SEAAD = "SEAAD (1.2M cells)",
            PBMC = "Parse PBMC (9.7M cells)")
clr <- c("scanpy" = "#2ca02c",
         "BPCells" = "#1f77b4",
         "brisc (iters=1)" = "#ff7f0e",
         "brisc (iters=2)" = "#d62728",
         "brisc (iters=6)" = "#e377c2")

datasets <- unique(df$dataset)
plots <- list()

for (i in seq_along(datasets)) {
  ds <- datasets[i]
  b <- brisc %>% filter(dataset == ds)
  bl_ds <- bl %>% filter(dataset == ds)

  plots[[i]] <- ggplot() +
    geom_hline(data = bl_ds, aes(yintercept = recall,
      linetype = method, color = method),
      linewidth = 1) +
    geom_line(data = b, aes(x = time_s, y = recall,
      group = group, color = group), linewidth = 1) +
    geom_point(data = b,
      aes(x = time_s, y = recall, color = group),
      size = 2) +
    geom_text(data = b, aes(x = time_s, y = recall,
      label = min_clusters),
      vjust = -0.8, size = 2.5) +
    facet_wrap(~ threading, scales = "free_x") +
    scale_y_continuous(limits = c(NA, 1), n.breaks = 15) +
    scale_color_manual(values = clr, name = NULL) +
    scale_linetype_manual(
      values = c(scanpy = "dashed", BPCells = "dashed"),
      guide = "none") +
    labs(x = if (i == length(datasets))
        "Time (seconds)" else NULL,
        y = "Recall@20", title = ds_lab[ds]) +
    theme_classic() +
    theme(
      panel.grid.major.y = element_line(color = "grey70", linewidth = 0.5),
      panel.grid.minor.y = element_line(color = "grey90", linewidth = 0.25,
        linetype = "dashed"),
      panel.grid.major.x = element_line(color = "grey70", linewidth = 0.5),
      panel.grid.minor.x = element_line(color = "grey90", linewidth = 0.25,
        linetype = "dashed"),
      text = element_text(color = "black"),
      axis.text = element_text(color = "black"),
      plot.title = element_text(size = 11, hjust = 0.5),
      plot.margin = margin(6, 6, 6, 6))
}

combined <- plots[[1]] / plots[[2]] +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

ggsave(paste0(fig_dir, "/knn_tune_clusters.png"),
  combined, width = 10, height = 7)
invisible(dev.off())
''')
