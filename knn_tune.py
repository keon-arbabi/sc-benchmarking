import contextlib
import gc
import io
import os
import re
import sys
import time
import numpy as np
import polars as pl
# from ryp import r, to_r
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

NUM_CLUSTERS_MULT = [4, 8]
MIN_CLUSTERS_SEARCHED = [24, 64, 100]
KMEANS_ITERS = [2]
NUM_THREADS = [-1]

def log(msg):
    print(msg, flush=True)

def parse_timer_s(text, name):
    m = re.search(rf'{name} took (.+)', text)
    if not m:
        return None
    units = {'d': 86400, 'h': 3600, 'm': 60, 's': 1,
             'ms': 1e-3, 'µs': 1e-6, 'ns': 1e-9}
    total = 0.0
    for part in m.group(1).split():
        for suffix, mult in units.items():
            if part.endswith(suffix) and part[:-len(suffix)].isdigit():
                total += int(part[:-len(suffix)]) * mult
                break
    return total

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
done = {(r['dataset'], r['num_threads'], r['kmeans_iters'],
         r['mcs'], r['num_clusters_mult'])
    for r in all_rows if r['method'] == 'brisc'}

for dataset_name, dataset_path in DATASETS.items():

    log(f'[{dataset_name}] Preprocessing...')
    data_sc = SingleCell(dataset_path)\
        .qc(allow_float=True).hvg(batch_column='donor').normalize().PCA()
    num_cells = len(data_sc.obsm['PCs'])

    gt_cache = f'{OUTPUT_DIR}/knn_exact_brisc_{dataset_name}.npz'
    cached = np.load(gt_cache, allow_pickle=True)
    gt_neighbors, query_idx = cached['gt_neighbors'], cached['query_idx']
    if query_idx.ndim == 0:
        query_idx = None

    for method in ('scanpy', 'seurat'):
        if (dataset_name, method) in done_bl:
            log(f'  {method}: cached')
            continue
        timer_path = f'{OUTPUT_DIR}/basic_{method}_{dataset_name}_timer.csv'
        time_s = float(pl.read_csv(timer_path)
            .filter(pl.col('operation') == 'Nearest neighbors')['duration'][0])
        recall = float(recall_summary
            .filter((pl.col('method') == method) &
                    (pl.col('dataset') == dataset_name))['mean'][0])
        log(f'  {method}: recall={recall:.4f} time={time_s:.1f}s')
        all_rows.append(dict(method=method, recall=recall, time_s=time_s,
            kmeans_s=None, knn_s=None, kmeans_iters=None, mcs=None,
            num_clusters_mult=None, num_clusters=None, num_threads=None,
            dataset=dataset_name))
        pl.DataFrame(all_rows).write_csv(results_csv)

    for nt in NUM_THREADS:
        for n_iter in KMEANS_ITERS:
            for mcs in MIN_CLUSTERS_SEARCHED:
                for nc_mult in NUM_CLUSTERS_MULT:
                    num_clusters = int(np.ceil(nc_mult * np.sqrt(num_cells)))
                    if mcs > num_clusters:
                        continue
                    if (dataset_name, nt, n_iter, mcs, nc_mult) in done:
                        log(f'  brisc {"MT" if nt == -1 else "ST"}'
                            f' nc={nc_mult}x iters={n_iter}'
                            f' mcs={mcs}... cached')
                        continue
                    t0 = time.time()
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        sc_nb = data_sc.neighbors(
                            num_clusters=num_clusters,
                            min_clusters_searched=mcs,
                            num_kmeans_iterations=n_iter,
                            num_threads=nt,
                            overwrite=True,
                            verbose=False)
                    sc_tmp = sc_nb.shared_neighbors()
                    elapsed = time.time() - t0
                    captured = buf.getvalue()
                    print(captured, end='', flush=True)
                    kmeans_s = parse_timer_s(captured, 'kmeans')
                    knn_s = parse_timer_s(captured, 'knn_self')
                    nb = sc_tmp.obsm['neighbors']
                    if query_idx is not None:
                        nb = nb[query_idx]
                    rc = recall_at_k(gt_neighbors, nb).mean()
                    nt_label = 'MT' if nt == -1 else 'ST'
                    log(f'  brisc {nt_label} nc={nc_mult}x iters={n_iter}'
                        f' mcs={mcs}... recall={rc:.4f} time={elapsed:.1f}s'
                        f' (kmeans={kmeans_s:.1f}s knn={knn_s:.1f}s)')
                    all_rows.append(dict(method='brisc', recall=float(rc),
                        time_s=elapsed, kmeans_s=kmeans_s, knn_s=knn_s,
                        kmeans_iters=n_iter, mcs=mcs,
                        num_clusters_mult=nc_mult, num_clusters=num_clusters,
                        num_threads=nt, dataset=dataset_name))
                    pl.DataFrame(all_rows).write_csv(results_csv)

    log(f'\n=== {dataset_name} ===')
    print_df(pl.DataFrame(all_rows).filter(pl.col('dataset') == dataset_name))

    del data_sc, gt_neighbors, query_idx
    gc.collect()

# results_csv = f'{OUTPUT_DIR}/knn_tune_clusters.csv'
# to_r(results_csv, 'csv_path')
# to_r(FIGURES_DIR, 'fig_dir')
# r('''
# suppressPackageStartupMessages({
#   library(tidyverse)
#   library(scales)
#   library(ggrepel)
#   library(patchwork)
# })

# pdf(NULL)
# df <- read_csv(csv_path, show_col_types = FALSE)

# ds_lab <- c(SEAAD = "SEAAD (1.2M cells)", PBMC = "Parse PBMC (9.7M cells)")

# brisc <- df %>%
#   filter(method == "brisc") %>%
#   mutate(
#     threading = ifelse(num_threads == -1, "MT", "ST"),
#     panel = paste0(ds_lab[dataset], " \u2014 ", threading),
#     nc_f = factor(num_clusters_mult,
#       levels = c(1, 2, 4, 8, 16),
#       labels = paste0(c(1, 2, 4, 8, 16), "x sqrt(n)")),
#     iters_f = factor(kmeans_iters,
#       labels = paste0("iters=", sort(unique(kmeans_iters)))),
#     alpha_val = ifelse(num_clusters_mult == 2, 1.0, 0.5)) %>%
#   arrange(mcs)

# panels_in_data <- unique(brisc$panel)
# bl <- df %>%
#   filter(method != "brisc") %>%
#   crossing(threading = c("ST", "MT")) %>%
#   mutate(panel = paste0(ds_lab[dataset], " \u2014 ", threading)) %>%
#   filter(panel %in% panels_in_data)

# nc_colors <- setNames(
#   viridis_pal(option = "C", begin = 0.15, end = 0.85)(5),
#   paste0(c(1, 2, 4, 8, 16), "x sqrt(n)"))
# bl_colors <- c(scanpy = "#2ca02c", seurat = "#1f77b4")

# iter_vals <- sort(unique(brisc$kmeans_iters))
# iter_labs <- paste0("iters=", iter_vals)
# iter_lty <- setNames(
#   c("solid", "dashed", "dotted")[seq_along(iter_vals)], iter_labs)
# iter_shape <- setNames(
#   c(16, 17, 15)[seq_along(iter_vals)], iter_labs)

# base_theme <- theme_classic() +
#   theme(
#     panel.grid.major.y = element_line(color = "grey70", linewidth = 0.4),
#     panel.grid.minor.y = element_line(
#       color = "grey90", linewidth = 0.2, linetype = "dashed"),
#     panel.grid.major.x = element_line(color = "grey70", linewidth = 0.4),
#     panel.grid.minor.x = element_line(
#       color = "grey90", linewidth = 0.2, linetype = "dashed"),
#     text = element_text(color = "black"),
#     axis.text = element_text(color = "black"),
#     strip.text = element_text(size = 9),
#     plot.margin = margin(6, 6, 6, 6))

# p1 <- ggplot() +
#   geom_line(data = brisc,
#     aes(x = time_s, y = recall,
#         group = interaction(nc_f, iters_f),
#         color = nc_f, linetype = iters_f, alpha = alpha_val),
#     linewidth = 0.65) +
#   geom_point(data = brisc,
#     aes(x = time_s, y = recall, color = nc_f,
#         shape = iters_f, alpha = alpha_val),
#     size = 1.8) +
#   geom_text_repel(data = brisc,
#     aes(x = time_s, y = recall, label = mcs, color = nc_f),
#     size = 2.2, max.overlaps = 20, show.legend = FALSE,
#     segment.color = "grey70", segment.size = 0.3,
#     min.segment.length = 0.2, seed = 0) +
#   geom_point(data = bl,
#     aes(x = time_s, y = recall, color = method),
#     size = 3, shape = 4, stroke = 1.2) +
#   geom_text_repel(data = bl,
#     aes(x = time_s, y = recall, label = method),
#     size = 2.8, fontface = "bold", seed = 0,
#     show.legend = FALSE) +
#   facet_wrap(~ panel, scales = "free") +
#   scale_color_manual(values = c(nc_colors, bl_colors),
#     name = "num_clusters") +
#   scale_alpha_identity() +
#   scale_linetype_manual(values = iter_lty, name = "kmeans_iters") +
#   scale_shape_manual(values = iter_shape, name = "kmeans_iters") +
#   scale_x_log10(labels = label_comma()) +
#   scale_y_continuous(limits = c(NA, 1), n.breaks = 12) +
#   labs(x = "Time (seconds, log scale)", y = "Recall@20") +
#   base_theme

# timing <- brisc %>%
#   filter(!is.na(kmeans_s) & !is.na(knn_s)) %>%
#   pivot_longer(c(kmeans_s, knn_s),
#     names_to = "component", values_to = "seconds") %>%
#   mutate(
#     component = factor(component,
#       levels = c("kmeans_s", "knn_s"),
#       labels = c("k-means", "knn search")),
#     bar_label = paste0(num_clusters_mult, "x/", mcs)) %>%
#   arrange(num_clusters_mult, mcs) %>%
#   mutate(bar_label = fct_inorder(bar_label))

# p2 <- ggplot(timing,
#     aes(x = bar_label, y = seconds, fill = component)) +
#   geom_col() +
#   facet_wrap(~ panel, scales = "free", nrow = 1) +
#   scale_fill_manual(
#     values = c("k-means" = "#e76f51", "knn search" = "#264653")) +
#   scale_y_continuous(labels = label_comma()) +
#   labs(x = "num_clusters_mult / min_clusters_searched",
#        y = "Time (seconds)", fill = NULL) +
#   base_theme +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6))

# p1 / p2 +
#   plot_layout(heights = c(3, 2), guides = "collect") &
#   theme(legend.position = "bottom", legend.box = "horizontal")

# ggsave(paste0(fig_dir, "/knn_tune_clusters.png"),
#   width = 12, height = 13)
# invisible(dev.off())
# ''')


