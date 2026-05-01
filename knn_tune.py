import gc
import os
import sys
import time
import numpy as np
import polars as pl
os.environ['R_HOME'] = os.path.expanduser('~/miniforge3/lib/R')
from ryp import r, to_r
sys.path.append('/home/karbabi')
from single_cell import SingleCell

DATASETS = ['SEAAD', 'Parse', 'PanSci']
DATA_PATH = 'single-cell/{name}/{name}_raw.h5ad'
OUTPUT_DIR = 'sc-benchmarking/output'
FIGURES_DIR = 'sc-benchmarking/figures'

NUM_CLUSTERS_MULT = [1, 2, 4, 8]
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
done = {(r['dataset'], r['ncs'], r['num_clusters_mult'],
         r['num_kmeans_iterations'])
    for r in all_rows if r['method'] == 'brisc'}

for dataset_name in DATASETS:

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
            num_kmeans_iterations=None, dataset=dataset_name))
        pl.DataFrame(all_rows).write_csv(results_csv)

    log(f'[{dataset_name}] Preprocessing...')
    data_sc = SingleCell(DATA_PATH.format(name=dataset_name))\
        .qc(subset=True, allow_float=True)\
        .hvg(batch_column='donor').normalize().pca()
    num_cells = len(data_sc.obsm['pca'])

    for ncs in NUM_CLUSTERS_SEARCHED:
        for nc_mult in NUM_CLUSTERS_MULT:
            for kmi in KMEANS_ITERS:
                num_clusters = int(np.ceil(nc_mult * np.sqrt(num_cells)))
                if ncs > num_clusters:
                    continue
                if (dataset_name, ncs, nc_mult, kmi) in done:
                    log(f'  brisc nc={nc_mult}x ncs={ncs} kmi={kmi}... cached')
                    continue
                t0 = time.time()
                sc_nb = data_sc.neighbors(
                    num_clusters=num_clusters,
                    num_clusters_searched=ncs,
                    num_kmeans_iterations=kmi,
                    num_threads=NUM_THREADS,
                    overwrite=True,
                    verbose=False)
                elapsed = time.time() - t0
                nb = sc_nb.obsm['neighbors']
                if query_idx is not None:
                    nb = nb[query_idx]
                rc = recall_at_k(gt_neighbors, nb).mean()
                log(f'  brisc nc={nc_mult}x ncs={ncs} kmi={kmi}...'
                    f' recall={rc:.4f} time={elapsed:.1f}s')
                all_rows.append(dict(method='brisc', recall=float(rc),
                    time_s=elapsed, ncs=ncs,
                    num_clusters_mult=nc_mult, num_clusters=num_clusters,
                    num_kmeans_iterations=kmi,
                    dataset=dataset_name))
                pl.DataFrame(all_rows).write_csv(results_csv)

    del data_sc, gt_neighbors, query_idx
    gc.collect()

results_df = pl.read_csv('sc-benchmarking/output/knn_tune_clusters.csv')
to_r(results_df, 'df')
to_r(FIGURES_DIR, 'fig_dir')
r('''
suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
})

pdf(NULL)

ds_lab <- c(SEAAD = "SEAAD (1.2M cells)",
            Parse = "Parse PBMC (9.7M cells)",
            PanSci = "PanSci (22M cells)")
nc_palette <- c("#3b1f5e", "#7e2f8e", "#d8456a", "#f57e3a", "#fcd84a")
far_methods <- c("scanpy", "seurat")

base_theme <- theme_classic() +
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

make_fig <- function(mult_keep, out_name, show_baselines = TRUE) {
  brisc_src <- df %>% filter(method == "brisc")
  if (!is.null(mult_keep)) {
    brisc_src <- brisc_src %>%
      filter(num_clusters_mult %in% mult_keep)
  }
  nc_levels <- paste0(sort(unique(brisc_src$num_clusters_mult)),
    "x sqrt(n)")
  brisc <- brisc_src %>%
    mutate(
      nc_lab = factor(paste0(num_clusters_mult, "x sqrt(n)"),
        levels = nc_levels),
      iter_lab = paste0("iters=", num_kmeans_iterations),
      group = paste0(nc_lab, "_", iter_lab)
    ) %>%
    arrange(num_clusters_mult, num_kmeans_iterations, time_s)
  bl <- if (show_baselines) df %>% filter(method != "brisc")
        else df[0, ]

  clr <- c(setNames(nc_palette[seq_along(nc_levels)], nc_levels),
           "scanpy" = "#2ca02c",
           "seurat" = "#1f77b4",
           "rapids" = "#9467bd")
  color_breaks <- if (show_baselines)
    c(nc_levels, "scanpy", "seurat", "rapids") else nc_levels

  datasets <- unique(df$dataset)
  plots <- list()

  for (i in seq_along(datasets)) {
    ds <- datasets[i]
    b <- brisc %>% filter(dataset == ds)
    bl_ds <- bl %>% filter(dataset == ds)
    near <- bl_ds %>% filter(!method %in% far_methods)
    far <- bl_ds %>% filter(method %in% far_methods)

    x_lab <- if (i == length(datasets))
      "Time (seconds, log scale)" else NULL

    y_vals <- c(b$recall, bl_ds$recall)
    y_pad <- diff(range(y_vals)) * 0.05
    y_lim <- c(min(y_vals) - y_pad, min(1, max(y_vals) + y_pad))
    y_breaks <- pretty(y_lim, n = 10)

    p_near <- ggplot() +
      geom_line(data = b, aes(x = time_s, y = recall,
        group = group, color = nc_lab, linetype = iter_lab),
        linewidth = 0.6) +
      geom_point(data = b,
        aes(x = time_s, y = recall, color = nc_lab),
        size = 2) +
      geom_text(data = b, aes(x = time_s, y = recall,
        label = ncs, color = nc_lab),
        vjust = -0.8, size = 2.5, show.legend = FALSE) +
      geom_point(data = near, aes(x = time_s, y = recall,
        color = method), shape = 4, size = 5, stroke = 1.8) +
      geom_text(data = near, aes(x = time_s, y = recall,
        label = method, color = method),
        vjust = -1.2, size = 3, fontface = "bold",
        show.legend = FALSE) +
      scale_x_log10() +
      scale_y_continuous(limits = y_lim, breaks = y_breaks) +
      scale_color_manual(values = clr, name = NULL,
        breaks = color_breaks,
        drop = FALSE) +
      scale_linetype_manual(
        values = c("iters=1" = "solid", "iters=2" = "dashed"),
        name = "kmeans_iters") +
      labs(x = x_lab, y = "Recall@20", title = ds_lab[ds]) +
      base_theme +
      theme(plot.margin = margin(6, 0, 6, 6))

    if (nrow(far) == 0) {
      plots[[i]] <- p_near
      next
    }

    far_pad <- diff(range(log10(far$time_s))) * 0.5
    if (!is.finite(far_pad) || far_pad == 0) far_pad <- 0.15
    far_xlim <- 10 ^ (range(log10(far$time_s)) + c(-far_pad, far_pad))

    p_far <- ggplot() +
      geom_point(data = far, aes(x = time_s, y = recall,
        color = method), shape = 4, size = 5, stroke = 1.8) +
      geom_text(data = far, aes(x = time_s, y = recall,
        label = method, color = method),
        vjust = -1.2, size = 3, fontface = "bold",
        show.legend = FALSE) +
      scale_x_log10(limits = far_xlim) +
      scale_y_continuous(limits = y_lim, breaks = y_breaks) +
      scale_color_manual(values = clr, name = NULL,
        breaks = color_breaks,
        drop = FALSE) +
      scale_linetype_manual(
        values = c("iters=1" = "solid", "iters=2" = "dashed"),
        name = "kmeans_iters") +
      labs(x = NULL, y = NULL, title = NULL) +
      base_theme +
      theme(
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.line.y = element_blank(),
        plot.margin = margin(6, 6, 6, 0))

    plots[[i]] <- p_near + p_far +
      plot_layout(widths = c(5, 1.4))
  }

  combined <- wrap_plots(plots, ncol = 1) +
    plot_layout(guides = "collect") &
    theme(legend.position = "bottom")

  ggsave(paste0(fig_dir, "/", out_name),
    combined, width = 10, height = 3.5 * length(datasets))
}

make_fig(NULL, "knn_tune_clusters.png")
make_fig(c(1, 2, 4), "knn_tune_clusters_brisc_124x.png",
  show_baselines = FALSE)

invisible(dev.off())
''')
