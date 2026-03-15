import os
import sys
import time
import colorsys
import hashlib
import itertools
import numpy as np
import polars as pl
from ryp import r, to_r
from collections import defaultdict
sys.path.append('/home/karbabi')
from single_cell import SingleCell

DATASETS = {
    'SEAAD': 'single-cell/SEAAD/SEAAD_raw.h5ad',
    'PBMC': 'single-cell/PBMC/Parse_PBMC_raw.h5ad',
}
OUTPUT_DIR = 'sc-benchmarking/output'
FIGURES_DIR = 'sc-benchmarking/figures'

DEFAULTS = dict(
    num_neighbors=10,
    num_mid_near_pairs=5,
    num_further_pairs=20,
    num_iterations=(100, 100, 250),
)
PARAM_SWEEPS = {
    'num_neighbors': [3, 5, 7, 10],
    'num_mid_near_pairs': [2, 5, 10, 15],
    'num_further_pairs': [5, 10, 20, 40],
    'num_iterations': [(50, 50, 150), (100, 100, 250), (200, 200, 500)],
}

def log(msg):
    print(msg, flush=True)

param_key = lambda p: frozenset((k, str(v)) for k, v in p.items())

COMBO_LETTERS = {param_key(p): chr(65 + i) for i, p in enumerate(
    [dict(DEFAULTS)] +
    [{**DEFAULTS, pn: v} for pn, vs in PARAM_SWEEPS.items()
     for v in vs if v != DEFAULTS[pn]])}

def cluster_color_map(cluster_col, broad_col, broad_base_colors, prefix):
    obs = pl.DataFrame({
        'cluster': cluster_col.cast(pl.String),
        'cell_type_broad': broad_col})
    cluster_broad = obs\
        .group_by(['cluster', 'cell_type_broad'])\
        .len()\
        .sort(['cluster', 'len'], descending=[False, True])\
        .unique('cluster', keep='first')\
        .select(['cluster', 'cell_type_broad'])
    broad_to_clusters = defaultdict(list)
    for row in cluster_broad.iter_rows(named=True):
        broad_to_clusters[row['cell_type_broad']].append(row['cluster'])
    colors = {}
    for broad_type, clusters in broad_to_clusters.items():
        base_h, base_l, base_s = colorsys.rgb_to_hls(
            *broad_base_colors[broad_type])
        n = len(clusters)
        for i, cl in enumerate(sorted(clusters, key=int)):
            l_ = 0.85 - (0.85 - base_l) * i / max(1, n - 1)
            r_, g_, b_ = colorsys.hls_to_rgb(base_h, l_, base_s)
            colors[f'{prefix}_{cl}'] = (
                f'#{int(r_ * 255):02x}'
                f'{int(g_ * 255):02x}'
                f'{int(b_ * 255):02x}')
    return colors

def embed_cache_path(dataset_name, params):
    key = str(sorted({k: str(v) for k, v in params.items()}.items()))
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return f'{OUTPUT_DIR}/embed_cache_{dataset_name}_{h}.npz'

def get_embedding(data, dataset_name, params, mem_cache):
    key = param_key(params)
    if key in mem_cache:
        return mem_cache[key]
    disk_path = embed_cache_path(dataset_name, params)
    changed = {k: v for k, v in params.items() if v != DEFAULTS[k]}
    label = ', '.join(f'{k}={v}' for k, v in changed.items()) or 'defaults'
    if os.path.exists(disk_path):
        log(f'[{dataset_name}] load embedding({label})')
        arr = np.load(disk_path)
        result = (arr['e1'], arr['e2'], float(arr['elapsed']))
        mem_cache[key] = result
        return result
    log(f'[{dataset_name}] calculate embedding({label})')
    t0 = time.time()
    coords = data.embed(**params, verbose=False).obsm['LocalMAP']
    elapsed = time.time() - t0
    result = (coords[:, 0].astype('float64'),
              coords[:, 1].astype('float64'), elapsed)
    np.savez(disk_path, e1=result[0], e2=result[1],
             elapsed=np.array(elapsed))
    mem_cache[key] = result
    return result

for dataset_name, data_path in DATASETS.items():
    log(f'[{dataset_name}] Preprocessing...')
    data = SingleCell(data_path, num_threads=-1)\
        .qc(subset=False, allow_float=True)\
        .hvg()\
        .normalize()\
        .PCA()\
        .neighbors().shared_neighbors()\
        .cluster(resolution=1)\
        .filter_obs(pl.col('passed_QC'))

    broad_types = sorted(data.obs['cell_type_broad'].unique().to_list())
    broad_base_colors = {
        bt: colorsys.hls_to_rgb(i / len(broad_types), 0.45, 0.70)
        for i, bt in enumerate(broad_types)
    }
    cluster_colors = cluster_color_map(
        data.obs['cluster'], data.obs['cell_type_broad'],
        broad_base_colors, prefix='brisc')
    obs_df = data.obs.select(
        (pl.lit('brisc_') + pl.col('cluster').cast(pl.String)).alias('cluster'),
        pl.col('cell_type_broad').cast(pl.String))
    mem_cache = {}

    all_rows = []
    ref_order = {'scanpy': 0, 'seurat': 1, 'brisc': 2}
    for ref_name in ('scanpy', 'seurat'):
        ref_path = (f'{OUTPUT_DIR}/basic_{ref_name}'
                    f'_{dataset_name}_embedding.csv')
        if not os.path.exists(ref_path):
            log(f'[{dataset_name}] Skipping {ref_name} (file not found)')
            continue
        log(f'[{dataset_name}] Loading {ref_name}...')
        ref_df = pl.read_csv(ref_path)
        ref_cluster_col = ref_df['cluster_res_1.0'].cast(pl.String)
        cluster_colors.update(cluster_color_map(
            ref_cluster_col, ref_df['cell_type_broad'],
            broad_base_colors, prefix=ref_name))
        all_rows.append(
            ref_df.select(
                pl.col('embed_1'),
                pl.col('embed_2'),
                (pl.lit(f'{ref_name}_') + pl.col('cluster_res_1.0')
                    .cast(pl.String)).alias('cluster'),
                pl.col('cell_type_broad'),
            ).with_columns(
                pl.lit('reference methods').alias('param_group'),
                pl.lit(ref_name).alias('panel_label'),
                pl.lit(ref_order[ref_name]).cast(pl.Int32)
                    .alias('value_order'),
                pl.lit(False).alias('is_default'),
                pl.lit('').alias('panel_id')))

    e1, e2, elapsed = get_embedding(
        data, dataset_name, dict(DEFAULTS), mem_cache)
    n_refs = sum(
        1 for row in all_rows
        if row['param_group'][0] == 'reference methods')
    all_rows.insert(n_refs,
        pl.DataFrame({
            'embed_1': e1, 'embed_2': e2,
            'cluster': obs_df['cluster'],
            'cell_type_broad': obs_df['cell_type_broad'],
        }).with_columns(
            pl.lit('reference methods').alias('param_group'),
            pl.lit('brisc (default)\n' +
                ', '.join(f'{k}={str(v).replace(" ","")}'
                for k, v in DEFAULTS.items()) +
                f'\n{elapsed:.1f}s').alias('panel_label'),
            pl.lit(ref_order['brisc']).cast(pl.Int32).alias('value_order'),
            pl.lit(True).alias('is_default'),
            pl.lit(COMBO_LETTERS[param_key(DEFAULTS)]).alias('panel_id')))

    for param_name, values in PARAM_SWEEPS.items():
        default_val = DEFAULTS[param_name]
        for order_idx, val in enumerate(
                v for v in values if v != default_val):
            params = {**DEFAULTS, param_name: val}
            e1, e2, elapsed = get_embedding(
                data, dataset_name, params, mem_cache)
            all_rows.append(
                pl.DataFrame({
                    'embed_1': e1, 'embed_2': e2,
                    'cluster': obs_df['cluster'],
                    'cell_type_broad': obs_df['cell_type_broad'],
                }).with_columns(
                    pl.lit(param_name).alias('param_group'),
                    pl.lit(f'{val}\n{elapsed:.1f}s').alias('panel_label'),
                    pl.lit(order_idx).cast(pl.Int32).alias('value_order'),
                    pl.lit(False).alias('is_default'),
                    pl.lit(COMBO_LETTERS[param_key(params)])
                        .alias('panel_id')))

    embed_data = pl.concat(all_rows)
    cluster_colors_df = pl.DataFrame({
        'cluster': list(cluster_colors.keys()),
        'color':   list(cluster_colors.values()),
    })
    to_r(embed_data, 'embed_data')
    to_r(cluster_colors_df, 'cluster_colors')
    to_r(dataset_name, 'dataset_name')
    to_r(FIGURES_DIR, 'fig_dir')

    r('''
    suppressPackageStartupMessages({
        library(tidyverse)
        library(patchwork)
    })
    pdf(NULL)
    clr <- setNames(cluster_colors$color, cluster_colors$cluster)
    param_groups <- unique(embed_data$param_group)
    group_plots <- lapply(param_groups, function(grp) {
        df <- embed_data %>%
            filter(param_group == grp) %>%
            mutate(panel_label = factor(
                panel_label,
                levels = unique(panel_label[order(value_order)])))
        df_labels  <- df %>%
            filter(panel_id != '') %>%
            distinct(panel_label, panel_id) %>%
            mutate(x = -Inf, y = Inf)
        ggplot(df, aes(embed_1, embed_2, color = cluster)) +
            geom_text(
                data = df_labels,
                aes(x = x, y = y, label = panel_id),
                hjust = -0.3, vjust = 1.5, size = 3.5,
                fontface = 'bold', color = 'black',
                inherit.aes = FALSE) +
            geom_point(size = 0.15, stroke = 0, alpha = 0.5) +
            scale_color_manual(values = clr, guide = 'none') +
            facet_wrap(~ panel_label, nrow = 1, scales = 'free') +
            labs(title = grp, x = NULL, y = NULL) +
            theme_bw() +
            theme(
                text = element_text(color = 'black'),
                plot.title = element_text(
                    size = 10, face = 'bold', hjust = 0),
                strip.text = element_text(size = 7.5, color = 'black'),
                strip.background = element_rect(fill = 'grey90'),
                axis.text = element_blank(),
                axis.ticks = element_blank(),
                panel.grid = element_blank(),
                plot.margin = margin(4, 4, 4, 4))
    })
    combined <- wrap_plots(
            lapply(group_plots, function(p) wrap_elements(full = p)),
            ncol = 1) +
        plot_annotation(
            title = paste('embed() hyperparameter sweep \u2014',
                          dataset_name),
            theme = theme(
                plot.title = element_text(size = 13, hjust = 0.5)))
    ggsave(
        paste0(fig_dir, '/embed_tune_', dataset_name, '.png'),
        combined,
        width = 16,
        height = 4.5 * length(group_plots),
        dpi = 300)
    invisible(dev.off())
    ''')
    log(f'[{dataset_name}] Saved embed_tune_{dataset_name}.png')

    for pA, pB in itertools.combinations(PARAM_SWEEPS.keys(), 2):
        rows = []
        for ri, vA in enumerate(PARAM_SWEEPS[pA]):
            for ci, vB in enumerate(PARAM_SWEEPS[pB]):
                params = {**DEFAULTS, pA: vA, pB: vB}
                e1, e2, elapsed = get_embedding(
                    data, dataset_name, params, mem_cache)
                rows.append(
                    pl.DataFrame({
                        'embed_1': e1, 'embed_2': e2,
                        'cluster': obs_df['cluster'],
                        'cell_type_broad': obs_df['cell_type_broad'],
                    }).with_columns(
                        pl.lit(str(vA)).alias('row_label'),
                        pl.lit(str(vB)).alias('col_label'),
                        pl.lit(ri).cast(pl.Int32).alias('row_order'),
                        pl.lit(ci).cast(pl.Int32).alias('col_order'),
                        pl.lit(vA == DEFAULTS[pA]).alias('is_row_default'),
                        pl.lit(vB == DEFAULTS[pB]).alias('is_col_default'),
                        pl.lit(
                            COMBO_LETTERS[param_key({**DEFAULTS, pA: vA})]
                            if vB == DEFAULTS[pB] else
                            COMBO_LETTERS[param_key({**DEFAULTS, pB: vB})]
                            if vA == DEFAULTS[pA] else
                            COMBO_LETTERS[param_key({**DEFAULTS, pA: vA})]
                            + '\u00d7'
                            + COMBO_LETTERS[param_key({**DEFAULTS, pB: vB})]
                        ).alias('panel_id'),
                        pl.lit(round(elapsed, 1)).alias('elapsed')))
        interact_id = f'{pA}_x_{pB}'
        n_row_vals = len(PARAM_SWEEPS[pA])
        n_col_vals = len(PARAM_SWEEPS[pB])
        to_r(pl.concat(rows), 'interact_data')
        to_r(f'{pA} \u00d7 {pB}', 'interact_title')
        to_r(interact_id, 'interact_id')
        to_r(n_row_vals, 'n_row_vals')
        to_r(n_col_vals, 'n_col_vals')
        r('''
        ref_df <- embed_data %>%
            filter(param_group == 'reference methods') %>%
            mutate(panel_label = factor(panel_label,
                levels = unique(panel_label[order(value_order)])))
        ref_labels  <- ref_df %>% filter(panel_id != '') %>%
            distinct(panel_label, panel_id) %>% mutate(x = -Inf, y = Inf)
        p_ref <- ggplot(ref_df, aes(embed_1, embed_2, color = cluster)) +
            geom_text(
                data = ref_labels, aes(x = x, y = y, label = panel_id),
                hjust = -0.3, vjust = 1.5, size = 3,
                fontface = 'bold', color = 'black', inherit.aes = FALSE) +
            geom_point(size = 0.1, stroke = 0, alpha = 0.4) +
            scale_color_manual(values = clr, guide = 'none') +
            facet_wrap(~ panel_label, nrow = 1, scales = 'free') +
            labs(title = 'reference methods', x = NULL, y = NULL) +
            theme_bw() +
            theme(
                text = element_text(color = 'black'),
                plot.title = element_text(size = 10, face = 'bold', hjust = 0),
                strip.text = element_text(size = 7.5, color = 'black'),
                strip.background = element_rect(fill = 'grey90'),
                axis.text = element_blank(),
                axis.ticks = element_blank(),
                panel.grid = element_blank(),
                plot.margin = margin(4, 4, 4, 4))
        df <- interact_data %>%
            mutate(
                row_label = factor(row_label,
                    levels = unique(row_label[order(row_order)])),
                col_label = factor(col_label,
                    levels = unique(col_label[order(col_order)])))
        df_labels <- df %>%
            distinct(row_label, col_label, panel_id, elapsed) %>%
            mutate(x = -Inf, y = Inf,
                   label = paste0(panel_id, '\n', elapsed, 's'))
        p <- ggplot(df, aes(embed_1, embed_2, color = cluster)) +
            geom_text(
                data = df_labels,
                aes(x = x, y = y, label = label),
                hjust = -0.3, vjust = 1.5, size = 3,
                fontface = 'bold', color = 'black',
                inherit.aes = FALSE) +
            geom_point(size = 0.1, stroke = 0, alpha = 0.4) +
            scale_color_manual(values = clr, guide = 'none') +
            facet_wrap(~ row_label + col_label, nrow = n_row_vals, scales = 'free') +
            labs(x = NULL, y = NULL) +
            theme_bw() +
            theme(
                text = element_text(color = 'black'),
                strip.text = element_text(size = 7.5, color = 'black'),
                strip.background = element_rect(fill = 'grey90'),
                axis.text = element_blank(),
                axis.ticks = element_blank(),
                panel.grid = element_blank())
        combined <- (p_ref / p + plot_layout(heights = c(1, n_row_vals))) +
            plot_annotation(
                title = interact_title,
                theme = theme(plot.title = element_text(
                    size = 11, face = 'bold', hjust = 0.5)))
        ggsave(
            paste0(fig_dir, '/embed_interact_', dataset_name,
                   '_', interact_id, '.png'),
            combined,
            width  = 3.5 * n_col_vals + 0.5,
            height = 3.5 * (n_row_vals + 1) + 1.0,
            dpi = 300)
        ''')
        log(f'[{dataset_name}] Saved'
            f' embed_interact_{dataset_name}_{interact_id}.png')
