suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(patchwork)
})

work_dir <- file.path(path.expand("~"), "sc-benchmarking")

results <- list.files(
  file.path(work_dir, "output"), "_timer\\.csv$", full.names = TRUE) %>%
  map(fread) %>% rbindlist(fill = TRUE) %>% as_tibble()

grp <- list(
  order = c("Seurat", "scanpy", "brisc (ST)", "brisc (MT)"),
  colors = c("brisc (MT)" = "#d62728", "brisc (ST)" = "#e8726a",
             "scanpy" = "#8fbc8f", "Seurat" = "#a8c4dc"))

datasets <- c("SEAAD", "PBMC", "PANSCI")
ds_lab <- c(
  SEAAD = "1.2M cells (SEAAD)",
  PBMC = "9.7M cells (Parse PBMC)",
            PANSCI = "20.3M cells (PanSci)")
ds_unit <- c(SEAAD = "s", PBMC = "h", PANSCI = "m")
ds_div <- c(SEAAD = 1, PBMC = 3600, PANSCI = 60)

wf_lab <- c(basic = "Basic workflow", transfer = "Label transfer",
            de = "Differential expression")

op_order <- list(
  basic = c(
    "Load data", "Quality control", "Normalization",
    "Feature selection", "PCA", "Nearest neighbors",
    "Clustering", "Embedding", "Find markers"),
  transfer = c("Load data", "Quality control", "Split data",
               "Normalization", "Feature selection", "PCA",
               "Transfer labels"),
  de = c("Load data", "Quality control", "Pseudobulk",
         "Filter", "Differential expression"),
  commands = c("Get expression by cell", "Get expression by gene",
               "Subset to one cell type",
               "Subset to highly variable genes",
               "Subsample to 10,000 cells",
               "Select categorical columns",
               "Split by cell type", "Concatenate cell types"))

fmt_time <- function(s) {
  case_when(s >= 3600 ~ paste0(round(s / 3600, 1), "h"),
            s >= 60   ~ paste0(round(s / 60, 1), "m"),
            s >= 0.1  ~ paste0(round(s, 1), "s"),
            .default  = paste0(round(s * 1000), "ms"))
}

make_label <- function(time_s, ref_s) {
  lbl <- fmt_time(time_s)
  if_else(!is.na(ref_s) & ref_s > 0,
          paste0(lbl, " (", round(time_s / ref_s, 1), "x)"), lbl)
}

base_theme <- theme_bw() + theme(
  text = element_text(color = "black"),
  axis.text = element_text(color = "black"),
  plot.title = element_text(size = 11, hjust = 0.5),
  panel.grid = element_blank(),
  plot.margin = margin(6, 6, 6, 6))

# --- Data preparation --------------------------------------------------------

prepared <- results %>%
  mutate(
    workflow = recode(test, manipulation = "commands") %>%
      factor(levels = c(names(wf_lab), "commands")),
    dataset = factor(dataset, datasets),
    group = case_when(
      library == "brisc" & num_threads == -1 ~ "brisc (MT)",
      library == "brisc" ~ "brisc (ST)",
      library == "scanpy" ~ "scanpy",
      .default = "Seurat") %>%
      factor(levels = grp$order)) %>%
  filter(library != "rapids")

# --- Total runtime figure ----------------------------------------------------

ref_total <- prepared %>%
  filter(workflow %in% names(wf_lab), group == "brisc (MT)") %>%
  summarize(ref = sum(duration), .by = c(workflow, dataset))

totals <- prepared %>%
  filter(workflow %in% names(wf_lab)) %>%
  summarize(total_s = sum(duration), .by = c(group, workflow, dataset)) %>%
  left_join(ref_total, by = c("workflow", "dataset")) %>%
  mutate(ref = if_else(group == "brisc (MT)", NA_real_, ref),
         dur_plot = total_s / ds_div[as.character(dataset)],
         label = make_label(total_s, ref))

peak_mem <- prepared %>%
  filter(workflow %in% names(wf_lab)) %>%
  summarize(memory = max(memory), .by = c(group, workflow, dataset))

plot_bar <- function(wf, ds) {
  totals %>%
    filter(workflow == wf, dataset == ds) %>%
    ggplot(aes(dur_plot, group, fill = group)) +
    geom_col(width = 0.7) +
    geom_text(aes(label = label),
              hjust = -0.05, size = 2.5, na.rm = TRUE) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.35))) +
    scale_y_discrete(drop = FALSE) +
    scale_fill_manual(values = grp$colors, guide = "none") +
    labs(title = if (wf == "basic") ds_lab[ds],
         x = if (wf == "de") paste0("Duration (", ds_unit[ds], ")"),
         y = if (ds == "SEAAD") wf_lab[wf]) +
    base_theme + theme(
      axis.title.y = element_text(size = 11, margin = margin(r = 10)),
      axis.text.y = if (ds != "SEAAD") element_blank()
                    else element_text(),
      axis.ticks.y = if (ds != "SEAAD") element_blank()
                     else element_line())
}

plot_mem <- function(wf, ds) {
  peak_mem %>%
    filter(workflow == wf, dataset == ds) %>%
    ggplot(aes(memory, group)) +
    geom_col(fill = "#bdbdbd", width = 0.7) +
    geom_text(aes(x = 0, label = paste0(round(memory, 1), " GiB")),
              hjust = -0.05, size = 2.5) +
    scale_x_continuous(breaks = scales::breaks_pretty(3),
                       expand = expansion(mult = c(0, 0.5))) +
    scale_y_discrete(drop = FALSE) +
    labs(title = if (wf == "basic") "Peak memory",
         x = if (wf == "de") "GiB", y = NULL) +
    base_theme + theme(
      axis.text.y = element_blank(), axis.ticks.y = element_blank())
}

fig_totals <- map(names(wf_lab), \(wf) {
  map(datasets, \(ds) list(plot_bar(wf, ds), plot_mem(wf, ds))) %>%
    list_flatten()
}) %>% list_flatten() %>%
  wrap_plots(ncol = 6, byrow = TRUE, widths = rep(c(4, 1), 3))

ggsave(file.path(work_dir, "figures", "fig_total_runtime.png"),
       fig_totals, width = 15, height = 6)

# --- Per-step figures --------------------------------------------------------

step_ref <- prepared %>%
  filter(group == "brisc (MT)") %>%
  select(operation, workflow, dataset, ref = duration)

steps <- prepared %>%
  left_join(step_ref, by = c("operation", "workflow", "dataset")) %>%
  mutate(ref = if_else(group == "brisc (MT)", NA_real_, ref),
         dur_plot = duration / ds_div[as.character(dataset)],
         label = make_label(duration, ref))

step_theme <- base_theme + theme(
  plot.margin = margin(0, 6, 0, 6),
  axis.text.x = element_blank(), axis.ticks.x = element_blank())

plot_step <- function(d, ds, op, is_first, show_leg = FALSE) {
  d %>%
    filter(dataset == ds) %>%
    ggplot(aes(dur_plot, group, fill = group)) +
    geom_col(width = 0.8) +
    geom_text(aes(label = label),
              hjust = -0.05, size = 2, na.rm = TRUE) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.35))) +
    scale_y_discrete(drop = FALSE) +
    scale_fill_manual(
      values = grp$colors, limits = grp$order, name = NULL,
      guide = if (show_leg) guide_legend(reverse = TRUE)
              else "none") +
    labs(title = if (is_first) ds_lab[ds], x = NULL,
         y = if (ds == "SEAAD") op) +
    step_theme + theme(
      axis.title.y = element_text(
        size = 8, angle = 0, vjust = 0.5, hjust = 1,
        margin = margin(r = 5)),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank())
}

plot_step_mem <- function(d, ds, is_first) {
  d %>%
    filter(dataset == ds) %>%
    ggplot(aes(memory, group)) +
    geom_col(fill = "#bdbdbd", width = 0.8) +
    geom_text(aes(x = 0, label = paste0(round(memory, 1), " GiB")),
              hjust = -0.05, size = 2) +
    scale_x_continuous(breaks = scales::breaks_pretty(3),
                       expand = expansion(mult = c(0, 0.5))) +
    scale_y_discrete(drop = FALSE) +
    labs(title = if (is_first) "Peak memory", x = NULL, y = NULL) +
    step_theme + theme(
      axis.text.y = element_blank(), axis.ticks.y = element_blank())
}

build_step_fig <- function(wf) {
  ops <- intersect(op_order[[wf]],
                   unique(steps$operation[steps$workflow == wf]))
  imap(ops, \(op, i) {
    d <- steps %>% filter(workflow == wf, operation == op)
    map(datasets, \(ds) list(
      plot_step(d, ds, op, i == 1,
                show_leg = (i == 1 & ds == datasets[1])),
      plot_step_mem(d, ds, i == 1))) %>%
      list_flatten()
  }) %>% list_flatten() %>%
    wrap_plots(ncol = 6, byrow = TRUE,
               widths = rep(c(4, 1), 3)) +
    plot_layout(guides = "collect")
}

walk(c(names(wf_lab), "commands"), \(wf) {
  n <- length(intersect(
    op_order[[wf]], unique(steps$operation[steps$workflow == wf])))
  ggsave(file.path(work_dir, "figures",
                   paste0("fig_", wf, "_steps.png")),
         build_step_fig(wf),
         width = 13.5, height = max(3, n * 0.8))
})

# --- Embedding figures --------------------------------------------------------
# Hue per broad cell type, lightness ramp per cluster within each type

embed_files <- c(
  scanpy = "basic_scanpy_%s_embedding.csv",
  Seurat = "basic_seurat_%s_embedding.csv",
  brisc  = "basic_brisc_%s_-1_embedding.csv",
  rapids = "basic_rapids_%s_gpu_embedding.csv")

embed_methods <- c(
  scanpy = "UMAP", Seurat = "UMAP", brisc = "PaCMAP", rapids = "UMAP")

for (ds in datasets) {
  paths <- file.path(work_dir, "output", sprintf(embed_files, ds))
  names(paths) <- names(embed_files)
  paths <- paths[file.exists(paths)]
  if (length(paths) == 0) next

  embed_data <- imap(paths, \(path, lib) {
    df <- fread(path) %>% as_tibble() %>%
      slice_sample(n = min(nrow(.), 50000)) %>%
      mutate(library = lib)
    if (lib == "brisc") {
      df %>% rename(embed_1 = pacmap_1, embed_2 = pacmap_2)
    } else df
  }) %>% bind_rows() %>%
    select(embed_1, embed_2, cluster_res_1.0, cell_type_broad, library) %>%
    mutate(cluster_id = paste0(library, "_", cluster_res_1.0))

  # One hue per broad type
  broad_types <- sort(unique(embed_data$cell_type_broad))
  hues <- setNames(
    head(seq(0, 360, length.out = length(broad_types) + 1), -1),
    broad_types)

  # Map each (library, cluster) to its dominant broad type,
  # then shade light -> dark within each type
  pal_df <- embed_data %>%
    count(library, cluster_res_1.0, cell_type_broad, sort = TRUE) %>%
    distinct(library, cluster_res_1.0, .keep_all = TRUE) %>%
    arrange(as.numeric(cluster_res_1.0)) %>%
    mutate(rank = row_number(), n_cl = n(),
           .by = c(library, cell_type_broad)) %>%
    mutate(color = hcl(
      h = hues[cell_type_broad], c = 55,
      l = 85 - (85 - 45) * (rank - 1) / pmax(n_cl - 1, 1)))

  pal <- setNames(pal_df$color,
    paste0(pal_df$library, "_", pal_df$cluster_res_1.0))

  # Facet labels: library (method, time)
  embed_times <- results %>%
    filter(test == "basic", operation == "Embedding", dataset == ds) %>%
    filter(library != "brisc" | num_threads == -1) %>%
    transmute(lib = case_when(
      library == "brisc" ~ "brisc", library == "rapids" ~ "rapids",
      library == "scanpy" ~ "scanpy", .default = "Seurat"),
      duration) %>% deframe()

  lib_labels <- sapply(names(embed_files), \(lib) {
    t <- embed_times[lib]
    paste0(lib, " (", embed_methods[lib],
           if (!is.na(t)) paste0(", ", fmt_time(t)), ")")
  })
  embed_data <- embed_data %>%
    mutate(library = factor(lib_labels[library], lib_labels))
  broad_pal <- setNames(
    hcl(h = hues[broad_types], c = 55, l = 60), broad_types)

  p <- ggplot(embed_data, aes(embed_1, embed_2)) +
    geom_point(aes(color = cluster_id),
               size = 0.75, stroke = 0, alpha = 1) +
    scale_color_manual(values = pal, guide = "none") +
    geom_point(aes(fill = cell_type_broad),
               shape = 22, size = 0, stroke = 0) +
    scale_fill_manual(values = broad_pal, name = NULL,
      guide = guide_legend(override.aes = list(size = 4))) +
    facet_wrap(~ library, nrow = 2, scales = "free") +
    labs(title = ds_lab[ds], x = NULL, y = NULL) +
    base_theme +
    theme(axis.text = element_blank(), axis.ticks = element_blank(),
          strip.text = element_text(size = 11),
          strip.background = element_rect(fill = "white"),
          legend.position = "right")

  ggsave(file.path(work_dir, "figures",
                   paste0("fig_embeddings_", ds, ".png")),
         p, width = 10, height = 10, dpi = 300)
}
