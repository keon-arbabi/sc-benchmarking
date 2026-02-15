suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(patchwork)
})

work_dir <- "sc-benchmarking"

results <- list.files(
    file.path(work_dir, "output"),
    "_timer\\.csv$", full.names = TRUE
  ) %>%
  map(fread) %>%
  rbindlist(fill = TRUE) %>%
  as_tibble()

grp <- list(
  order = c("Seurat", "scanpy",
    "brisc (ST)", "brisc (MT)"),
  colors = c("brisc (MT)" = "#d62728",
    "brisc (ST)" = "#e8726a",
    "scanpy" = "#8fbc8f", "Seurat" = "#a8c4dc")
)

ds_lab <- c(SEAAD = "1.2M cells (SEAAD)",
  PBMC = "9.7M cells (Parse PBMC)")

wf_lab <- c(basic = "Basic workflow",
  transfer = "Label transfer",
  de = "Differential expression")

fmt_time <- function(s) {
  case_when(
    s >= 3600 ~ paste0(round(s / 3600, 1), "h"),
    s >= 60 ~ paste0(round(s / 60, 1), "m"),
    .default = paste0(round(s, 1), "s"))
}

prepared <- results %>%
  filter(test != "de_deseq") %>%
  mutate(
    workflow = str_remove(test, "_wilcox$") %>%
      factor(levels = names(wf_lab)),
    dataset = factor(dataset, c("SEAAD", "PBMC")),
    group = case_when(
      library == "brisc" &
        num_threads == "multi-threaded" ~ "brisc (MT)",
      library == "brisc" ~ "brisc (ST)",
      library == "scanpy" ~ "scanpy",
      .default = "Seurat"
    ) %>% factor(levels = grp$order))

totals <- prepared %>%
  summarize(duration_s = sum(duration),
    .by = c(group, workflow, dataset))

ref_times <- totals %>%
  filter(group == "brisc (MT)") %>%
  select(workflow, dataset, ref = duration_s)

totals <- totals %>%
  left_join(ref_times,
    by = c("workflow", "dataset")) %>%
  mutate(
    duration = if_else(dataset == "PBMC",
      duration_s / 3600, duration_s),
    time_lbl = fmt_time(duration_s),
    left_lbl = if_else(
      group %in% c("scanpy", "Seurat"),
      paste0(time_lbl, " (",
        round(duration_s / ref, 1),
        "x)"),
      time_lbl))

peak_mem <- prepared %>%
  summarize(memory = max(memory),
    .by = c(group, workflow, dataset))

base_theme <- theme_bw() +
  theme(
    text = element_text(color = "black"),
    axis.text = element_text(color = "black"),
    plot.title = element_text(
      size = 11, hjust = 0.5),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = margin(6, 6, 6, 6))

plot_panel <- function(wf, ds) {
  unit <- if (ds == "SEAAD") "s" else "h"
  totals %>%
    filter(workflow == wf, dataset == ds) %>%
    ggplot(aes(duration, group, fill = group)) +
    geom_col(width = 0.7) +
    geom_text(aes(label = left_lbl),
      hjust = -0.05, size = 2.5, na.rm = TRUE) +
    scale_x_continuous(
      expand = expansion(mult = c(0, 0.35))) +
    scale_fill_manual(
      values = grp$colors, guide = "none") +
    labs(
      title = if (wf == "basic")
        ds_lab[ds] else NULL,
      x = if (wf == "de")
        paste0("Duration (", unit, ")")
        else NULL,
      y = if (ds == "SEAAD")
        wf_lab[wf] else NULL) +
    base_theme +
    theme(
      axis.title.y = element_text(
        size = 11, margin = margin(r = 10)),
      axis.text.y = if (ds == "PBMC")
        element_blank() else element_text(),
      axis.ticks.y = if (ds == "PBMC")
        element_blank() else element_line())
}

plot_mem <- function(wf, ds) {
  peak_mem %>%
    filter(workflow == wf, dataset == ds) %>%
    ggplot(aes(memory, group)) +
    geom_col(fill = "#bdbdbd", width = 0.7) +
    geom_text(aes(x = 0, label = paste0(
        round(memory, 1), " GiB")),
      hjust = -0.05, size = 2.5) +
    scale_x_continuous(
      breaks = scales::breaks_pretty(3),
      expand = expansion(mult = c(0, 0.5))) +
    labs(
      title = if (wf == "basic")
        "Peak memory" else NULL,
      x = if (wf == "de") "GiB" else NULL,
      y = NULL) +
    base_theme +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank())
}

p_totals <- map(names(wf_lab), \(wf) list(
    plot_panel(wf, "SEAAD"),
    plot_mem(wf, "SEAAD"),
    plot_panel(wf, "PBMC"),
    plot_mem(wf, "PBMC"))) %>%
  list_flatten() %>%
  wrap_plots(ncol = 4, byrow = TRUE,
    widths = c(4, 1, 4, 1))

ggsave(
  file.path(work_dir, "figures", "total_runtime.png"),
  p_totals, width = 10.5, height = 6)

op_order <- list(
  basic = c("Load data", "Quality control",
    "Doublet detection", "Normalization",
    "Feature selection", "PCA",
    "Nearest neighbors",
    "Clustering (3 res.)", "Embedding",
    "Find markers"),
  transfer = c("Load data", "Quality control",
    "Doublet detection", "Split data",
    "Normalization", "Feature selection",
    "PCA", "Transfer labels"),
  de = c("Load data", "Quality control",
    "Doublet detection", "Data transformation",
    "Normalization", "Differential expression"))

basic_dd <- prepared %>%
  filter(workflow == "basic",
    operation == "Doublet detection",
    group %in% c("scanpy", "Seurat"))

dd_rows <- c("transfer", "de") %>%
  map(\(wf) mutate(basic_dd,
    workflow = factor(wf,
      levels = names(wf_lab)))) %>%
  bind_rows()

steps <- bind_rows(prepared, dd_rows) %>%
  mutate(
    duration_s = duration,
    duration = if_else(dataset == "PBMC",
      duration_s / 3600, duration_s))

step_ref <- steps %>%
  filter(group == "brisc (MT)") %>%
  select(operation, workflow, dataset,
    ref = duration_s)

steps <- steps %>%
  left_join(step_ref, by = c("operation",
    "workflow", "dataset")) %>%
  mutate(
    time_lbl = fmt_time(duration_s),
    left_lbl = if_else(
      group %in% c("scanpy", "Seurat") &
        !is.na(ref) & ref > 0,
      paste0(time_lbl, " (",
        round(duration_s / ref, 1), "x)"),
      time_lbl))

step_theme <- base_theme +
  theme(
    plot.margin = margin(0, 6, 0, 6),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank())

plot_step_time <- function(d, ds, op,
    is_first, show_leg = FALSE) {
  d %>%
    filter(dataset == ds) %>%
    ggplot(aes(duration, group, fill = group)) +
    geom_col(width = 0.8) +
    geom_text(aes(label = left_lbl),
      hjust = -0.05, size = 2,
      na.rm = TRUE) +
    scale_x_continuous(
      expand = expansion(mult = c(0, 0.35))) +
    scale_fill_manual(values = grp$colors,
      limits = grp$order, name = NULL,
      guide = if (show_leg)
        guide_legend(reverse = TRUE)
        else "none") +
    labs(
      title = if (is_first)
        ds_lab[ds] else NULL,
      x = NULL,
      y = if (ds == "SEAAD") op else NULL) +
    step_theme +
    theme(
      axis.title.y = element_text(size = 8,
        angle = 0, vjust = 0.5, hjust = 1,
        margin = margin(r = 5)),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank())
}

plot_step_mem <- function(d, ds, is_first) {
  d %>%
    filter(dataset == ds) %>%
    ggplot(aes(memory, group)) +
    geom_col(fill = "#bdbdbd", width = 0.8) +
    geom_text(aes(x = 0, label = paste0(
        round(memory, 1), " GiB")),
      hjust = -0.05, size = 2) +
    scale_x_continuous(
      breaks = scales::breaks_pretty(3),
      expand = expansion(mult = c(0, 0.5))) +
    labs(
      title = if (is_first)
        "Peak memory" else NULL,
      x = NULL, y = NULL) +
    step_theme +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank())
}

build_step_fig <- function(wf) {
  wf_ops <- intersect(op_order[[wf]],
    unique(steps$operation[steps$workflow == wf]))
  n <- length(wf_ops)
  imap(wf_ops, \(op, i) {
    d <- steps %>%
      filter(workflow == wf, operation == op)
    f <- i == 1
    list(
      plot_step_time(d, "SEAAD", op, f,
        show_leg = f),
      plot_step_mem(d, "SEAAD", f),
      plot_step_time(d, "PBMC", op, f),
      plot_step_mem(d, "PBMC", f))
  }) %>%
    list_flatten() %>%
    wrap_plots(ncol = 4, byrow = TRUE,
      widths = c(4, 1, 4, 1)) +
    plot_layout(guides = "collect")
}

walk(names(op_order), function(wf) {
  n <- length(intersect(op_order[[wf]],
    unique(steps$operation[
      steps$workflow == wf])))
  ggsave(
    file.path(work_dir, "figures",
      paste0(wf, "_steps1.png")),
    build_step_fig(wf),
    width = 9,
    height = max(3, n * 0.8))
})

