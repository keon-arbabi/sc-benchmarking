suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(patchwork)
})

work_dir <- "sc-benchmarking"

results <- list.files(
  file.path(work_dir, "output"), "_timer\\.csv$", full.names = TRUE) %>%
  map(fread) %>% rbindlist(fill = TRUE) %>% as_tibble()

grp <- list(
  order = c("Seurat", "scanpy", "brisc (ST)", "brisc (MT)"),
  colors = c("brisc (MT)" = "#d62728", "brisc (ST)" = "#e8726a",
             "scanpy" = "#8fbc8f", "Seurat" = "#a8c4dc")
)

ds_lab <- c(SEAAD = "1.2M cells (SEAAD)", PBMC = "9.7M cells (Parse PBMC)")
wf_lab <- c(basic = "Basic workflow", transfer = "Label transfer",
            de = "Differential expression")

fmt_time <- function(s) {
  case_when(s >= 3600 ~ paste0(round(s / 3600, 1), "h"),
            s >= 60 ~ paste0(round(s / 60, 1), "m"),
            .default = paste0(round(s, 1), "s"))
}

make_label <- function(time_s, ref_s) {
  lbl <- fmt_time(time_s)
  if_else(!is.na(ref_s) & ref_s > 0,
          paste0(lbl, " (", round(time_s / ref_s, 1), "x)"), lbl)
}

prepared <- results %>%
  mutate(
    workflow = factor(test, levels = names(wf_lab)),
    dataset = factor(dataset, c("SEAAD", "PBMC")),
    group = case_when(
      library == "brisc" & num_threads == -1 ~ "brisc (MT)",
      library == "brisc" ~ "brisc (ST)",
      library == "scanpy" ~ "scanpy",
      .default = "Seurat"
    ) %>% factor(levels = grp$order)
  )

# --- Total runtime figure ----------------------------------------------------

totals <- prepared %>%
  summarize(total_s = sum(duration), .by = c(group, workflow, dataset)) %>%
  left_join(
    prepared %>%
      filter(group == "brisc (MT)") %>%
      summarize(ref = sum(duration), .by = c(workflow, dataset)),
    by = c("workflow", "dataset")) %>%
  mutate(
    ref = if_else(group == "brisc (MT)", NA_real_, ref),
    dur_plot = if_else(dataset == "PBMC", total_s / 3600, total_s),
    label = make_label(total_s, ref))

peak_mem <- prepared %>%
  summarize(memory = max(memory), .by = c(group, workflow, dataset))

base_theme <- theme_bw() + theme(
  text = element_text(color = "black"),
  axis.text = element_text(color = "black"),
  plot.title = element_text(size = 11, hjust = 0.5),
  panel.grid = element_blank(),
  plot.margin = margin(6, 6, 6, 6)
)

plot_bar <- function(wf, ds) {
  unit <- if (ds == "SEAAD") "s" else "h"
  totals %>%
    filter(workflow == wf, dataset == ds) %>%
    ggplot(aes(dur_plot, group, fill = group)) +
    geom_col(width = 0.7) +
    geom_text(aes(label = label),
              hjust = -0.05, size = 2.5, na.rm = TRUE) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.35))) +
    scale_fill_manual(values = grp$colors, guide = "none") +
    labs(title = if (wf == "basic") ds_lab[ds],
         x = if (wf == "de") paste0("Duration (", unit, ")"),
         y = if (ds == "SEAAD") wf_lab[wf]) +
    base_theme + theme(
      axis.title.y = element_text(size = 11, margin = margin(r = 10)),
      axis.text.y = if (ds == "PBMC") element_blank() else element_text(),
      axis.ticks.y = if (ds == "PBMC") element_blank() else element_line()
    )
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
    labs(title = if (wf == "basic") "Peak memory",
         x = if (wf == "de") "GiB", y = NULL) +
    base_theme + theme(
      axis.text.y = element_blank(), axis.ticks.y = element_blank()
    )
}

p_totals <- map(names(wf_lab), \(wf) list(
  plot_bar(wf, "SEAAD"), plot_mem(wf, "SEAAD"),
  plot_bar(wf, "PBMC"), plot_mem(wf, "PBMC"))) %>%
  list_flatten() %>%
  wrap_plots(ncol = 4, byrow = TRUE, widths = c(4, 1, 4, 1))

ggsave(file.path(work_dir, "figures", "fig_total_runtime.png"),
       p_totals, width = 10.5, height = 6)

# --- Per-step figures ---------------------------------------------------------

op_order <- list(
  basic = c(
    "Load data", "Quality control", "Normalization",
    "Feature selection", "PCA", "Nearest neighbors",
    "Clustering", "Embedding", "Find markers"),
  transfer = c(
    "Load data", "Quality control", "Split data",
    "Normalization", "Feature selection", "PCA",
    "Transfer labels"),
  de = c(
    "Load data", "Quality control", "Pseudobulk",
    "Filter", "Differential expression")
)

step_ref <- prepared %>%
  filter(group == "brisc (MT)") %>%
  select(operation, workflow, dataset, ref = duration)

steps <- prepared %>%
  left_join(step_ref, by = c("operation", "workflow", "dataset")) %>%
  mutate(
    ref = if_else(group == "brisc (MT)", NA_real_, ref),
    dur_plot = if_else(dataset == "PBMC", duration / 3600, duration),
    label = make_label(duration, ref)
  )

step_theme <- base_theme + theme(
  plot.margin = margin(0, 6, 0, 6),
  axis.text.x = element_blank(), axis.ticks.x = element_blank()
)

plot_step <- function(d, ds, op, is_first, show_leg = FALSE) {
  d %>%
    filter(dataset == ds) %>%
    ggplot(aes(dur_plot, group, fill = group)) +
    geom_col(width = 0.8) +
    geom_text(aes(label = label),
              hjust = -0.05, size = 2, na.rm = TRUE) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.35))) +
    scale_fill_manual(
      values = grp$colors, limits = grp$order, name = NULL,
      guide = if (show_leg) guide_legend(reverse = TRUE) else "none"
    ) +
    labs(title = if (is_first) ds_lab[ds], x = NULL,
         y = if (ds == "SEAAD") op) +
    step_theme + theme(
      axis.title.y = element_text(
        size = 8, angle = 0, vjust = 0.5, hjust = 1,
        margin = margin(r = 5)),
      axis.text.y = element_blank(), axis.ticks.y = element_blank()
    )
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
    labs(title = if (is_first) "Peak memory", x = NULL, y = NULL) +
    step_theme + theme(
      axis.text.y = element_blank(), axis.ticks.y = element_blank()
    )
}

build_step_fig <- function(wf) {
  ops <- intersect(op_order[[wf]],
                   unique(steps$operation[steps$workflow == wf]))
  imap(ops, \(op, i) {
    d <- steps %>% filter(workflow == wf, operation == op)
    list(plot_step(d, "SEAAD", op, i == 1, show_leg = i == 1),
         plot_step_mem(d, "SEAAD", i == 1),
         plot_step(d, "PBMC", op, i == 1),
         plot_step_mem(d, "PBMC", i == 1))
  }) %>% list_flatten() %>%
    wrap_plots(ncol = 4, byrow = TRUE, widths = c(4, 1, 4, 1)) +
    plot_layout(guides = "collect")
}

walk(names(op_order), \(wf) {
  n <- length(intersect(
    op_order[[wf]], unique(steps$operation[steps$workflow == wf])))
  ggsave(file.path(work_dir, "figures", paste0("fig_", wf, "_steps.png")),
         build_step_fig(wf), width = 9, height = max(3, n * 0.8))
})
