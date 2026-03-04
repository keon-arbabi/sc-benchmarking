suppressPackageStartupMessages({
  library(dplyr)
  library(Seurat)
  library(BPCells)
})

options(future.globals.maxSize = Inf)
source("sc-benchmarking/utils_local.R")

args = commandArgs(trailingOnly=TRUE)
DATA_NAME <- args[1]
DATA_PATH <- args[2]
OUTPUT_PATH_TIME <- args[3]
OUTPUT_PATH_DE <- args[4]

bpcells_dir <- file.path(
  Sys.getenv("SCRATCH"), "bpcells", "de", paste0("data_", DATA_NAME))
unlink(bpcells_dir, recursive = TRUE)

system_info()
cat("--- Params ---\n")
cat("seurat de deseq\n")
cat(sprintf("DATA_PATH=%s\n", DATA_PATH))

timers <- MemoryTimer(silent = FALSE)

timers$with_timer("Load data", {
  mat_disk <- open_matrix_anndata_hdf5(path = DATA_PATH)
  mat_disk <- convert_matrix_type(mat_disk, type = "uint32_t")
  write_matrix_dir(mat = mat_disk, dir = bpcells_dir)
  mat <- open_matrix_dir(dir = bpcells_dir)
  # Custom reader required
  obs_metadata <- read_h5ad_obs(DATA_PATH)
  data <- CreateSeuratObject(counts = mat, meta.data = obs_metadata)
})

timers$with_timer("Quality control", {
  data[["percent.mt"]] <- PercentageFeatureSet(data, pattern = "^MT-")
  data <- subset(
    data, subset = nFeature_RNA >= 100 & percent.mt <= 5 & MALAT1 > 0,
    slot = "counts")
})

if (DATA_NAME == "SEAAD") {
  data$cond <- ifelse(data$cond == 1, "AD", "Control")
  group_cols <- c("cond", "sample", "cell_type")
  ident_pairs <- list(test = "AD", ref = "Control")
} else if (DATA_NAME == "PBMC") {
  data <- subset(data, subset = cytokine %in% c("IFN-gamma", "PBS"))
  group_cols <- c("cytokine", "sample", "cell_type")
  ident_pairs <- list(test = "IFN-gamma", ref = "PBS")
}

# Seurat lacks pseudobulk-level filtering

# BPCells native function required
# Seurat::AggregateExpression internally coerces the input
# count matrix through dgCMatrix (via as.sparse / %*%),
# which fails with x[i,j] too dense for [CR]sparseMatrix
# when the input matrix exceeds R's 32-bit index limit
timers$with_timer("Pseudobulk", {
  cell_groups <- do.call(paste, c(data@meta.data[group_cols], sep = "_"))
  mat <- GetAssayData(data, layer = "counts")
  pb_mat <- pseudobulk_matrix(mat, cell_groups, method = "sum")
  pb_meta <- data@meta.data %>%
    mutate(group = cell_groups) %>%
    distinct(group, .keep_all = TRUE)
  rownames(pb_meta) <- pb_meta$group
  pb_meta <- pb_meta[colnames(pb_mat), ]
  data <- CreateSeuratObject(counts = pb_mat, meta.data = pb_meta)
  # Required for Seurat::FindMarkers internal filtering
  data <- NormalizeData(data, verbose = FALSE)
})

timers$with_timer("Differential expression", {
  data$group <- paste(
    data$cell_type, data@meta.data[[group_cols[1]]], sep = "_")
  Idents(data) <- "group"
  de_list <- list()
  for (ct in unique(data$cell_type)) {
    de_list[[ct]] <- FindMarkers(
      data,
      ident.1 = paste(ct, ident_pairs$test, sep = "_"),
      ident.2 = paste(ct, ident_pairs$ref, sep = "_"),
      test.use = "DESeq2")
  }
  de <- do.call(rbind, de_list)
})

de_df <- do.call(rbind, lapply(names(de_list), function(ct) {
  df <- de_list[[ct]]
  data.frame(
    cell_type = ct,
    gene = rownames(df),
    logFC = df$avg_log2FC,
    p_value = df$p_val,
    p_value_adj = df$p_val_adj
  )
}))
write.csv(de_df, OUTPUT_PATH_DE, row.names = FALSE)

timers$print_summary(unit = "s")

timers_df <- timers$to_dataframe(unit = "s", sort = FALSE)
timers_df$library <- "seurat"
timers_df$test <- "de"
timers_df$dataset <- DATA_NAME
write.csv(timers_df, OUTPUT_PATH_TIME, row.names = FALSE)

if (!any(timers_df$aborted)) {
  cat("--- Completed successfully ---\n")
}

cat("\n--- Session Info ---\n")
print(sessionInfo())

unlink(bpcells_dir, recursive = TRUE)
