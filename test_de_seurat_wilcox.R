suppressPackageStartupMessages({
  library(dplyr)
  library(Seurat)
  library(BPCells)
})

options(future.globals.maxSize = Inf)
source("sc-benchmarking/utils_local.R")

args = commandArgs(trailingOnly=TRUE)
DATASET_NAME <- args[1]
DATA_PATH <- args[2]
OUTPUT_PATH_TIME <- args[3]
INPUT_PATH_DOUBLET <- args[4]

bpcells_dir <- file.path(
  Sys.getenv("SCRATCH"), "bpcells", "de_wilcox", paste0("data_", DATASET_NAME))
unlink(bpcells_dir, recursive = TRUE)

system_info()
cat("--- Params ---\n")
cat("seurat de wilcox\n")
cat(sprintf("DATASET_NAME=%s\n", DATASET_NAME))

timers <- MemoryTimer(silent = FALSE)

timers$with_timer("Load data", {
  mat_disk <- open_matrix_anndata_hdf5(path = DATA_PATH)
  mat_disk <- convert_matrix_type(mat_disk, type = "uint32_t")
  write_matrix_dir(mat = mat_disk, dir = bpcells_dir)
  mat <- open_matrix_dir(dir = bpcells_dir)
  obs_metadata <- read_h5ad_obs(DATA_PATH)
  data <- CreateSeuratObject(counts = mat, meta.data = obs_metadata)
})

timers$with_timer("Quality control", {
  data[["percent.mt"]] <- PercentageFeatureSet(data, pattern = "^MT-")
  data <- subset(data, subset = nFeature_RNA > 200 & percent.mt < 5)
})

doublet_df <- read.csv(INPUT_PATH_DOUBLET)
doublets <- doublet_df$cell_id[doublet_df$is_doublet]
data <- subset(data, cells = setdiff(colnames(data), doublets))

timers$with_timer("Normalization", {
  data <- NormalizeData(
    data, normalization.method = "LogNormalize", scale.factor = 10000)
})

if (DATASET_NAME == "SEAAD") {
  data$cond <- ifelse(data$cond == 1, "AD", "Control")

  timers$with_timer("Differential expression", {
    data$group <- paste(data$cell_type, data$cond, sep = "_")
    Idents(data) <- "group"
    de_list <- list()
    for (ct in unique(data$cell_type)) {
      de_list[[ct]] <- FindMarkers(
        data,
        ident.1 = paste(ct, "AD", sep = "_"),
        ident.2 = paste(ct, "Control", sep = "_"))
    }
    de <- do.call(rbind, de_list)
  })

} else if (DATASET_NAME == "PBMC") {
  data <- subset(data, subset = cytokine %in% c("IFN-gamma", "PBS"))

  timers$with_timer("Differential expression", {
    data$group <- paste(data$cell_type, data$cytokine, sep = "_")
    Idents(data) <- "group"
    de_list <- list()
    for (ct in unique(data$cell_type)) {
      de_list[[ct]] <- FindMarkers(
        data,
        ident.1 = paste(ct, "IFN-gamma", sep = "_"),
        ident.2 = paste(ct, "PBS", sep = "_"))
    }
    de <- do.call(rbind, de_list)
  })
}

timers$print_summary(unit = "s")

timers_df <- timers$to_dataframe(unit = "s", sort = FALSE)
timers_df$library <- "seurat"
timers_df$test <- "de_wilcox"
timers_df$dataset <- DATASET_NAME
write.csv(timers_df, OUTPUT_PATH_TIME, row.names = FALSE)

if (!any(timers_df$aborted)) {
  cat("--- Completed successfully ---\n")
}

cat("\n--- Session Info ---\n")
print(sessionInfo())

unlink(bpcells_dir, recursive = TRUE)
rm(data, de, de_list, timers, timers_df, mat, mat_disk, obs_metadata)
gc()
