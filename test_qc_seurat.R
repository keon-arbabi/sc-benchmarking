suppressPackageStartupMessages({
  library(BPCells)
  library(Seurat)
  library(SingleCellExperiment)
  library(scDblFinder)
  library(BiocParallel)
})

options(future.globals.maxSize = Inf)
source("sc-benchmarking/utils_local.R")

ARGS <- commandArgs(trailingOnly=TRUE)
DATASET_NAME <- ARGS[1]
DATA_PATH <- ARGS[2]
OUTPUT_PATH_TIME <- ARGS[3]
OUTPUT_PATH_DOUBLET <- ARGS[4]

system_info()
cat("--- Params ---\n")
cat("seurat qc\n")
cat(sprintf("DATA_PATH=%s\n", DATA_PATH))

timers <- MemoryTimer(silent = FALSE)

bpcells_dir <- file.path(
  Sys.getenv("SCRATCH"), "bpcells", "qc", paste0("data_", DATASET_NAME))
unlink(bpcells_dir, recursive = TRUE)

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

timers$with_timer("Doublet detection", {
  counts_full <- LayerData(data, assay = "RNA", layer = "counts")
  bp <- MulticoreParam(workers = 120, RNGseed = 123)
  doublet_results <- do.call(rbind, lapply(unique(data$sample), function(s) {
    cells <- colnames(data)[data$sample == s]
    counts <- as(counts_full[, cells], "dgCMatrix")
    sce <- scDblFinder(
      SingleCellExperiment(assays = list(counts = counts)),
      BPPARAM = bp,
      verbose = FALSE)
    data.frame(
      cell_id = colnames(sce),
      doublet_score = sce$scDblFinder.score,
      is_doublet = sce$scDblFinder.class == "doublet")
  }))
  bpstop(bp)
})

write.csv(doublet_results, OUTPUT_PATH_DOUBLET, row.names = FALSE)

timers$print_summary(unit = "s")

timers_df <- timers$to_dataframe(unit = "s", sort = FALSE)
timers_df$library <- "seurat"
timers_df$test <- "qc"
timers_df$dataset <- DATASET_NAME

write.csv(timers_df, OUTPUT_PATH_TIME, row.names = FALSE)

unlink(bpcells_dir, recursive = TRUE)
rm(data, doublet_results, timers, timers_df, mat, mat_disk, obs_metadata)
gc()

cat("--- Completed successfully ---\n")

cat("\n--- Session Info ---\n")
print(sessionInfo())
