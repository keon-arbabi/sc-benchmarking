suppressPackageStartupMessages({
  library(Seurat)
  library(BPCells)
})

source("sc-benchmarking/utils_local.R")

args = commandArgs(trailingOnly=TRUE)
DATASET_NAME <- args[1]
DATA_PATH <- args[2]
OUTPUT_PATH_TIME <- args[3]
OUTPUT_PATH_ACC <- args[4]

scratch_dir <- Sys.getenv("SCRATCH")
bpcells_dir_test <- file.path(scratch_dir, "bpcells", "transfer")
if (!dir.exists(bpcells_dir_test)) {
    dir.create(bpcells_dir_test, recursive = TRUE)
}

system_info()

cat("--- Params ---\n")
cat("seurat transfer\n")
cat(sprintf("DATASET_NAME=%s\n", DATASET_NAME))

timers <- MemoryTimer(silent = FALSE)

query_dir <- file.path(bpcells_dir_test, "query")
if (file.exists(query_dir)) {
  unlink(query_dir, recursive = TRUE)
}

timers$with_timer("Load data", {
  mat_disk <- open_matrix_anndata_hdf5(path = DATA_PATH)
  mat_disk <- convert_matrix_type(mat_disk, type = "uint32_t")
  write_matrix_dir(mat = mat_disk, dir = query_dir)
  mat <- open_matrix_dir(dir = query_dir)
  obs_metadata <- read_h5ad_obs(DATA_PATH)
  data <- CreateSeuratObject(counts = mat, meta.data = obs_metadata)
})

timers$with_timer("Quality control", {
  data[["percent.mt"]] <- PercentageFeatureSet(data, pattern = "^MT-")
  data <- subset(data, subset = nFeature_RNA > 200 & percent.mt < 5)
})

timers$with_timer("Split data", {
  if (DATASET_NAME == 'SEAAD') {
    data_ref <- subset(data, subset = cond == 0)
    data_query <- subset(data, subset = cond == 1)
  } else if (DATASET_NAME == 'PBMC') {
    data_ref <- subset(data, subset = cond == 'PBS')
    data_query <- subset(data, subset = cond == 'cytokine')
  }
})

rm(data); gc()

timers$with_timer("Normalization", {
  data_ref <- NormalizeData(data_ref)
  data_query <- NormalizeData(data_query)
})

timers$with_timer("Feature selection", {
  data_ref <- FindVariableFeatures(data_ref)
})

timers$with_timer("PCA", {
  data_ref <- ScaleData(data_ref)
  data_ref <- RunPCA(data_ref)
})

timers$with_timer("Transfer labels", {
  anchors <- FindTransferAnchors(
    reference = data_ref,
    query = data_query,
    dims = 1:30,
    reference.reduction = "pca")
  predictions <- TransferData(
    anchorset = anchors,
    refdata = data_ref$cell_type,
    dims = 1:30)
  data_query <- AddMetaData(
    object = data_query,
    metadata = predictions)
})

accuracy_df <- transfer_accuracy(
  data_query@meta.data, "cell_type", "predicted.id")
write.csv(accuracy_df, OUTPUT_PATH_ACC, row.names = FALSE)

timers$print_summary(unit = "s")

timers_df <- timers$to_dataframe(unit = "s", sort = FALSE)
timers_df$library <- 'seurat'
timers_df$test <- 'transfer'
timers_df$dataset <- DATASET_NAME
timers_df$num_threads <- 'single-threaded'
write.csv(timers_df, OUTPUT_PATH_TIME, row.names = FALSE)

if (!any(timers_df$aborted)) {
  cat("--- Completed successfully ---\n")
}

unlink(query_dir, recursive = TRUE)

rm(data_query, data_ref, anchors, predictions, timers, 
  timers_df, accuracy_df, mat, mat_disk, obs_metadata)
gc()
