suppressPackageStartupMessages({
  library(tidyverse)
  library(Seurat)
  library(BPCells)
})

source("sc-benchmarking/utils_local.R")

args = commandArgs(trailingOnly=TRUE)
# DATASET_NAME <- args[1]
# DATA_PATH <- args[2]
# REF_PATH <- args[3]
# OUTPUT_PATH <- args[4]

DATASET_NAME <- 'SEAAD'
DATA_PATH <- 'single-cell/SEAAD/SEAAD_raw.h5ad'
REF_PATH <- 'single-cell/SEAAD/SEAAD_ref.h5ad'
OUTPUT_PATH <- 'sc-benchmarking/output/test_transfer_seurat_SEAAD.csv'

scratch_dir <- Sys.getenv("SCRATCH")
bpcells_dir_test <- file.path(scratch_dir, "bpcells", "transfer")
if (!dir.exists(bpcells_dir_test)) {
    dir.create(bpcells_dir_test, recursive = TRUE)
}

system_info()

cat("--- Params ---\n")
cat("seurat transfer\n")
cat(sprintf("R.version=%s\n", R.version.string))
cat(sprintf("DATASET_NAME=%s\n", DATASET_NAME))

timers <- MemoryTimer(silent = FALSE)

# Not timed - cleanup existing directories
query_dir <- file.path(bpcells_dir_test, "query")
ref_dir <- file.path(bpcells_dir_test, "ref")
if (file.exists(query_dir)) {
  unlink(query_dir, recursive = TRUE)
}
if (file.exists(ref_dir)) {
  unlink(ref_dir, recursive = TRUE)
}

# Load data (query) 
timers$with_timer("Load data (query)", {
  mat_disk <- open_matrix_anndata_hdf5(path = DATA_PATH)
  mat_disk <- convert_matrix_type(mat_disk, type = "uint32_t")
  write_matrix_dir(mat = mat_disk, dir = query_dir)
  mat <- open_matrix_dir(dir = query_dir)
  obs_metadata <- read_h5ad_obs(DATA_PATH)
  data_query <- CreateSeuratObject(counts = mat, meta.data = obs_metadata)
})

# Load data (ref) 
timers$with_timer("Load data (ref)", {
  mat_disk <- open_matrix_anndata_hdf5(path = REF_PATH)
  mat_disk <- convert_matrix_type(mat_disk, type = "uint32_t")
  write_matrix_dir(mat = mat_disk, dir = ref_dir)
  mat <- open_matrix_dir(dir = ref_dir)
  obs_metadata <- read_h5ad_obs(REF_PATH)
  data_ref <- CreateSeuratObject(counts = mat, meta.data = obs_metadata)
})

# Quality control 
timers$with_timer("Quality control", {
  data_query[["percent.mt"]] <- PercentageFeatureSet(
    data_query, pattern = "^MT-")
  data_query <- subset(
    data_query, subset = nFeature_RNA > 200 & percent.mt < 5)
})

# Normalization ####
timers$with_timer("Normalization", {
  data_ref <- NormalizeData(data_ref)
  data_query <- NormalizeData(data_query)
})

# Feature selection ####
timers$with_timer("Feature selection", {
  data_ref <- FindVariableFeatures(data_ref)
  data_query <- FindVariableFeatures(data_query)
})

# PCA ####
timers$with_timer("PCA", {
  data_ref <- ScaleData(data_ref)
  data_ref <- RunPCA(data_ref)
  data_query <- ScaleData(data_query)
  data_query <- RunPCA(data_query)
})

# Transfer labels ####
timers$with_timer("Transfer labels", {
  anchors <- FindTransferAnchors(
    reference = data_ref, query = data_query, dims = 1:30,
    reference.reduction = "pca")
  predictions <- TransferData(
    anchorset = anchors, refdata = data_ref$cell_type)
  data_query <- AddMetaData(
    object = data_query, metadata = predictions)
})

cat("--- Transfer Accuracy ---\n")
accuracy_df <- transfer_accuracy(
  data_query@meta.data, "cell_type", "predicted.id")
print(accuracy_df, n = Inf)

timers$print_summary(unit = "s")

timers_df <- timers$to_dataframe(unit = "s", sort = FALSE)
timers_df$library <- 'seurat'
timers_df$test <- 'transfer'
timers_df$dataset <- DATASET_NAME
timers_df$num_threads <- 'single-threaded'
write.csv(timers_df, OUTPUT_PATH, row.names = FALSE)

if (!any(timers_df$aborted)) {
  cat("--- Completed successfully ---\n")
}

unlink(query_dir, recursive = TRUE)
unlink(ref_dir, recursive = TRUE)

rm(data_query, data_ref, anchors, predictions, timers, 
  timers_df, accuracy_df, mat, mat_disk, obs_metadata)
gc()