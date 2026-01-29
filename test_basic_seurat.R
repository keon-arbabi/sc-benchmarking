suppressPackageStartupMessages({
  library(BPCells)
  library(Seurat)
  library(ggplot2)
})

work_dir <- "sc-benchmarking"
source(file.path(work_dir, "utils_local.R"))

ARGS <- commandArgs(trailingOnly=TRUE)
DATASET_NAME <- ARGS[1]
DATA_PATH <- ARGS[2]
OUTPUT_PATH_TIME <- ARGS[3]

system_info()
timers <- MemoryTimer(silent = FALSE)

scratch_dir <- "single-cell/BPCells-Scratch"
bpcells_dir_test <- file.path(scratch_dir, "bpcells", "basic")
if (!dir.exists(bpcells_dir_test)) {
    dir.create(bpcells_dir_test, recursive = TRUE)
}
if (file.exists(bpcells_dir_test)) {
  unlink(bpcells_dir_test, recursive = TRUE)
}

timers$with_timer("Load data", {
  mat_disk <- open_matrix_anndata_hdf5(
    path = DATA_PATH)
  mat_disk <- convert_matrix_type(mat_disk, type = "uint32_t")
  file_path <- file.path(bpcells_dir_test)
  write_matrix_dir(
    mat = mat_disk,
    dir = file_path
  )
})

timers$with_timer("Load data", {
  mat <- open_matrix_dir(dir = file_path)
  # custom reader required
  obs_metadata <- read_h5ad_obs(DATA_PATH)
  data <- CreateSeuratObject(counts = mat, meta.data = obs_metadata)
})

timers$with_timer("Quality control", {
  data[["percent.mt"]] <- PercentageFeatureSet(data, pattern = "^MT-")
  data <- subset(data, subset = nFeature_RNA > 200 & percent.mt < 5)
})

timers$with_timer("Normalization", {
  data <- NormalizeData(
    data, normalization.method = "LogNormalize", scale.factor = 10000)
})

timers$with_timer("Feature selection", {
  data <- FindVariableFeatures(
    data, selection.method = "vst", nfeatures = 2000)
})

timers$with_timer("PCA", {
  all.genes <- rownames(data)
  data <- ScaleData(data, features = all.genes)
  data <- RunPCA(data, features = VariableFeatures(object = data))
})

timers$with_timer("Nearest neighbors", {
  data <- FindNeighbors(data, dims = 1:10)
})

timers$with_timer("Clustering (3 res.)", {
  for (resolution in c(0.5, 2, 1)) {
    data <- FindClusters(data, resolution = resolution)
  }
})

timers$with_timer("Embedding", {
  data <- RunUMAP(data, dims = 1:10)
})

timers$with_timer("Plot embedding", {
  DimPlot(data, reduction = "umap", group.by = "cell_type")
  ggsave(paste0(work_dir, "/figures/seurat_embedding_", DATASET_NAME, ".png"),
        dpi = 300, units = "in", width = 10, height = 10)
})

timers$with_timer("Find markers", {
  markers <- FindAllMarkers(data, group.by = "cell_type", only.pos = TRUE)
})

timers$print_summary(unit = "s")

timers_df <- timers$to_dataframe(unit = "s", sort = FALSE)
timers_df$library <- "seurat"
timers_df$test <- "basic"
timers_df$dataset <- DATASET_NAME

write.csv(timers_df, OUTPUT_PATH_TIME, row.names = FALSE)

unlink(bpcells_dir_test, recursive = TRUE)
rm(data, markers, timers, timers_df, mat, mat_disk, obs_metadata)
gc()
