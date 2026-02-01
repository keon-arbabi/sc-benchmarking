suppressPackageStartupMessages({
  library(BPCells)
  library(Seurat)
  library(ggplot2)
})

source("sc-benchmarking/utils_local.R")

ARGS <- commandArgs(trailingOnly=TRUE)
DATASET_NAME <- ARGS[1]
DATA_PATH <- ARGS[2]
OUTPUT_PATH_TIME <- ARGS[3]

system_info()
timers <- MemoryTimer(silent = FALSE)

bpcells_dir <- file.path(
  Sys.getenv("SCRATCH"), "bpcells", "basic", paste0("data_", DATASET_NAME))
unlink(bpcells_dir, recursive = TRUE)

timers$with_timer("Load data", {
  mat_disk <- open_matrix_anndata_hdf5(path = DATA_PATH)
  mat_disk <- convert_matrix_type(mat_disk, type = "uint32_t")
  write_matrix_dir(mat = mat_disk, dir = bpcells_dir)
  mat <- open_matrix_dir(dir = bpcells_dir)
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
  p <- DimPlot(data, reduction = "umap", group.by = "cell_type")
  ggsave(
    paste0("sc-benchmarking/figures/seurat_embedding_", DATASET_NAME, ".png"),
    plot = p, dpi = 300, units = "in", width = 10, height = 10)
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

unlink(bpcells_dir, recursive = TRUE)
rm(data, markers, timers, timers_df, mat, mat_disk, obs_metadata)
gc()

cat("--- Completed successfully ---\n")
