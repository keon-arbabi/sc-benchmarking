suppressPackageStartupMessages({
  library(BPCells)
  library(Seurat)
  library(ggplot2)
})

options(future.globals.maxSize = Inf)
source("sc-benchmarking/utils_local.R")

ARGS <- commandArgs(trailingOnly=TRUE)
DATASET_NAME <- ARGS[1]
DATA_PATH <- ARGS[2]
OUTPUT_PATH_TIME <- ARGS[3]
OUTPUT_PATH_EMBEDDING <- ARGS[4]

system_info()
cat("--- Params ---\n")
cat("seurat basic\n")
cat(sprintf("DATA_PATH=%s\n", DATA_PATH))

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

cells_keep <- colnames(data)[as.logical(data@meta.data[["_passed_QC"]])]
data <- subset(data, cells = cells_keep)

timers$with_timer("Normalization", {
  data <- NormalizeData(
    data, normalization.method = "LogNormalize", scale.factor = 10000)
})

timers$with_timer("Feature selection", {
  data <- FindVariableFeatures(
    data, selection.method = "vst", nfeatures = 2000)
})

timers$with_timer("PCA", {
  data <- ScaleData(data)
  data <- RunPCA(data, features = VariableFeatures(object = data))
})

timers$with_timer("Nearest neighbors", {
  pca_mat <- Embeddings(data, "pca")
  knn <- knn_hnsw(pca_mat, k = 20)
  snn <- knn_to_snn_graph(knn)
})

timers$with_timer("Clustering (3 res.)", {
  for (resolution in c(0.5, 2, 1)) {
    cl <- cluster_graph_leiden(snn, resolution = resolution)
    data[[paste0("clusters_", format(resolution, nsmall = 1))]] <-
      as.character(cl)
  }
})

timers$with_timer("Embedding", {
  data <- RunUMAP(data, dims = 1:10)
})

embedding_df <- data.frame(
  cell_id = colnames(data),
  embed_1 = Embeddings(data, "umap")[, 1],
  embed_2 = Embeddings(data, "umap")[, 2],
  cell_type = data$cell_type,
  clusters_0.5 = data[["clusters_0.5"]],
  clusters_1.0 = data[["clusters_1.0"]],
  clusters_2.0 = data[["clusters_2.0"]]
)
write.csv(embedding_df, OUTPUT_PATH_EMBEDDING, row.names = FALSE)

timers$with_timer("Plot embedding", {
  p <- DimPlot(data, reduction = "umap", group.by = "clusters_1.0")
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
rm(data, markers, timers, timers_df, mat, mat_disk, obs_metadata, cells_keep)
gc()

cat("--- Completed successfully ---\n")

cat("\n--- Session Info ---\n")
print(sessionInfo())
