suppressPackageStartupMessages({
  library(BPCells)
  library(Seurat)
  library(ggplot2)
})

options(future.globals.maxSize = Inf)
source("sc-benchmarking/utils_local.R")

ARGS <- commandArgs(trailingOnly=TRUE)
DATA_NAME <- ARGS[1]
DATA_PATH <- ARGS[2]
OUTPUT_PATH_TIME <- ARGS[3]
OUTPUT_PATH_EMBEDDING <- ARGS[4]

system_info()
cat("--- Params ---\n")
cat("seurat basic\n")
cat(sprintf("DATA_PATH=%s\n", DATA_PATH))

timers <- MemoryTimer(silent = FALSE)

bpcells_dir <- file.path(
  Sys.getenv("SCRATCH"), "bpcells", "basic", paste0("data_", DATA_NAME))
unlink(bpcells_dir, recursive = TRUE)

timers$with_timer("Load data", {
  mat_disk <- open_matrix_anndata_hdf5(path = DATA_PATH)
  mat_disk <- convert_matrix_type(mat_disk, type = "uint32_t")
  write_matrix_dir(mat = mat_disk, dir = bpcells_dir)
  mat <- open_matrix_dir(dir = bpcells_dir)
  # Custom h5ad reader function required
  obs_metadata <- read_h5ad_obs(DATA_PATH)
  data <- CreateSeuratObject(counts = mat, meta.data = obs_metadata)
})

timers$with_timer("Quality control", {
  data[["percent.mt"]] <- PercentageFeatureSet(data, pattern = "^MT-")
  data <- subset(
    data, subset = nFeature_RNA >= 100 & percent.mt <= 5 & MALAT1 > 0,
    slot = "counts")
})

timers$with_timer("Normalization", {
  data <- NormalizeData(data)
})

timers$with_timer("Feature selection", {
  data <- FindVariableFeatures(data)
})

timers$with_timer("PCA", {
  data <- ScaleData(data)
  data <- RunPCA(data)
})

# BPCells native functions required
# Seurat::ComputeSNN fails with std::bad_alloc for large data
# as the resulting sparse matrix in memory exceeds R's
# dgCMatrix 32-bit integer index limit
timers$with_timer("Nearest neighbors", {
  pca_mat <- Embeddings(data, "pca")
  knn <- knn_hnsw(pca_mat)
  snn <- knn_to_snn_graph(knn)
  rownames(snn) <- rownames(pca_mat)
  colnames(snn) <- rownames(pca_mat)
})

# BPCells function calls Seurat::FindClusters
timers$with_timer("Clustering", {
  for (resolution in c(0.5, 2, 1)) {
    cl <- cluster_graph_seurat(snn, resolution = resolution)
    data[[paste0("clusters_", format(resolution, nsmall = 1))]] <-
      as.character(cl)
  }
})

timers$with_timer("Embedding", {
  data <- RunUMAP(data, dims = 1:10)
})

timers$with_timer("Plot embedding", {
  p <- DimPlot(
    data, reduction = "umap", group.by = "cell_type")
  ggsave(
    paste0("sc-benchmarking/figures/seurat_embedding_", DATA_NAME, ".png"),
    plot = p, dpi = 300, units = "in", width = 10, height = 10)
})

timers$with_timer("Find markers", {
  markers <- FindAllMarkers(data, group.by = "cell_type", only.pos = TRUE)
})

embedding_df <- data.frame(
  cell_id = colnames(data),
  embed_1 = Embeddings(data, "umap")[, 1],
  embed_2 = Embeddings(data, "umap")[, 2],
  cell_type = data$cell_type
)
write.csv(embedding_df, OUTPUT_PATH_EMBEDDING, row.names = FALSE)

timers$print_summary(unit = "s")

timers_df <- timers$to_dataframe(unit = "s", sort = FALSE)
timers_df$library <- "seurat"
timers_df$test <- "basic"
timers_df$dataset <- DATA_NAME

write.csv(timers_df, OUTPUT_PATH_TIME, row.names = FALSE)

cat("--- Completed successfully ---\n")

cat("\n--- Session Info ---\n")
print(sessionInfo())
