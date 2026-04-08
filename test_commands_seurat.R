suppressPackageStartupMessages({
  library(BPCells)
  library(Seurat)
})

source(file.path(path.expand("~"), "sc-benchmarking", "utils_local.R"))

ARGS <- commandArgs(trailingOnly=TRUE)
DATA_NAME <- ARGS[1]
DATA_PATH <- ARGS[2]
OUTPUT_PATH_TIME <- ARGS[3]

system_info()
cat("--- Params ---\n")
cat("seurat manipulation\n")
cat(sprintf("DATA_PATH=%s\n", DATA_PATH))

timers <- MemoryTimer(
  silent = FALSE, csv_path = OUTPUT_PATH_TIME, summary_unit = "ms",
  csv_columns = list(library = "seurat", test = "manipulation",
                     dataset = DATA_NAME))

bpcells_dir <- file.path(
  "/tmp", "bpcells", "manipulation", paste0("data_", DATA_NAME))
unlink(bpcells_dir, recursive = TRUE)

temp_file <- file.path("/tmp", basename(DATA_PATH))
file.copy(DATA_PATH, temp_file, overwrite = TRUE)

# Setup
mat_disk <- open_matrix_anndata_hdf5(path = temp_file)
mat_disk <- convert_matrix_type(mat_disk, type = "uint32_t")
write_matrix_dir(mat = mat_disk, dir = bpcells_dir)
mat <- open_matrix_dir(dir = bpcells_dir)
obs_metadata <- read_h5ad_obs(temp_file)
data <- CreateSeuratObject(counts = mat, meta.data = obs_metadata)
rm(mat_disk, mat, obs_metadata); gc()

data <- FindVariableFeatures(data)

counts <- GetAssayData(data, layer = "counts")
cell_name <- colnames(data)[1]
gene_name <- rownames(data)[1]
cell_type_select <- as.character(data$cell_type[1])

timers$with_timer("Get expression by cell", {
  as.matrix(counts[, cell_name])
})

timers$with_timer("Get expression by gene", {
  as.matrix(counts[gene_name, ])
})

timers$with_timer("Subset to one cell type", {
  data[, data$cell_type == cell_type_select]
})

timers$with_timer("Subset to highly variable genes", {
  subset(data, features = VariableFeatures(data))
})

timers$with_timer("Subsample to 10,000 cells", {
  subset(data, cells = sample(colnames(data), 10000))
})

timers$with_timer("Select categorical columns", {
  data@meta.data[, !sapply(data@meta.data, is.numeric), drop = FALSE]
})

timers$with_timer("Split by cell type", {
  data_split <- SplitObject(data, split.by = "cell_type_broad")
})

rm(data); gc()

timers$with_timer("Concatenate cell types", {
  data <- merge(data_split[[1]], y = data_split[-1])
})

timers$shutdown()
cat("--- Completed successfully ---\n")
