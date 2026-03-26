suppressPackageStartupMessages({
  library(BPCells)
  library(Seurat)
})

.script_dir <- dirname(normalizePath(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
source(file.path(.script_dir, "utils_local.R"))

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
  Sys.getenv("SCRATCH"), "bpcells", "manipulation", paste0("data_", DATA_NAME))
unlink(bpcells_dir, recursive = TRUE)

# Setup
mat_disk <- open_matrix_anndata_hdf5(path = DATA_PATH)
mat_disk <- convert_matrix_type(mat_disk, type = "uint32_t")
write_matrix_dir(mat = mat_disk, dir = bpcells_dir)
mat <- open_matrix_dir(dir = bpcells_dir)
obs_metadata <- read_h5ad_obs(DATA_PATH)
data <- CreateSeuratObject(counts = mat, meta.data = obs_metadata)
rm(mat_disk, mat, obs_metadata); gc()

data <- FindVariableFeatures(data)

counts <- GetAssayData(data, layer = "counts")
cell_name <- colnames(data)[1]
gene_name <- rownames(data)[1]
cell_type_select <- as.character(data$cell_type[1])
donors <- sort(unique(as.character(data$donor)))
donor_df <- data.frame(
  donor = donors,
  donor_index = seq_along(donors) - 1L,
  stringsAsFactors = FALSE
)

timers$with_timer("Get expression by cell", {
  as.matrix(counts[, cell_name])
})

timers$with_timer("Get expression by gene", {
  as.matrix(counts[gene_name, ])
})

timers$with_timer("Subset cells", {
  cells_keep <- colnames(data)[data$cell_type == cell_type_select]
  subset(data, cells = cells_keep)
})

timers$with_timer("Subset genes", {
  subset(data, features = VariableFeatures(data))
})

timers$with_timer("Subsample cells", {
  subset(data, cells = sample(colnames(data), 10000))
})

timers$with_timer("Select obs columns", {
  data@meta.data[, !sapply(data@meta.data, is.numeric), drop = FALSE]
})

timers$with_timer("Add metadata column", {
  ones <- rep(1L, ncol(data))
  ct_donor <- ave(ones, data$donor, data$cell_type, FUN = length)
  donor_total <- ave(ones, data$donor, FUN = length)
  ct_total <- ave(ones, data$cell_type, FUN = length)
  data$cell_type_enrichment <- (ct_donor / donor_total) / (ct_total / ncol(data))
})

timers$with_timer("Cast obs column", {
  data$cell_type <- as.character(data$cell_type)
})

timers$with_timer("Rename obs column", {
  idx <- which(names(data@meta.data) == "cell_type_enrichment")
  names(data@meta.data)[idx] <- "ct_enrichment"
})

timers$with_timer("Remove metadata column", {
  data$ct_enrichment <- NULL
})

timers$with_timer("Join obs metadata", {
  idx <- match(as.character(data$donor), donor_df$donor)
  data$donor_index <- donor_df$donor_index[idx]
})

timers$with_timer("Rename cells", {
  data <- RenameCells(data, add.cell.id = "prefix")
})

timers$with_timer("Split by obs column", {
  data_split <- SplitObject(data, split.by = "cell_type_broad")
})

timers$with_timer("Concatenate objects", {
  data <- merge(data_split[[1]], y = data_split[-1])
})

timers$shutdown()
cat("--- Completed successfully ---\n")
