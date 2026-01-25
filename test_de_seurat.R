suppressPackageStartupMessages({
  library(dplyr)
  library(Seurat)
  library(BPCells)
})  

source("sc-benchmarking/utils_local.R")

args = commandArgs(trailingOnly=TRUE)
DATASET_NAME <- args[1]
DATA_PATH <- args[2]
OUTPUT_PATH_TIME <- args[3]

scratch_dir <- Sys.getenv("SCRATCH")
bpcells_dir_test <- file.path(scratch_dir, "bpcells", "de")
if (!dir.exists(bpcells_dir_test)) {
  dir.create(bpcells_dir_test, recursive = TRUE)
}

system_info()

cat("--- Params ---\n")
cat("seurat de\n")
cat(sprintf("R.version=%s\n", R.version.string))
cat(sprintf("DATASET_NAME=%s\n", DATASET_NAME))

timers <- MemoryTimer(silent = FALSE)

bpcells_dir <- file.path(bpcells_dir_test, "data")
if (file.exists(bpcells_dir)) {
  unlink(bpcells_dir, recursive = TRUE)
}

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

data$ad_dx <- ifelse(data$ad_dx == "1", "AD", "Control")

timers$with_timer("Data transformation (pseudobulk / normalization)", {
  data <- AggregateExpression(
    data, 
    assays = "RNA", 
    return.seurat = TRUE, 
    group.by = c("ad_dx", "sample", "subclass")
  )
})

timers$with_timer("Differential expression", {
  data$group <- paste(data$subclass, data$ad_dx, sep = "_")
  Idents(data) <- "group"

  de_list <- list()
  for (subclass in unique(data$subclass)) {
    de_list[[subclass]] <- FindMarkers(
      data, 
      ident.1 = paste(subclass, "AD", sep = "_"), 
      ident.2 = paste(subclass, "Control", sep = "_"), 
      test.use = "DESeq2"
    )
  }
  de <- do.call(rbind, de_list)
})

timers$print_summary(unit = "s")

timers_df <- timers$to_dataframe(unit = "s", sort = FALSE)
timers_df$library <- "seurat"
timers_df$test <- "de"
timers_df$dataset <- DATASET_NAME
write.csv(timers_df, OUTPUT_PATH_TIME, row.names = FALSE)

if (!any(timers_df$aborted)) {
  cat("--- Completed successfully ---\n")
}

unlink(bpcells_dir, recursive = TRUE)
rm(data, de, de_list, timers, timers_df, mat, mat_disk, obs_metadata)
gc()