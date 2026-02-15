# sc-benchmarking Review Summary

Benchmarks single-cell RNA-seq analysis tools to compare **BRISC** vs **Scanpy** (Python) and **Seurat+BPCells** (R).

---

## Repository Structure

- **run_all.py** — SLURM orchestration with job dependencies
- **prep_data.py** — Dataset preparation using BRISC
- **monitor_mem.sh** — Memory monitoring subprocess
- **utils_local.py/.R** — MemoryTimer class, accuracy helpers, SLURM utilities
- **test_basic_\*** — Basic analysis benchmarks
- **test_de_\*** — Differential expression benchmarks
- **test_transfer_\*** — Label transfer benchmarks
- **comparison.R** — Results visualization

---

## Workflow Steps

**Basic Pipeline**
1. Load data
2. Quality control
3. Doublet detection
4. Feature selection (HVG)
5. Normalization
6. PCA
7. Nearest neighbors
8. Clustering (3 resolutions)
9. Embedding
10. Plot embedding
11. Find markers

**DE Pipeline**
1. Load data
2. Quality control
3. Doublet detection (or load from cache)
4. Data transformation (pseudobulk for BRISC/Seurat DESeq2)
5. Differential expression

**Transfer Pipeline**
1. Load data
2. Quality control
3. Doublet detection (or load from cache)
4. Split into reference/query
5. Feature selection (HVG)
6. Normalization
7. PCA
8. Transfer labels

---

## Datasets

**SEAAD** (Alzheimer's MTG nuclei)
- Full: 1.2M cells
- Subsampled: 50K, stratified by cell_type

**PBMC** (Parse 10M cytokine study)
- Full: 9.7M cells
- Subsampled: 200K, stratified by cell_type

---

## Memory Measurement

Memory is tracked via a background shell process that polls at 50ms intervals:

- Reads PSS (Proportional Set Size) from /proc/[pid]/smaps_rollup for the main process and all child processes
- Tracks /dev/shm usage delta to capture multiprocessing shared memory allocations
- Reports total = PSS sum + shared memory delta
- Also calculates percentage of total system memory
- Peak values across all samples are recorded per operation

PSS is more accurate than RSS for shared memory scenarios since it divides shared pages proportionally among processes.

TODO: Track across time?


---

## Implementation Quirks

### Doublet Caching (Scanpy & Seurat only)

Scanpy and Seurat write doublet results to a shared CSV file during the basic workflow. DE and transfer workflows then read from this cache instead of recomputing. SLURM job dependencies ensure the basic job completes first.

- **Scanpy** uses Scrublet with batch_key='sample'
- **Seurat** uses scDblFinder (see below)
- **BRISC** does not cache; runs find_doublets() fresh each time

### scDblFinder Complexity for Seurat

scDblFinder requires SingleCellExperiment objects, not Seurat objects. BPCells on-disk matrices cannot be passed directly. The workaround:

1. Extract full counts matrix from Seurat object
2. Loop through each sample individually
3. Convert each sample's counts from BPCells to dgCMatrix (dense)
4. Create a SingleCellExperiment for that sample
5. Run scDblFinder with BiocParallel (120 workers)
6. Collect and merge results

### BPCells Disk-Backed Storage

Seurat cannot read h5ad files directly. The loading process:

1. Open h5ad matrix via BPCells
2. Convert to uint32_t type
3. Write to disk-backed BPCells directory on $SCRATCH
4. Open the on-disk matrix
5. Read obs metadata separately via custom read_h5ad_obs() function
6. Create Seurat object from both pieces

### Seurat+BPCells SNN Graph Limitation

Seurat's `FindNeighbors` → `ComputeSNN` fails on very large datasets (e.g., PBMC 9.7M cells) with `std::bad_alloc`. The failure is in Seurat's C++ SNN implementation, which cannot handle this scale.

**Solution:** Replace `FindNeighbors()` + `FindClusters()` with BPCells native functions: `knn_hnsw()` + `knn_to_snn_graph()` + `cluster_graph_leiden()`. Applied consistently to both SEAAD and PBMC datasets.

- `knn_hnsw()` uses HNSW approximate nearest neighbors (same class as Seurat's Annoy-based search)
- `knn_to_snn_graph()` explicitly implements "the algorithm that Seurat uses to compute SNN graphs" (per BPCells source)
- `cluster_graph_leiden()` uses the Leiden community detection algorithm
- BPCells' `build_snn_graph_cpp` computes the filtered SNN graph directly in C++, avoiding the intermediate allocation issue in Seurat's `ComputeSNN`

This is defensible because the pipeline already uses BPCells for data loading and disk-backed storage — extending to kNN/SNN/clustering is the natural integration of the officially recommended Seurat+BPCells stack.

---

## DE Testing Design Differences

**BRISC**
- Aggregation: Pseudobulk by sample × cell_type
- Method: Linear model
- Covariates: Full adjustment (cond, apoe4_dosage, sex, age_at_death, log2 num_cells, log2 library_size)
- QC: `Pseudobulk.qc()` filters to samples with ≥10 cells, removes outliers (>3 SD zero-gene count), keeps genes expressed in ≥80% of samples per group, requires ≥2 samples per group; cell types failing criteria are excluded

**Scanpy**
- Aggregation: None (cell-level)
- Method: Wilcoxon rank-sum
- Covariates: None

**Seurat DESeq2**
- Aggregation: AggregateExpression()
- Method: DESeq2
- Covariates: None (simple group comparison)

**Seurat Wilcoxon**
- Aggregation: None (cell-level)
- Method: Wilcoxon rank-sum
- Covariates: None

BRISC is the only tool performing proper pseudobulk DE with covariate adjustment. Scanpy and Seurat Wilcoxon run cell-level tests per cell type, which inflates statistical power and ignores sample-level variation.

---

## Label Transfer Methods

**BRISC**
- Uses harmonize() for batch correction followed by label_transfer_from()
- Harmony-based alignment between reference and query

**Scanpy**
- Uses sc.tl.ingest()
- Projects query onto reference PCA space, then kNN-based label transfer

**Seurat**
- Uses FindTransferAnchors() + TransferData()
- CCA-based anchor identification (dims 1:30)

---

## Other Notable Differences

**Embedding**
- BRISC: LocalMAP
- Scanpy: UMAP
- Seurat: UMAP

**Clustering**
- BRISC: Custom implementation, tests 3 resolutions (0.5, 1.0, 2.0)
- Scanpy: Leiden via igraph, 2 iterations, tests 3 resolutions
- Seurat: Louvain-based FindClusters, tests 3 resolutions

**Highly Variable Genes**
- BRISC: Built-in hvg() method
- Scanpy: highly_variable_genes() with batch_key='sample', n_top_genes=2000
- Seurat: FindVariableFeatures() with VST, nfeatures=2000

**Marker Detection**
- BRISC: find_markers() by cell_type
- Scanpy: rank_genes_groups() with Wilcoxon method
- Seurat: FindAllMarkers() with only.pos=TRUE

**QC Thresholds**
- BRISC: qc() with allow_float=True, doublet filtering via passed_QC column
- Scanpy: min_genes=100 or 200, min_cells=3
- Seurat: nFeature_RNA > 200, percent.mt < 5

**Threading**
- BRISC: Configurable (1 = single-threaded, -1 = multi-threaded/all cores)
- Scanpy: Single-threaded
- Seurat: Single-threaded (except BiocParallel for scDblFinder)

---

## Output Files

- **output/{job}_timer.csv** — Per-operation timing and peak memory
- **output/{job}_accuracy.csv** — Label transfer accuracy by cell type
- **output/{job}_embedding.csv** — 2D coordinates (cell_id, embed_1, embed_2)
- **figures/{tool}_embedding_{dataset}.png** — UMAP/LocalMAP visualizations
