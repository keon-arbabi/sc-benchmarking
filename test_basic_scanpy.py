import sys
import polars as pl  
import scanpy as sc  
import matplotlib.pyplot as plt  
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info

# DATASET_NAME = sys.argv[1]
# DATA_PATH = sys.argv[2]
# OUTPUT_PATH = sys.argv[3]

DATASET_NAME = 'SEAAD'
DATA_PATH = 'single-cell/SEAAD/SEAAD_raw.h5ad'
OUTPUT_PATH = 'sc-benchmarking/output/test_basic_brisc_SEAAD_1.csv'

system_info()
timers = MemoryTimer(silent=True)

with timers('Load data'):
    data = sc.read_h5ad(DATA_PATH)

with timers('Quality control'):
    data.var['mt'] = data.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(data, qc_vars=['mt'], inplace=True, log1p=True)
    sc.pp.filter_cells(data, min_genes=100)
    sc.pp.filter_genes(data, min_cells=3)

with timers('Doublet detection'):
    sc.pp.scrublet(data, batch_key='sample')

with timers('Quality control'):
    data = data[data.obs['predicted_doublet'] == False].copy()

with timers('Normalization'):
    sc.pp.normalize_total(data)
    sc.pp.log1p(data)

with timers('Feature selection'):
    sc.pp.highly_variable_genes(data, n_top_genes=2000, batch_key='sample')

with timers('PCA'):
    sc.tl.pca(data)

with timers('Nearest neighbors'):
    sc.pp.neighbors(data)

with timers('Embedding'):
    sc.tl.umap(data)

with timers('Clustering (3 res.)'):
    for res in [0.5, 1.0, 2.0]:
        sc.tl.leiden(
            data, 
            resolution=res,
            flavor='igraph',
            n_iterations=2,
            key_added=f'leiden_res_{res:4.2f}')

with timers('Plot embedding'):
    sc.pl.umap(data, color=['cell_type'])
    plt.savefig(
        f'sc-benchmarking/figures/scanpy_embedding.png',
        dpi=300,
        bbox_inches='tight',
        pad_inches='layout')

with timers('Find markers'):
    sc.tl.rank_genes_groups(data, groupby='cell_type', method='wilcoxon')

timers.print_summary(sort=False)

timers_df = timers.to_dataframe(sort=False, unit='s').with_columns(
    pl.lit('scanpy').alias('library'),
    pl.lit('basic').alias('test'),
    pl.lit(DATASET_NAME).alias('dataset'),)
timers_df.write_csv(OUTPUT_PATH)

if not all(timers_df['aborted']):
    print('--- Completed successfully ---')

