import os 
import gc
import sys
import polars as pl  
sys.path.append('/home/karbabi') 
from single_cell import SingleCell  
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info

os.environ['R_HOME'] = os.path.expanduser('~/miniforge3/lib/R')
from ryp import r
r('.libPaths(c(file.path(Sys.getenv("CONDA_PREFIX"), "lib/R/library")))')

# DATASET_NAME = sys.argv[1]
# DATA_PATH = sys.argv[2]
# NUM_THREADS = int(sys.argv[3])
# OUTPUT_PATH = sys.argv[4]

DATASET_NAME = 'PBMC'
DATA_PATH = 'single-cell/PBMC/Parse_PBMC_raw.h5ad'
NUM_THREADS = -1
OUTPUT_PATH_1= 'sc-benchmarking/output/test_de_brisc_PBMC_-1_timers.csv'
OUTPUT_PATH_2= 'sc-benchmarking/output/test_de_brisc_PBMC_-1_table.csv'

DATASET_NAME = 'SEAAD'
DATA_PATH = 'single-cell/SEAAD/SEAAD_raw_50K.h5ad'
NUM_THREADS = -1
OUTPUT_PATH = 'sc-benchmarking/output/test_de_brisc_SEAAD_-1.csv'

system_info()

print('--- Params ---')
print('brisc de')
print(f'{DATASET_NAME=}')
print(f'{NUM_THREADS=}')

timers = MemoryTimer(silent=False)

with timers('Load data'):
    data_sc = SingleCell(DATA_PATH, num_threads=NUM_THREADS)

with timers('Quality control'):
    data_sc = data_sc.qc(
        subset=False,
        remove_doublets=False,
        allow_float=True,
        verbose=False)

with timers('Doublet detection'):
    data_sc = data_sc.find_doublets(batch_column='sample')

with timers('Quality control'):
    data_sc = data_sc.filter_obs(
        pl.col('doublet').not_() & pl.col('passed_QC'))

with timers('Data transformation (pseudobulk / normalization)'):
    data_pb = data_sc.pseudobulk('sample', 'cell_type')

del data_sc; gc.collect()

with timers('Data transformation (pseudobulk / normalization)'):
    data_pb = data_pb.qc('cond', verbose=False)

with timers('Differential expression'):
    if DATASET_NAME == 'SEAAD':
        formula = '~ cond + apoe4_dosage + sex + age_at_death + ' \
            'log2(num_cells) + log2(library_size)'
        de = data_pb\
            .library_size()\
            .DE(formula,
                group='cond',
                verbose=False)

    elif DATASET_NAME == 'PBMC':
        formula = '~ 0 + cytokine + donor + ' \
            'log2(num_cells) + log2(library_size)'
        contrasts = {'IFN_gamma_vs_PBS': 'cytokineIFN_gamma - cytokinePBS'}
        de = data_pb\
            .library_size()\
            .DE(formula,
                contrasts=contrasts,
                group='cytokine',
                categorical_columns=['donor', 'cytokine'],
                verbose=True)

de.table.write_csv(OUTPUT_PATH_2)                

timers.print_summary(unit='s')

timers_df = timers.to_dataframe(sort=False, unit='s')\
    .with_columns(
        pl.lit('brisc').alias('library'),
        pl.lit('de').alias('test'),
        pl.lit(DATASET_NAME).alias('dataset'),
        pl.lit('single-threaded' if NUM_THREADS == 1 else 'multi-threaded')
        .alias('num_threads'))
timers_df.write_csv(OUTPUT_PATH)

if not any(timers_df['aborted']):
    print('--- Completed successfully ---')

del timers, timers_df, data_pb, de
gc.collect()
