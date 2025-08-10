import gc
import sys
import polars as pl  
from single_cell import SingleCell  
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info

DATASET_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
NUM_THREADS = int(sys.argv[3])
OUTPUT_PATH = sys.argv[4]

DATASET_NAME = 'SEAAD'
DATA_PATH = 'single-cell/SEAAD/SEAAD_raw_50K.h5ad'
NUM_THREADS = -1
OUTPUT_PATH = 'sc-benchmarking/output/test_de_brisc_SEAAD_-1.csv'

DATASET_NAME = 'PBMC'
DATA_PATH = 'single-cell/PBMC/Parse_PBMC_raw.h5ad'
NUM_THREADS = -1
OUTPUT_PATH = 'sc-benchmarking/output/test_de_brisc_PBMC_-1.csv'

print('--- Params ---')
print(f'{NUM_THREADS=}')

system_info()
timers = MemoryTimer(silent=True)

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

with timers('Data transformation (pseudobulk / normalization)'):
    data_pb = data_sc.pseudobulk('sample', 'cell_type')

del data_sc; gc.collect()

with timers('Data transformation (pseudobulk / normalization)'):
    if DATASET_NAME == 'SEAAD':
        data_pb = data_pb.qc('ad_dx', verbose=False)

    elif DATASET_NAME == 'PBMC':
        data_pb = data_pb.qc('treatment', verbose=False)

with timers('Differential expression'):
    if DATASET_NAME == 'SEAAD':
        formula = '~ ad_dx + apoe4_dosage + sex + age_at_death + ' \
            'log2(num_cells) + log2(library_size)'
        de = data_pb\
            .library_size()\
            .DE(formula,
                group='ad_dx',
                verbose=False)
                
    elif DATASET_NAME == 'PBMC':
        formula = '~ 0 + cytokine + donor + ' \
            'log2(num_cells) + log2(library_size)'
        contrasts = {
            cell_type: {
                f'{c}_vs_PBS': f'cytokine{c} - cytokinePBS'
                for c in obs['cytokine'].unique() if c != 'PBS'
            }
            for cell_type, (_, obs, _) in data_pb.items()
        }
        de_results = data_pb\
            .library_size()\
            .DE(formula,
                contrasts=contrasts,
                group='cytokine',
                verbose=False)


    
timers.print_summary(sort=False)

df = timers.to_dataframe(sort=False, unit='s').with_columns(
    pl.lit('brisc').alias('library'),
    pl.lit('de').alias('test'),
    pl.lit(DATASET_NAME).alias('dataset'),
    pl.lit(NUM_THREADS).alias('num_threads'),
)
df.write_csv(OUTPUT_PATH)

del data, de, timers, df
gc.collect()
