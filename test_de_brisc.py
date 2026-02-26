import gc
import sys
import polars as pl
sys.path.append('/home/karbabi')
from single_cell import SingleCell
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info

DATA_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
NUM_THREADS = int(sys.argv[3])
OUTPUT_PATH_TIME = sys.argv[4]
OUTPUT_PATH_DE = sys.argv[5]

system_info()
print('--- Params ---')
print('brisc de')
print(f'{DATA_PATH=}')
print(f'{NUM_THREADS=}')

if __name__ == '__main__':

    timers = MemoryTimer(silent=False)

    with timers('Load data'):
        data_sc = SingleCell(DATA_PATH, num_threads=NUM_THREADS)

    with timers('Quality control'):
        data_sc = data_sc.qc(subset=False, allow_float=True)

    with timers('Pseudobulk'):
        data_pb = data_sc.pseudobulk('sample', 'cell_type')

    del data_sc; gc.collect()

    with timers('Filter'):
        if DATA_NAME == 'SEAAD':
            data_pb = data_pb.qc('cond', verbose=False)

        elif DATA_NAME == 'PBMC':
            data_pb = data_pb\
                .filter_obs(pl.col('cytokine').is_in(['IFN-gamma', 'PBS']))\
                .qc('cytokine', verbose=False)

    with timers('Differential expression'):
        if DATA_NAME == 'SEAAD':
            formula = '~ cond + apoe4_dosage + sex + age_at_death + ' \
                'log2(num_cells) + log2(library_size)'
            de = data_pb\
                .library_size()\
                .DE(formula,
                    group='cond',
                    verbose=False)

        elif DATA_NAME == 'PBMC':
            formula = '~ 0 + cytokine + donor + ' \
                'log2(num_cells) + log2(library_size)'
            contrasts = {
                'IFN-gamma_vs_PBS': '`cytokineIFN-gamma` - `cytokinePBS`'}
            de = data_pb\
                .library_size()\
                .DE(formula,
                    contrasts=contrasts,
                    group='cytokine',
                    categorical_columns=['donor', 'cytokine'],
                    verbose=False)

    de_df = de.table\
        .select('cell_type', 'gene', 'logFC', 'p', 'FDR')\
        .rename({'p': 'p_value', 'FDR': 'p_value_adj'})\
    de_df.write_csv(OUTPUT_PATH_DE)

    timers.print_summary(unit='s')

    timers_df = timers.to_dataframe(sort=False, unit='s')\
        .with_columns(
            pl.lit('brisc').alias('library'),
            pl.lit('de').alias('test'),
            pl.lit(DATA_NAME).alias('dataset'),
            pl.lit(NUM_THREADS).alias('num_threads'))
    timers_df.write_csv(OUTPUT_PATH_TIME)

    if not any(timers_df['aborted']):
        print('--- Completed successfully ---')
