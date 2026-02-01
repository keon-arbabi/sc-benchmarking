import gc
import sys
import polars as pl
sys.path.append('/home/karbabi')
from single_cell import SingleCell
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info

DATASET_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
NUM_THREADS = int(sys.argv[3])
OUTPUT_PATH_TIME = sys.argv[4]

if __name__ == '__main__':

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

    with timers('Data transformation'):
        data_pb = data_sc.pseudobulk('sample', 'cell_type')

    del data_sc; gc.collect()

    with timers('Data transformation'):
        if DATASET_NAME == 'SEAAD':
            data_pb = data_pb.qc('cond', verbose=False)

        elif DATASET_NAME == 'PBMC':
            data_pb = data_pb\
                .filter_obs(pl.col('cytokine').is_in(['IFN-gamma', 'PBS']))\
                .qc('cytokine', verbose=False)

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
            contrasts = {
                'IFN-gamma_vs_PBS': 'cytokineIFN-gamma - cytokinePBS'}
            de = data_pb\
                .library_size()\
                .DE(formula,
                    contrasts=contrasts,
                    group='cytokine',
                    categorical_columns=['donor', 'cytokine'],
                    verbose=False)

    timers.print_summary(unit='s')

    timers_df = timers.to_dataframe(sort=False, unit='s')\
        .with_columns(
            pl.lit('brisc').alias('library'),
            pl.lit('de').alias('test'),
            pl.lit(DATASET_NAME).alias('dataset'),
            pl.lit('single-threaded' if NUM_THREADS == 1 else 'multi-threaded')
            .alias('num_threads'))
    timers_df.write_csv(OUTPUT_PATH_TIME)

    if not any(timers_df['aborted']):
        print('--- Completed successfully ---')

    del timers, timers_df, data_pb, de
    gc.collect()
