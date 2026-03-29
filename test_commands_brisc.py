import gc
import sys
from pathlib import Path
import polars as pl
import polars.selectors as cs
sys.path.append(f'{Path.home()}')
sys.path.append(f'{Path.home()}/sc-benchmarking')
from single_cell import SingleCell
from utils_local import MemoryTimer, system_info

DATA_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
NUM_THREADS = int(sys.argv[3])
OUTPUT_PATH_TIME = sys.argv[4]

if __name__ == '__main__':

    system_info()
    print('--- Params ---')
    print('brisc manipulation')
    print(f'{DATA_PATH=}')
    print(f'{NUM_THREADS=}')

    timers = MemoryTimer(
        silent=False, csv_path=OUTPUT_PATH_TIME, summary_unit='ms',
        csv_columns={'library': 'brisc', 'test': 'manipulation',
                     'dataset': DATA_NAME, 'num_threads': NUM_THREADS})

    # Setup
    data = SingleCell(DATA_PATH, num_threads=NUM_THREADS)\
        .qc(subset=False, allow_float=True)\
        .hvg(batch_column='donor')

    cell_name = data.obs_names[0]
    gene_name = data.var_names[0]
    cell_type_select = data.obs['cell_type'][0]
    donor_df = pl.DataFrame({
        'donor': data.obs['donor'].unique().sort(),
        'donor_index': range(data.obs['donor'].n_unique())
    })

    with timers('Get expression by cell'):
        data.cell(cell_name)

    with timers('Get expression by gene'):
        data.gene(gene_name)

    with timers('Subset cells'):
        data.filter_obs(pl.col('cell_type').eq(cell_type_select))

    with timers('Subset genes'):
        data.filter_var(pl.col('highly_variable').eq(True))

    with timers('Subsample cells'):
        data.subsample_obs(n=10_000)

    with timers('Select obs columns'):
        data.select_obs(cs.exclude(cs.numeric(), cs.first()))

    with timers('Add metadata column'):
        data = data.with_columns_obs(
            ((pl.len().over(['donor', 'cell_type']) /
              pl.len().over('donor')) /
             (pl.len().over('cell_type') / pl.len()))
            .alias('cell_type_enrichment'))

    with timers('Cast obs column'):
        data = data.cast_obs({'cell_type': pl.String})

    with timers('Rename obs column'):
        data = data.rename_obs({'cell_type_enrichment': 'ct_enrichment'})

    with timers('Remove metadata column'):
        data = data.drop_obs('ct_enrichment')

    with timers('Join obs metadata'):
        data = data.join_obs(donor_df, on='donor', validate='m:1')

    with timers('Rename cells'):
        obs_col = data.obs_names.name
        data = data.with_columns_obs(
            ('prefix_' + pl.col(obs_col).cast(pl.String)).alias(obs_col))

    with timers('Split by obs column'):
        data_split = list(data.split_by_obs('cell_type_broad').values())

    with timers('Concatenate objects'):
        data = data_split[0].concat_obs(data_split[1:])

    del data_split; gc.collect()

    with timers('Copy object'):
        data_copy = data.copy(deep=True)

    timers.shutdown()
    print('--- Completed successfully ---')
