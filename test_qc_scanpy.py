import gc
import sys
import warnings
import polars as pl
import scanpy as sc
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info

warnings.filterwarnings("ignore")

DATASET_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH_TIME = sys.argv[3]
OUTPUT_PATH_DOUBLET = sys.argv[4]

if __name__ == '__main__':

    system_info()
    print('--- Params ---')
    print('scanpy qc')
    print(f'{DATA_PATH=}')

    timers = MemoryTimer(silent=False)

    with timers('Load data'):
        data = sc.read_h5ad(DATA_PATH)

    with timers('Quality control'):
        data.var['mt'] = data.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(
            data, qc_vars=['mt'], inplace=True, log1p=True)
        sc.pp.filter_cells(data, min_genes=100)
        sc.pp.filter_genes(data, min_cells=3)

    with timers('Doublet detection'):
        sc.pp.scrublet(data, batch_key='sample')

    pl.DataFrame({
        'cell_id': data.obs_names.tolist(),
        'doublet_score': data.obs['doublet_score'].tolist(),
        'is_doublet': data.obs['predicted_doublet'].tolist()
    }).write_csv(OUTPUT_PATH_DOUBLET)

    with timers('Quality control'):
        data = data[data.obs['predicted_doublet'] == False].copy()

    timers.print_summary(unit='s')

    timers_df = timers.to_dataframe(sort=False, unit='s').with_columns(
        pl.lit('scanpy').alias('library'),
        pl.lit('qc').alias('test'),
        pl.lit(DATASET_NAME).alias('dataset'),)
    timers_df.write_csv(OUTPUT_PATH_TIME)

    if not any(timers_df['aborted']):
        print('--- Completed successfully ---')

    print('\n--- Session Info ---')
    sc.logging.print_header()

    del timers, timers_df, data
    gc.collect()
