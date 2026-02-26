import sys
import warnings
import polars as pl
import scanpy as sc
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info

warnings.filterwarnings('ignore')

DATA_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH_TIME = sys.argv[3]
OUTPUT_PATH_DE = sys.argv[4]

system_info()
print('--- Params ---')
print('scanpy de')
print(f'{DATA_PATH=}')

if __name__ == '__main__':

    timers = MemoryTimer(silent=False)

    with timers('Load data'):
        data = sc.read_h5ad(DATA_PATH)

    with timers('Quality control'):
        data.var['mt'] = data.var_names.str.startswith('MT-')
        data.var['malat1'] = data.var_names == 'MALAT1'
        sc.pp.calculate_qc_metrics(
            data, qc_vars=['mt', 'malat1'], inplace=True)
        keep = ((data.obs['n_genes_by_counts'].values >= 100) &
                (data.obs['pct_counts_mt'].values <= 5) &
                (data.obs['pct_counts_malat1'].values > 0))
        data = data[keep].copy()

    with timers('Data transformation'):
        sc.pp.normalize_total(data)
        sc.pp.log1p(data)

    with timers('Differential expression'):
        de = {}
        if DATA_NAME == 'SEAAD':
            data.obs['cond'] = data.obs['cond'].astype(str)
            for cell_type in data.obs['cell_type'].unique():
                data_sub = data[data.obs['cell_type'] == cell_type].copy()
                sc.tl.rank_genes_groups(
                    data_sub,
                    groupby='cond',
                    groups=['1'],
                    reference='0',
                    method='wilcoxon',
                    key_added=f'de_{cell_type}'
                )
                de[cell_type] = sc.get.rank_genes_groups_df(
                    data_sub,
                    group='1',
                    key=f'de_{cell_type}'
                )

        elif DATA_NAME == 'PBMC':
            data = data[data.obs['cytokine'].isin(
                ['IFN-gamma', 'PBS'])].copy()
            for cell_type in data.obs['cell_type'].unique():
                data_sub = data[data.obs['cell_type'] == cell_type].copy()
                sc.tl.rank_genes_groups(
                    data_sub,
                    groupby='cytokine',
                    groups=['IFN-gamma'],
                    reference='PBS',
                    method='wilcoxon',
                    key_added=f'de_{cell_type}'
                )
                de[cell_type] = sc.get.rank_genes_groups_df(
                    data_sub,
                    group='IFN-gamma',
                    key=f'de_{cell_type}'
                )

    de_df = pl.concat([
        pl.from_pandas(
            df[['names', 'logfoldchanges', 'pvals', 'pvals_adj']])
        .rename({
            'names': 'gene', 'logfoldchanges': 'logFC',
            'pvals': 'p_value', 'pvals_adj': 'p_value_adj'})
        .with_columns(pl.lit(ct).alias('cell_type'))
        for ct, df in de.items()
    ])
    de_df.write_csv(OUTPUT_PATH_DE)

    timers.print_summary(unit='s')

    timers_df = timers\
        .to_dataframe(sort=False, unit='s')\
        .with_columns(
            pl.lit('scanpy').alias('library'),
            pl.lit('de').alias('test'),
            pl.lit(DATA_NAME).alias('dataset'),)
    timers_df.write_csv(OUTPUT_PATH_TIME)

    if not any(timers_df['aborted']):
        print('--- Completed successfully ---')

    print('\n--- Session Info ---')
    sc.logging.print_header()
