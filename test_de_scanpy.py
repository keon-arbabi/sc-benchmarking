import os
import gc
import sys
import decoupler as dc
import polars as pl
import scanpy as sc
from pydeseq2.ds import DeseqStats
from pydeseq2.dds import DeseqDataSet, DefaultInference
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info

DATA_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH_TIME = sys.argv[3]
OUTPUT_PATH_DE = sys.argv[4]

if __name__ == '__main__':

    system_info()
    print('--- Params ---')
    print('scanpy de deseq')
    print(f'{DATA_PATH=}')

    timers = MemoryTimer(silent=False)

    with timers('Load data'):
        data_sc = sc.read_h5ad(DATA_PATH)

    with timers('Quality control'):
        data_sc.var['mt'] = data_sc.var_names.str.startswith('MT-')
        data_sc.var['malat1'] = data_sc.var_names == 'MALAT1'
        sc.pp.calculate_qc_metrics(
            data_sc, qc_vars=['mt', 'malat1'], inplace=True)
        keep = ((data_sc.obs['n_genes_by_counts'].values >= 100) &
                (data_sc.obs['pct_counts_mt'].values <= 5) &
                (data_sc.obs['pct_counts_malat1'].values > 0))
        data_sc = data_sc[keep].copy()

    with timers('Pseudobulk'):
        data_pb = dc.pp.pseudobulk(
            data_sc, sample_col='sample', groups_col='cell_type', mode='sum')

    del data_sc; gc.collect()

    data_pb = data_pb[data_pb.obs['cond'].notna()].copy()

    with timers('Filter'):
        dc.pp.filter_samples(data_pb, min_cells=10, min_counts=1000)

    with timers('Differential expression'):
        inference = DefaultInference(n_cpus=os.cpu_count())
        cell_types = data_pb.obs['cell_type'].unique()

        if DATA_NAME == 'SEAAD':
            contrast = ['cond', 'AD', 'Control']
        elif DATA_NAME == 'PBMC':
            contrast = ['cond', 'IFN-gamma', 'PBS']

        de = {}
        for ct in cell_types:
            data_pb_ct = data_pb[data_pb.obs['cell_type'] == ct].copy()
            dc.pp.filter_by_expr(data_pb_ct, group='cond', min_count=10)
            dds = DeseqDataSet(
                adata=data_pb_ct, design='~ cond',
                refit_cooks=True, inference=inference, quiet=True)
            dds.deseq2()
            stat_res = DeseqStats(
                dds, contrast=contrast,
                inference=inference, quiet=True)
            stat_res.summary()
            de[ct] = stat_res.results_df

    de_df = pl.concat([
        pl.DataFrame({
            'cell_type': ct,
            'gene': df.index.tolist(),
            'logFC': df['log2FoldChange'].values,
            'p_value': df['pvalue'].values,
            'p_value_adj': df['padj'].values,
        })
        for ct, df in de.items()])
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
