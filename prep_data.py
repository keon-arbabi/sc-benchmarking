import os
import gc
import polars as pl
from single_cell import SingleCell
from utils import run 

# SEAAD 
# https://sea-ad-single-cell-profiling.s3.amazonaws.com/index.html

file_data = 'single-cell/SEAAD/SEAAD_MTG_RNAseq_all-nuclei.2024-02-13.h5ad'
file_metadata = 'single-cell/SEAAD/sea-ad_cohort_donor_metadata.xlsx'
file_ref_data = 'single-cell/SEAAD/Reference_MTG_RNAseq_final-nuclei.2022-06-07.h5ad'

if not os.path.exists(file_data):
    run(f'wget https://sea-ad-single-cell-profiling.s3.amazonaws.com/'
        'MTG/RNAseq/SEAAD_MTG_RNAseq_all-nuclei.2024-02-13.h5ad '
        '-O {file_data}')
    run(f'wget https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.'
        'divio-media.net/filer_public/b4/c7/b4c727e1-ede1-4c61-b2ee-bf1ae4a3ef68/'
        'sea-ad_cohort_donor_metadata_072524.xlsx '
        '-O {file_metadata}')
    run(f'wget https://sea-ad-single-cell-profiling.s3.amazonaws.com/MTG/' 
        'RNAseq/Reference_MTG_RNAseq_final-nuclei.2022-06-07.h5ad '
        '-O {file_ref_data}')
    
donor_metadata = pl.read_excel(file_metadata)
cols = ['cell_id', 'sample', 'cp_score', 'ad_dx', 'apoe4_dosage',
        'pmi', 'age_at_death', 'sex', 'class', 'subclass']

sc = SingleCell(file_data)
sc = sc\
    .cast_obs({'Donor ID': pl.String})\
    .join_obs(
        donor_metadata.select(['Donor ID'] +
        list(set(donor_metadata.columns).difference(sc.obs.columns))),
        on='Donor ID', validate='m:1')\
    .filter_obs(pl.col('Neurotypical reference').eq('False'))\
    .with_columns_obs(
        pl.when(pl.col('Consensus Clinical Dx (choice=Alzheimers disease)')
            .eq('Checked')).then(1)
            .when(pl.col('Consensus Clinical Dx (choice=Control)')
            .eq('Checked')).then(0)
            .otherwise(None)
            .alias('ad_dx'),
        pl.col('Class').cast(pl.String).replace_strict({
            'Non-neuronal and Non-neural': 'Non-neuronal',
            'Neuronal: Glutamatergic': 'Glutamatergic',
            'Neuronal: GABAergic': 'GABAergic'})
            .alias('Class'),
        pl.col('APOE Genotype')
            .cast(pl.String)
            .str.count_matches('4')
            .fill_null(strategy='mean')
            .round()
            .alias('apoe4_dosage'))\
    .rename_obs({
        'exp_component_name': '_index',
        'sample_id': 'cell_id', 'Donor ID': 'sample',
        'Continuous Pseudo-progression Score': 'cp_score',
        'PMI': 'pmi', 'Age at Death': 'age_at_death', 'Sex': 'sex',
        'Class': 'class', 'Subclass': 'subclass'})\
    .select_obs(cols)\
    .filter_obs(pl.all_horizontal(pl.col(cols).is_not_null()))\
    .drop_uns([
        'Great Apes Metadata', 'UW Clinical Metadata',
        'X_normalization', 'batch_condition', 'default_embedding',
        'title', 'normalized', 'QCed'])\
    .qc(subset=False,
        QC_column='tmp_passed_QC',
        max_mito_fraction=0.05, 
        min_genes=100,
        nonzero_MALAT1=True,
        remove_doublets=True,
        batch_column='sample',
        allow_float=True)\
    .with_uns(QCed=False)

'''
Starting with 1,214,581 cells.
Filtering to cells with ≤5.0% mitochondrial counts...
1,174,861 cells remain after filtering to cells with ≤5.0% mitochondrial counts.
Filtering to cells with ≥100 genes detected (with non-zero count)...
1,174,861 cells remain after filtering to cells with ≥100 genes detected.
Filtering to cells with non-zero MALAT1 expression...
1,174,759 cells remain after filtering to cells with non-zero MALAT1 expression.
Removing predicted doublets...
894,168 cells remain after removing predicted doublets.
Adding a Boolean column, obs['tmp_passed_QC'], indicating which cells passed QC...
'''

path = 'single-cell/SEAAD/subsampled'
sizes = {1000000: '1M', 400000: '400K', 20000: '20K'}

for n, label in sizes.items():
    print(f'Subsampling to {label}')
    sc_sub = sc.subsample_obs(n=n, by_column='subclass', QC_column=None)
    
    print(f'Saving {label} h5ad')
    sc_sub.save(f'{path}/SEAAD_raw_{label}.h5ad', overwrite=True)

    print(f'Saving {label} h5')
    sc_sub\
        .with_columns_obs(
            pl.col('cell_id').alias('barcodes'))\
        .with_columns_var(
            pl.col('gene_ids').alias('name'),
            pl.col('gene_ids').alias('id'),
            pl.lit('Gene Expression').alias('feature_type'),
            pl.lit('unknown').alias('genome'))\
        .save(f'{path}/SEAAD_raw_{label}.h5', overwrite=True)

    print(f'Saving {label} mtx.gz')
    sc_sub.save(f'{path}/SEAAD_raw_{label}/matrix.mtx.gz', overwrite=True)

    if label != '1M':
        print(f'Saving {label} rds')
        sc_sub.save(f'{path}/SEAAD_raw_{label}.rds', overwrite=True)

        print(f'Saving {label} h5Seurat')
        sc_sub.save(f'{path}/SEAAD_raw_{label}.h5Seurat', overwrite=True)

    del sc_sub; gc.collect()


# sc = SingleCell(
#     'projects/rrg-wainberg/single-cell/Green/'
#     'p400_qced_shareable.h5ad')
# sc.X.sort_indices()
# sc = sc\
#     .rename_obs({
#         '_index': 'cell_id', 'projid': 'sample',
#         'state': 'subclass'})\
#     .with_columns_obs(
#         pl.col('subset').cast(pl.String).replace({
#             'CUX2+': 'Excitatory'})
#             .alias('class'))\
#     .select_obs(
#         ['sample', 'class', 'subclass'])\
#     .rename_var({'_index': 'gene_symbol'})\
#     .filter_obs(
#         pl.col('sample').is_not_null() & 
#         pl.col('class').is_not_null() &
#         pl.col('subclass').is_not_null())