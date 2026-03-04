import os
import gc
import polars as pl
from utils import run
from single_cell import SingleCell

# SEAAD
# https://doi.org/10.21203/rs.3.rs-2921860/v1
# https://sea-ad-single-cell-profiling.s3.amazonaws.com/index.html

dir_data = 'single-cell/SEAAD'
os.makedirs(dir_data, exist_ok=True)

file_data = f'{dir_data}/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad'
file_metadata = f'{dir_data}/sea-ad_cohort_donor_metadata.xlsx'
if not os.path.exists(file_data):
    run(f'wget https://sea-ad-single-cell-profiling.s3.amazonaws.com/'
        f'MTG/RNAseq/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad '
        f'-O {file_data}')
if not os.path.exists(file_metadata):
    run(f'wget https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.'
        f'divio-media.net/filer_public/b4/c7/b4c727e1-ede1-4c61-b2ee-bf1ae4a3ef68/'
        f'sea-ad_cohort_donor_metadata_072524.xlsx '
        f'-O {file_metadata}')

donor_metadata = pl.read_excel(file_metadata)
cols = ['sample',  'cell_type', 'cond', 'apoe4_dosage',
        'pmi', 'age_at_death', 'sex']

sc = SingleCell(file_data)
sc = sc\
    .cast_obs({'Donor ID': pl.String})\
    .join_obs(
        donor_metadata.select(['Donor ID'] +
        list(set(donor_metadata.columns).difference(sc.obs.columns))),
        on='Donor ID', validate='m:1')\
    .filter_obs(
        pl.col('Neurotypical reference').eq('False'))\
    .rename_obs({
        'exp_component_name': 'cell_id',
        'Donor ID': 'sample', 'Subclass': 'cell_type',
        'Cognitive Status': 'cond', 'APOE Genotype': 'apoe4_dosage',
        'Continuous Pseudo-progression Score': 'cp_score',
        'Age at Death': 'age_at_death', 'PMI': 'pmi', 'Sex': 'sex'})\
    .with_columns_obs(
        pl.when(pl.col('Consensus Clinical Dx (choice=Alzheimers disease)')
            .eq('Checked')).then(1)
            .otherwise(0)
            .alias('cond'),
        pl.col('apoe4_dosage')
            .cast(pl.String)
            .str.count_matches('4')
            .fill_null(strategy='mean')
            .round(),
        pl.col('pmi')
            .cast(pl.String)
            .cast(pl.Float32))\
    .select_obs(cols)\
    .rename_var({'gene_ids': 'gene_symbol'})\
    .set_var_names('gene_symbol')\
    .drop_var('_index')\
    .drop_obsm(list(sc.obsm))\
    .drop_obsp(list(sc.obsp))\
    .drop_uns(list(sc.uns))\
    .qc_metrics(
        num_counts_column='nCount_RNA',
        num_genes_column='nFeature_RNA',
        mito_fraction_column='percent.mt',
        allow_float=True)

print(sc)
print(sc.peek_obs())
print(sc.peek_var())

'''
SingleCell dataset in CSR format with 1,240,908 cells (obs), 36,601 genes (var),
and 6,839,265,617 non-zero entries (X)
    obs: cell_id, sample, cell_type, cond, apoe4_dosage, pmi, age_at_death,
         sex, nCount_RNA, nFeature_RNA, percent.mt
    var: gene_symbol
    uns: QCed, normalized

column        value
 cell_id       GGTGATTAGGTCACTT-L8TX_210722_0…
 sample        H20.33.034
 cell_type     Oligodendrocyte
 cond          0
 apoe4_dosage  0
 pmi           10.016666666667
 age_at_death  85.0
 sex           Female
 nCount_RNA    4401
 nFeature_RNA  2300
 percent.mt    0.00045444217

column       value
 gene_symbol  MIR1302-2HG
'''

sc.save(f'{dir_data}/SEAAD_raw.h5ad', overwrite=True)
sc.subsample_obs(n=50000, by_column='cell_type', QC_column=None)\
    .save(f'{dir_data}/SEAAD_raw_50K.h5ad', overwrite=True)

del sc; gc.collect()

# Parse 10M PBMC

# https://www.biorxiv.org/content/10.64898/2025.12.12.693897v1
# https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-experiment/

dir_data = 'single-cell/PBMC'
os.makedirs(dir_data, exist_ok=True)

file_data = f'{dir_data}/Parse_PBMC_cytokines.h5ad'
if not os.path.exists(file_data):
    run(f'wget https://parse-wget.s3.us-west-2.amazonaws.com/10m/'
        f'Parse_10M_PBMC_cytokines.h5ad -O {file_data}')

sc = SingleCell(file_data)\
    .rename_obs({'_index': 'cell_id', 'treatment': 'cond'})\
    .select_obs('sample', 'donor', 'cytokine', 'cond', 'cell_type')\
    .with_columns_obs(
        pl.when(pl.col('cond') == 'cytokine').then(1)
            .when(pl.col('cond') == 'PBS').then(0)
            .alias('cond'))\
    .rename_var({'_index': 'gene_symbol'})\
    .drop_var('n_cells')\
    .qc_metrics(
        num_counts_column='nCount_RNA',
        num_genes_column='nFeature_RNA',
        mito_fraction_column='percent.mt',
        allow_float=True)

print(sc)
print(sc.peek_obs())
print(sc.peek_var())

'''
SingleCell dataset in CSR format with 9,697,974 cells (obs), 40,352 genes (var), and 18,830,591,942 non-zero entries (X)
    obs: cell_id, sample, donor, cytokine, cond, cell_type, nCount_RNA,
         nFeature_RNA, percent.mt
    var: gene_symbol
    uns: QCed, normalized

column        value
 cell_id       89_103_005__s1
 sample        Donor10_4-1BBL
 donor         Donor10
 cytokine      4-1BBL
 cond          1
 cell_type     CD8 Naive
 nCount_RNA    4700
 nFeature_RNA  2236
 percent.mt    0.011914894

column       value
 gene_symbol  TSPAN6
'''

sc.save(f'{dir_data}/Parse_PBMC_raw.h5ad', overwrite=True)
sc.subsample_obs(n=200000, by_column='cell_type', QC_column=None)\
    .save(f'{dir_data}/Parse_PBMC_raw_200K.h5ad', overwrite=True)

del sc; gc.collect()
