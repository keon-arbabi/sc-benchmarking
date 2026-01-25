import os
import gc
import polars as pl
from utils import run 
from single_cell import SingleCell

# region SEAAD 
# https://doi.org/10.21203/rs.3.rs-2921860/v1
# https://sea-ad-single-cell-profiling.s3.amazonaws.com/index.html

dir_data = 'single-cell/SEAAD'
os.makedirs(dir_data, exist_ok=True)

file_data = f'{dir_data}/SEAAD_MTG_RNAseq_all-nuclei.2024-02-13.h5ad'
file_metadata = f'{dir_data}/sea-ad_cohort_donor_metadata.xlsx'
file_ref_data = f'{dir_data}/Reference_MTG_RNAseq_final-nuclei.2022-06-07.h5ad'

# Prep SEAAD MTG data

if not os.path.exists(file_data):
    run(f'wget https://sea-ad-single-cell-profiling.s3.amazonaws.com/'
        f'MTG/RNAseq/SEAAD_MTG_RNAseq_all-nuclei.2024-02-13.h5ad '
        f'-O {file_data}')
if not os.path.exists(file_metadata):
    run(f'wget https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.'
        f'divio-media.net/filer_public/b4/c7/b4c727e1-ede1-4c61-b2ee-bf1ae4a3ef68/'
        f'sea-ad_cohort_donor_metadata_072524.xlsx '
        f'-O {file_metadata}')
if not os.path.exists(file_ref_data):
    run(f'wget https://sea-ad-single-cell-profiling.s3.amazonaws.com/MTG/' 
        f'RNAseq/Reference_MTG_RNAseq_final-nuclei.2022-06-07.h5ad '
        f'-O {file_ref_data}')
    
donor_metadata = pl.read_excel(file_metadata)
cols = ['sample',  'cell_type', 'cp_score', 'cond', 'apoe4_dosage',
        'pmi', 'age_at_death', 'sex']

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
            .alias('cond'),
        pl.col('APOE Genotype')
            .cast(pl.String)
            .str.count_matches('4')
            .fill_null(strategy='mean')
            .round()
            .alias('apoe4_dosage'),
        pl.col('PMI')
            .cast(pl.String)
            .cast(pl.Float64)
            .alias('pmi'))\
    .rename_obs({
        'exp_component_name': 'cell_id', 
        'Donor ID': 'sample', 'Subclass': 'cell_type',
        'Continuous Pseudo-progression Score': 'cp_score',
        'Age at Death': 'age_at_death', 'Sex': 'sex'})\
    .select_obs(cols)\
    .filter_obs(pl.all_horizontal(pl.col(cols).is_not_null()))\
    .rename_var({'gene_ids': 'gene_symbol'})\
    .set_var_names('gene_symbol')\
    .drop_var('_index')\
    .drop_uns([
        'Great Apes Metadata', 'UW Clinical Metadata',
        'X_normalization', 'batch_condition', 'default_embedding',
        'title', 'normalized', 'QCed'])\
    .qc_metrics(
        num_counts_column='nCount_RNA',
        num_genes_column='nFeature_RNA', 
        mito_fraction_column='percent.mt',
        allow_float=True)

print(sc)
print(sc.peek_obs())
print(sc.peek_var())

'''
SingleCell dataset with 1,214,581 cells (obs), 36,601 genes (var),
and 5,827,739,288 non-zero entries (X)
    obs: cell_id, sample, cell_type, cp_score, cond, apoe4_dosage, pmi, 
        age_at_death,
        sex, nCount_RNA, nFeature_RNA, percent.mt
    var: gene_symbol
    uns: QCed, normalized

column        value                           
 cell_id       TTCATGTCAATGTTGC-L8TX_210429_0… 
 sample        H21.33.003                      
 cell_type     Sst                             
 cp_score      0.238033394                     
 cond          0                               
 apoe4_dosage  0                               
 pmi           10.0                            
 age_at_death  78.0                            
 sex           Male                            
 nCount_RNA    20335                           
 nFeature_RNA  5779                            
 percent.mt    0.00049176294                

column       value       
 gene_symbol  MIR1302-2HG 
'''

sc.save(f'{dir_data}/SEAAD_raw.h5ad', overwrite=True)
sc.subsample_obs(n=50000, by_column='cell_type', QC_column=None)\
    .save(f'{dir_data}/SEAAD_raw_50K.h5ad', overwrite=True)

del sc; gc.collect()

# endregion

# region Parse 10M PBMC
# https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-experiment/
# https://cellxgene.cziscience.com/collections/4a9fd4d7-d870-4265-89a5-ad51ab811d89

dir_data = 'single-cell/PBMC'
os.makedirs(dir_data, exist_ok=True)

file_data = f'{dir_data}/Parse_PBMC_cytokines.h5ad'
file_ref_data = f'{dir_data}/ScaleBio_PBMC_reference.h5ad'

if not os.path.exists(file_data):
    run(f'wget https://parse-wget.s3.us-west-2.amazonaws.com/10m/'
        f'Parse_10M_PBMC_cytokines.h5ad -O {file_data}')
if not os.path.exists(file_ref_data):
    run(f'wget https://datasets.cellxgene.cziscience.com/'
        f'428b51a9-6ea7-4c5b-a80a-e0ae77f2a4da.h5ad -O {file_ref_data}')

sc = SingleCell(file_data)\
    .rename_obs({'_index': 'cell_id', 'treatment': 'cond'})\
    .select_obs('sample', 'donor', 'cytokine', 'cond', 'cell_type')\
    .with_columns_obs(
        pl.col('cytokine').cast(pl.String)
            .str.replace_all('-', '_').alias('cytokine'))\
    .rename_var({'_index': 'gene_symbol'})\
    .drop_var('n_cells')\
    .qc_metrics(
        num_counts_column='nCount_RNA',
        num_genes_column='nFeature_RNA', 
        mito_fraction_column='percent.mt',
        allow_float=True)

print(sc)
print(sc.peek_obs())

sc.save(f'{dir_data}/Parse_PBMC_raw.h5ad', overwrite=True)
sc.subsample_obs(n=200000, by_column='cell_type', QC_column=None)\
    .save(f'{dir_data}/Parse_PBMC_raw_200K.h5ad', overwrite=True)

del sc; gc.collect()



