import os
import gc
import polars as pl
from utils import run 
from single_cell import SingleCell

os.makedirs('sc-benchmarking/figures', exist_ok=True)
os.makedirs('sc-benchmarking/logs', exist_ok=True)
os.makedirs('sc-benchmarking/out', exist_ok=True)

# region SEAAD 
# https://doi.org/10.21203/rs.3.rs-2921860/v1
# https://sea-ad-single-cell-profiling.s3.amazonaws.com/index.html

dir_data = 'single-cell/SEAAD'
os.makedirs(dir_data, exist_ok=True)

file_data = f'{dir_data}/SEAAD_MTG_RNAseq_all-nuclei.2024-02-13.h5ad'
file_metadata = f'{dir_data}/sea-ad_cohort_donor_metadata.xlsx'
file_ref_data = f'{dir_data}/Reference_MTG_RNAseq_final-nuclei.2022-06-07.h5ad'

# Prep SEAAD full data

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
cols = ['sample',  'cell_type', 'cp_score', 'ad_dx', 'apoe4_dosage',
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
            .alias('ad_dx'),
        pl.col('APOE Genotype')
            .cast(pl.String)
            .str.count_matches('4')
            .fill_null(strategy='mean')
            .round()
            .alias('apoe4_dosage'),
        pl.col('PMI')
            .cast(pl.Float64)
            .alias('pmi'))\
    .rename_obs({
        '_index': 'cell_id', 'Donor ID': 'sample', 'Subclass': 'cell_type',
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
    obs: _index, cell_id, sample, cp_score, ad_dx, apoe4_dosage, pmi,
         age_at_death, sex, class, subclass, cell_type, nCount_RNA,
         nFeature_RNA, percent.mt
    var: _index, gene_symbol
    uns: normalized, QCed

column        value                           
 _index        TTCATGTCAATGTTGC-L8TX_210429_0… 
 cell_id       TTCATGTCAATGTTGC-L8TX_210429_0… 
 sample        H21.33.003                      
 cp_score      0.238033394                     
 ad_dx         0                               
 apoe4_dosage  0                               
 pmi           10.0                            
 age_at_death  78.0                            
 sex           Male                            
 class         GABAergic                       
 subclass      Sst                             
 cell_type     Sst                             
 nCount_RNA    20335                           
 nFeature_RNA  5779                            
 percent.mt    0.00049176294   

column       value       
 _index       MIR1302-2HG 
 gene_symbol  MIR1302-2HG   
'''

sc.save(f'{dir_data}/SEAAD_raw.h5ad', overwrite=True)
sc.subsample_obs(n=50000, by_column='subclass', QC_column=None)\
    .save(f'{dir_data}/SEAAD_raw_50K.h5ad', overwrite=True)

del sc; gc.collect()

# Prep SEAAD reference data

cols = ['sample', 'cell_type']

sc = SingleCell(file_ref_data)\
    .tocsr()\
    .rename_obs({
        'sample_name': 'cell_id', 'external_donor_name_label': 'sample', 
        'subclass_label': 'cell_type'})\
    .set_obs_names('cell_id')\
    .select_obs(cols)\
    .filter_obs(pl.all_horizontal(pl.col(cols).is_not_null()))\
    .rename_var({'_index': 'gene_symbol'})\
    .drop_obsm([
        'X_scVI', 'X_umap', '_scvi_extra_categoricals',
        '_scvi_extra_continuous'])\
    .drop_obsp([
        'connectivities', 'distances'])\
    .drop_uns([
        '_scvi', 'cluster_label_colors', 'neighbors', 
        'subclass_label_colors', 'umap'])\
    .with_uns(QCed = True)

print(sc)
print(sc.peek_obs())
print(sc.peek_var())

'''
SingleCell dataset with 137,303 cells (obs), 36,601 genes (var),
and 782,484,725 non-zero entries (X)
    obs: _index, cell_id, sample, sex, class, subclass, cell_type
    var: _index, gene_symbol
    uns: normalized, QCed

column     value                           
 _index     AAACCCACAACTCATG-LKTX_191204_0… 
 cell_id    AAACCCACAACTCATG-LKTX_191204_0… 
 sample     H18.30.002                      
 sex        M                               
 class      GABAergic                       
 subclass   Pax6                            
 cell_type  Pax6  

column       value       
 _index       MIR1302-2HG 
 gene_symbol  MIR1302-2HG 
'''

sc.save(f'{dir_data}/SEAAD_ref.h5ad', overwrite=True)

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

cols = ['sample',  'cell_type', 'treatment']

sc = SingleCell(file_data)\
    .rename_obs({'_index': 'cell_id'})\
    .select_obs(cols)\
    .filter_obs(pl.all_horizontal(pl.col(cols).is_not_null()))\
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
SingleCell dataset with 9,697,974 cells (obs), 40,352 genes (var),
and 18,830,591,942 non-zero entries (X)
    obs: cell_id, sample, cell_type, treatment, nCount_RNA, nFeature_RNA,
         percent.mt
    var: gene_symbol
    uns: normalized, QCed

column        value          
 cell_id       89_103_005__s1 
 sample        Donor10_4-1BBL 
 cell_type     CD8 Naive      
 treatment     cytokine       
 nCount_RNA    4700           
 nFeature_RNA  2236           
 percent.mt    0.011914894  

column       value  
 gene_symbol  TSPAN6 
'''

sc.save(f'{dir_data}/Parse_PBMC_raw.h5ad', overwrite=True)
sc.subsample_obs(n=50000, by_column='subclass', QC_column=None)\
    .save(f'{dir_data}/Parse_PBMC_raw_50K.h5ad', overwrite=True)












import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 

data_query = SingleCell(file_data)
data_ref = SingleCell(file_ref_data)

data_query = data_query.qc(allow_float=True)
data_ref = data_ref\
    .set_var_names('feature_name')\
    .make_var_names_unique(separator='~')\
    .qc(
        allow_float=True)

data_ref, data_query = data_ref.hvg(data_query)
data_ref = data_ref.normalize()
data_query = data_query.normalize()
data_ref, data_query = data_ref.PCA(data_query)
data_ref, data_query = data_ref.harmonize(data_query)
data_query = data_query.label_transfer_from(
    data_ref, 'celltype_level_1', cell_type_column='cell_type_transferred',
    overwrite=True)

def confusion_matrix_plot(sc_obs, orig_col, trans_col):
    cm = pd.crosstab(
        sc_obs[orig_col].to_pandas(),
        sc_obs[trans_col].to_pandas()
    )
    cm.to_csv('cell_type_confusion.csv')
    norm_cm = cm.div(cm.sum(axis=1), axis=0).fillna(0)
    height = 6 + norm_cm.shape[0] * 0.05
    width = 8 + norm_cm.shape[1] * 0.07
    plt.figure(figsize=(width, height))
    ax = sns.heatmap(
        norm_cm,
        square=False,
        linewidths=0.5,
        cmap='rocket_r',
        vmin=0,
        vmax=1,
        cbar_kws={'shrink': 0.7}
    )
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.set_xlabel(f"Predicted: {trans_col}")
    ax.set_ylabel(f"Original: {orig_col}")
    plt.tight_layout()
    plt.savefig('cell_type_confusion.png', dpi=300)
    plt.show()

data_query.obs = data_query.obs.with_columns([
    pl.col('cell_type').cast(pl.Categorical),
    pl.col('cell_type_transferred').cast(pl.Categorical)
])

confusion_matrix_plot(
    data_query.obs, 'cell_type', 'cell_type_transferred'
)