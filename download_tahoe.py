import os
cache_path = 'single-cell/Tahoe-100M'
os.environ['HF_HOME'] = cache_path
os.makedirs(cache_path, exist_ok=True)

from datasets import load_dataset
import anndata
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

expression = load_dataset(
    'vevotx/Tahoe-100M', streaming=False, split='train')
gene_meta = load_dataset(
    'vevotx/Tahoe-100M', name='gene_metadata', split='train')
sample_meta = load_dataset(
    'vevotx/Tahoe-100M', 'sample_metadata', split='train').to_pandas()
drug_meta = load_dataset(
    'vevotx/Tahoe-100M', 'drug_metadata', split='train').to_pandas()
cell_line_meta = load_dataset(
    'vevotx/Tahoe-100M', 'cell_line_metadata', split='train').to_pandas()

gene_vocab = {row['token_id']: row['ensembl_id'] for row in gene_meta}
token_ids, gene_names = zip(*sorted(gene_vocab.items()))
token_map = {token_id: i for i, token_id in enumerate(token_ids)}

data, indices, indptr = [], [], [0]
obs_data = []

n_cells_to_process = 50_000_000

for i, cell in enumerate(tqdm(expression, desc='Processing cells')):
    if i >= n_cells_to_process:
        break
        
    genes = cell['genes']
    exp = cell['expressions']
    
    if exp[0] < 0:
        genes, exp = genes[1:], exp[1:]

    cols = [token_map[g] for g in genes if g in token_map]
    exprs = [e for g, e in zip(genes, exp) if g in token_map]

    data.extend(exprs)
    indices.extend(cols)
    indptr.append(len(data))
    
    obs_data.append(
        {k: v for k, v in cell.items() if k not in ['genes', 'expressions']}
    )

shape = (len(indptr) - 1, len(gene_names))
expr_matrix = csr_matrix((data, indices, indptr), shape=shape)
obs_df = pd.DataFrame(obs_data)

adata = anndata.AnnData(X=expr_matrix, obs=obs_df)
adata.var.index = pd.Index(gene_names, name='ensembl_id')

adata.obs = adata.obs.merge(
    sample_meta.drop(columns=['drug', 'plate']), on='sample', how='left')
adata.obs = adata.obs.merge(drug_meta, on='drug', how='left')
adata.obs = adata.obs.merge(cell_line_meta, on='cell_line', how='left')

adata.write_h5ad(f'{cache_path}/Tahoe_50M.h5ad')