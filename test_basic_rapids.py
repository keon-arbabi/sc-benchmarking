import os
import gc
import sys
import h5py
import numpy as np
import polars as pl
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer, system_info

import warnings
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')

os.environ.setdefault('CUDA_PATH', os.path.join(
    os.path.dirname(os.__file__), 'site-packages', 'nvidia', 'cuda_runtime'))

import rapids_singlecell as rsc
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from packaging.version import parse as parse_version

DATA_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH_TIME = sys.argv[3]
OUTPUT_PATH_EMBEDDING = sys.argv[4]
OUTPUT_PATH_PCS = sys.argv[5]
OUTPUT_PATH_NEIGHBORS = sys.argv[6]

CHUNK_SIZE = 50_000

if __name__ == '__main__':

    system_info()
    print('--- Params ---')
    print('rapids basic')
    print(f'{DATA_PATH=}')

    cluster = LocalCUDACluster(
        threads_per_worker=10,
        protocol='ucx',
        rmm_pool_size='10GB',
        rmm_maximum_pool_size='110GB',
        rmm_allocator_external_lib_list='cupy',
    )
    client = Client(cluster)
    print(client)

    if parse_version(ad.__version__) < parse_version('0.12.0rc1'):
        from anndata.experimental import read_elem_as_dask as read_dask
    else:
        from anndata.experimental import read_elem_lazy as read_dask

    timers = MemoryTimer(
        silent=False, csv_path=OUTPUT_PATH_TIME,
        csv_columns={'library': 'rapids', 'test': 'basic',
                     'dataset': DATA_NAME})

    with timers('Load data'):
        f = h5py.File(DATA_PATH, 'r')
        shape = tuple(f['X'].attrs['shape'])
        data = ad.AnnData(
            X=read_dask(f['X'], (CHUNK_SIZE, shape[1])),
            obs=ad.io.read_elem(f['obs']),
            var=ad.io.read_elem(f['var']),
        )
        rsc.get.anndata_to_GPU(data)
        data.X = data.X.persist()
        data.X.compute_chunk_sizes()

    with timers('Quality control'):
        data.var['mt'] = data.var_names.str.upper().str.startswith('MT-')
        data.var['malat1'] = data.var_names.str.upper() == 'MALAT1'
        rsc.pp.calculate_qc_metrics(data, qc_vars=['mt', 'malat1'])
        keep = ((data.obs['n_genes_by_counts'].values >= 100) &
                (data.obs['pct_counts_mt'].values <= 5) &
                (data.obs['pct_counts_malat1'].values > 0))
        data = data[keep].copy()
        data.X = data.X.persist()
        data.X.compute_chunk_sizes()

    gc.collect()

    with timers('Normalization'):
        rsc.pp.normalize_total(data)
        rsc.pp.log1p(data)
        data.X = data.X.persist()
        data.X.compute_chunk_sizes()

    with timers('Feature selection'):
        rsc.pp.highly_variable_genes(
            data, n_top_genes=2000, batch_key='donor')
        data = data[:, data.var.highly_variable].copy()

    with timers('PCA'):
        n_workers = len(client.scheduler_info()['workers'])
        rows_per_worker = (data.shape[0] + n_workers - 1) // n_workers
        data.X = data.X.rechunk((rows_per_worker, data.shape[1])).persist()
        data.X.compute_chunk_sizes()

        rsc.tl.pca(data, n_comps=50, mask_var=None)
        data.obsm['X_pca'] = data.obsm['X_pca'].persist()
        data.obsm['X_pca'].compute_chunk_sizes()
        data.obsm['X_pca'] = data.obsm['X_pca'].compute()

    with timers('Nearest neighbors'):
        rsc.pp.neighbors(
            data, n_neighbors=15, n_pcs=50, algorithm='mg_ivfflat')

    with timers('Embedding'):
        rsc.tl.umap(data)

    with timers('Clustering'):
        for res in [0.25, 0.5, 1.0, 1.5, 2.0]:
            rsc.tl.leiden(
                data,
                resolution=res,
                key_added=f'leiden_res_{res:4.2f}')

    with timers('Plot embedding'):
        rsc.get.anndata_to_CPU(data)
        sc.pl.umap(data, color='cell_type')
        plt.savefig(
            f'sc-benchmarking/figures/rapids_embedding_{DATA_NAME}.png',
            bbox_inches='tight', dpi=300)

    with timers('Find markers'):
        sc.tl.rank_genes_groups(data, groupby='cell_type')

    # Save PCs
    pcs = data.obsm['X_pca']
    if hasattr(pcs, 'get'):
        pcs = pcs.get()
    pc_df = pl.DataFrame({
        f'PC_{i+1}': pcs[:, i] for i in range(pcs.shape[1])
    })
    pc_df.write_csv(OUTPUT_PATH_PCS)

    # Save neighbors
    dist = data.obsp['distances']
    n_neighbors = dist.indptr[1] - dist.indptr[0] - 1
    neighbors = dist.indices.reshape(data.n_obs, -1)[:, 1:].astype(np.uint32)
    neighbors_df = pl.DataFrame({
        f'neighbor_{i+1}': neighbors[:, i]
        for i in range(n_neighbors)
    })
    neighbors_df.write_csv(OUTPUT_PATH_NEIGHBORS)

    # Save embeddings
    umap_coords = data.obsm['X_umap']
    if hasattr(umap_coords, 'get'):
        umap_coords = umap_coords.get()
    embedding_df = pl.DataFrame({
        'cell_id': data.obs_names.tolist(),
        'embed_1': umap_coords[:, 0],
        'embed_2': umap_coords[:, 1],
        'cell_type': data.obs['cell_type'].tolist(),
        'cell_type_broad': data.obs['cell_type_broad'].tolist(),
        'cluster_res_0.25': data.obs['leiden_res_0.25'].tolist(),
        'cluster_res_0.5': data.obs['leiden_res_0.50'].tolist(),
        'cluster_res_1.0': data.obs['leiden_res_1.00'].tolist(),
        'cluster_res_1.5': data.obs['leiden_res_1.50'].tolist(),
        'cluster_res_2.0': data.obs['leiden_res_2.00'].tolist(),
    })
    embedding_df.write_csv(OUTPUT_PATH_EMBEDDING)

    timers.shutdown()
    client.close()
    cluster.close()
    print('--- Completed successfully ---')

    print('\n--- Session Info ---')
    sc.logging.print_header()
