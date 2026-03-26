import os
import sys
from pathlib import Path
import h5py
import numpy as np
import anndata as ad

_PROJECT_DIR = Path(__file__).resolve().parent
_HOME_DIR = _PROJECT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))
from utils_local import run

BASE_URL = 'https://storage.googleapis.com/arc-ctc-tahoe100/2025-02-25/h5ad'
PLATE_TEMPLATE = 'plate{}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad'
NUM_PLATES = 14
cache_path = str(_HOME_DIR / 'single-cell' / 'Tahoe-100M')
plates_dir = f'{cache_path}/plates'
output_file = f'{cache_path}/Tahoe_100M.h5ad'
os.makedirs(plates_dir, exist_ok=True)

# download per plate h5ads
for i in range(1, NUM_PLATES + 1):
    filename = PLATE_TEMPLATE.format(i)
    local_path = f'{plates_dir}/{filename}'
    if os.path.exists(local_path):
        print(f'Plate {i}/{NUM_PLATES} already downloaded, skipping')
        continue
    url = f'{BASE_URL}/{filename}'
    print(f'Downloading plate {i}/{NUM_PLATES}: {filename}')
    run(f'curl -fSL --retry 3 -o {local_path} {url}')
    print(f'  Saved ({os.path.getsize(local_path) / 1e9:.1f} GB)')

# downcast X/indices int64 -> int32
for i in range(1, NUM_PLATES + 1):
    path = f'{plates_dir}/{PLATE_TEMPLATE.format(i)}'
    with h5py.File(path, 'a') as f:
        if f['X/indices'].dtype != np.int64:
            print(f'Plate {i}/{NUM_PLATES}: already int32, skipping')
            continue
        print(f'Plate {i}/{NUM_PLATES}: downcasting indices int64 -> int32')
        src = f['X/indices']
        data = src[:].astype(np.int32)
        chunks = src.chunks
        del f['X/indices']
        f.create_dataset('X/indices', data=data, chunks=chunks)

# concat on disk
if os.path.exists(output_file):
    print(f'{output_file} already exists, skipping merge')
else:
    data_dict = {f'plate_{i}': f'{plates_dir}/{PLATE_TEMPLATE.format(i)}'
                 for i in range(1, NUM_PLATES + 1)}
    print(f'Merging {NUM_PLATES} plates with concat_on_disk...')
    ad.experimental.concat_on_disk(data_dict, output_file, label='plate')
    print(f'Wrote {output_file} ({os.path.getsize(output_file) / 1e9:.1f} GB)')

# verify
with h5py.File(output_file, 'r') as f:
    for key in ['X/data', 'X/indices', 'X/indptr']:
        print(f'{key}: dtype={f[key].dtype}, shape={f[key].shape}')
    print(f'obs columns: {list(f["obs"].keys())}')
