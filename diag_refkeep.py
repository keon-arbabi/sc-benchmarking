#!/usr/bin/env python3
"""
diag_refkeep.py: Verify whether np.frombuffer(ctypes_buf) keeps ctypes_buf
alive after _read_parallel returns.

Imports SingleCell from single_cell_debug.py (which has debug prints at three
checkpoints in the multi-threaded loading path):

  [DEBUG _read_parallel]  Inside _read_parallel, after np.frombuffer wraps
                          each buffer, before the function returns.
                          Shows refcount(buf) and arr.base chain.

  [DEBUG __init__] #1     Right after _read_parallel returns in __init__,
                          before assembly. PSS here reveals if shared memory
                          survived the return.

  [DEBUG __init__] #2     After full assembly (X, obs, var, uns are set).
                          Shows whether self.X.data chains back to a live
                          ctypes buffer, and the settled PSS.

Reading the output:
  refcount(buf) in step 1 ≥ 3  → numpy holds a ref; buffers survive return
  refcount(buf) in step 1 = 2  → numpy does NOT hold a ref; buffers freed on return
  PSS at step 2 ≈ buffer total  → buffers still alive
  PSS at step 2 ≈ 0             → buffers already freed (bug confirmed)
"""

import sys
sys.path.insert(0, '/home/karbabi')
sys.path.insert(0, 'sc-benchmarking')

# Import from the debug copy of single_cell
import single_cell_debug as sc_mod
SingleCell = sc_mod.SingleCell

from utils_local import system_info

DATA_PATH = 'single-cell/SEAAD/SEAAD_raw.h5ad'

system_info()
print()
print('=' * 70)
print('Loading with num_threads=-1 (multi-threaded, uses shared memory)')
print('=' * 70)

data = SingleCell(DATA_PATH, num_threads=-1)

print(f'\nSingleCell loaded successfully.')
print(f'data._X is not None: {data._X is not None}')
if data._X is not None:
    print(f'data._X.data.dtype:  {data._X.data.dtype}')
    print(f'data._X.data.shape:  {data._X.data.shape}')
    # Try actually reading a value to confirm the data is accessible
    try:
        val = data._X.data[0]
        print(f'data._X.data[0]:     {val}  (data is accessible)')
    except Exception as e:
        print(f'data._X.data[0] FAILED: {e}  (data may be freed/dangling)')

del data
