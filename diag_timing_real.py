#!/usr/bin/env python3
"""
diag_timing_real.py: Run the full SEAAD pipeline with MemoryTimer interactively.

Compares against two references:
  - Interactive Timer (no monitor): test.py / ipython runs
  - SLURM job MemoryTimer: submitted via run_all

If timings here match the SLURM job → SLURM environment causes the slowdown, not the monitor.
If timings here match the interactive Timer → the MemoryTimer monitor causes the slowdown.
"""

import sys
sys.path.append('/home/karbabi')
sys.path.append('sc-benchmarking')
from single_cell import SingleCell
from utils_local import MemoryTimer, system_info

DATA_PATH = 'single-cell/PBMC/Parse_PBMC_raw.h5ad'
NUM_THREADS = -1

system_info()
print()

timers = MemoryTimer(silent=False)

with timers('Load data'):
    data = SingleCell(DATA_PATH, num_threads=NUM_THREADS)

with timers('Quality control'):
    data = data.qc(subset=False, allow_float=True)

with timers('Feature selection'):
    data = data.hvg()

with timers('Normalization'):
    data = data.normalize()

with timers('PCA'):
    data = data.PCA()

with timers('Nearest neighbors'):
    data = data\
        .neighbors()\
        .shared_neighbors()

with timers('Clustering'):
    data = data.cluster(resolution=[0.5, 1.0, 2.0])

with timers('Embedding'):
    data = data.embed()

with timers('Plot embedding'):
    data.plot_embedding(
        'cell_type',
        'sc-benchmarking/figures/diag_timing_real_embedding.png')

with timers('Find markers'):
    markers = data.find_markers('cell_type')

timers.print_summary(unit='s')
