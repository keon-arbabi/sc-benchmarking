#!/usr/bin/env python3
"""
diag_load_seaad.py: Verify that memory reported for SEAAD data loading is
consistent between single-threaded (num_threads=1) and multi-threaded
(num_threads=-1) after the Shmem-delta fix to monitor_mem.sh.

Ground-truth check uses the same formula as the fixed monitor:
  corrected = PSS_main + max(0, Shmem_delta - Pss_Shmem_main)

For single-threaded: data is private anonymous memory, PSS_main = full dataset,
Shmem_delta = 0 → corrected = PSS_main.

For multi-threaded: workers write data into tmpfs-backed Arena mmap (main's page
table stays empty until first access), so PSS_main ≈ interpreter only, but
Shmem_delta = full dataset → corrected = interpreter + dataset ≈ dataset.

Both modes should report similar corrected values (same data, same footprint).
"""

import os
import sys
sys.path.append('/home/karbabi')
sys.path.append('sc-benchmarking')
from single_cell import SingleCell
from utils_local import MemoryTimer, system_info

DATA_PATH = 'single-cell/SEAAD/SEAAD_raw.h5ad'


def read_shmem_kb():
    """Read system-wide Shmem from /proc/meminfo (KB)."""
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('Shmem:'):
                return int(line.split()[1])
    return 0


def read_corrected_gib(shmem_baseline_kb):
    """
    Compute the monitor's corrected memory estimate.

    PSS_main counts pages in main's page table (correct for private/already-accessed
    memory). UNCOUNTED_SHMEM adds the tmpfs-backed Arena pages that workers wrote but
    main hasn't yet touched — invisible to PSS but tracked in /proc/meminfo Shmem:.
    Subtracting Pss_Shmem avoids double-counting once main starts accessing data.
    """
    pss_kb = pss_shmem_kb = 0
    with open(f'/proc/{os.getpid()}/smaps_rollup') as f:
        for line in f:
            if line.startswith('Pss:'):
                pss_kb = int(line.split()[1])
            elif line.startswith('Pss_Shmem:'):
                pss_shmem_kb = int(line.split()[1])

    shmem_delta_kb = max(0, read_shmem_kb() - shmem_baseline_kb)
    uncounted_kb   = max(0, shmem_delta_kb - pss_shmem_kb)
    corrected_kb   = pss_kb + uncounted_kb

    return {
        'pss':       round(pss_kb       / 2**20, 2),
        'shmem_delta': round(shmem_delta_kb / 2**20, 2),
        'corrected': round(corrected_kb / 2**20, 2),
    }


system_info()
print()

timers = MemoryTimer(silent=False)

for num_threads in [1, -1]:
    shmem_baseline_kb = read_shmem_kb()
    with timers(f'Load data (num_threads={num_threads})'):
        data = SingleCell(DATA_PATH, num_threads=num_threads)
        # Measure here: inside the with block, workers dead, data live,
        # gc.collect() has NOT yet run. This is the true settled footprint.
        m = read_corrected_gib(shmem_baseline_kb)
        print(f'  Ground-truth — PSS: {m["pss"]} GiB  '
              f'Shmem delta: {m["shmem_delta"]} GiB  '
              f'Corrected: {m["corrected"]} GiB', flush=True)
    del data

timers.print_summary(unit='s')
