#!/usr/bin/env python3
"""
diag_timing.py: Quantify timing overhead introduced by MemoryTimer's monitoring subprocess.

MemoryTimer spawns monitor_mem.sh which reads /proc/PID/smaps_rollup every 50ms for
the target process and all direct children. With many worker processes (e.g. 192 on a
full SLURM node), this creates:
  - More files read per polling cycle (linear in n_workers)
  - Repeated acquisition of each process's mmap_sem kernel lock in read mode
  - Contention with the main process's own memory operations (write-side of its mmap_sem)

This script isolates the effect by running an identical memory-intensive workload
(large array operations, similar to PCA's memory access pattern) with and without
the monitoring subprocess, across varying worker counts.

Expected result if monitoring causes overhead:
  - t_monitored > t_unmonitored
  - overhead ratio increases with n_workers

If ratios are flat and near 1.0, the PCA timing discrepancy between interactive and job
runs is due to other factors (different machine hardware, thread count, NUMA topology).
"""

import sys
import os
import time
import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from timeit import default_timer

sys.path.append('/home/karbabi')
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer

WORKER_COUNTS = [0, 4, 16, 64]   # Workers to hold shared memory during measurement
REPEATS = 3                        # Runs per configuration
WORKLOAD_GIB = 1                   # Array size for workload
WORKLOAD_PASSES = 10               # Passes over the array (more = more memory-intensive)
N = int((WORKLOAD_GIB * 1024**3) // 8)  # float64 elements


# ---------------------------------------------------------------------------
# Workload and worker definitions
# ---------------------------------------------------------------------------

def workload(arr):
    """
    Memory-intensive operation: repeated element-wise passes that touch every page.
    Chosen to mimic PCA's pattern of large, sequential memory access + arithmetic.
    """
    for _ in range(WORKLOAD_PASSES):
        arr += 0.001
    return float(arr[0])  # prevent dead-code elimination


def worker_hold(shm_name, shm_size, ready, stop):
    """
    Idle worker: maps the shared memory segment (so monitor_mem.sh polls its
    smaps_rollup), signals ready, then holds until stop. No memory operations —
    any overhead comes purely from the monitor reading this process's smaps_rollup.
    """
    shm = SharedMemory(name=shm_name)
    _ = memoryview(shm.buf)[0]   # touch one page to activate mapping
    ready.set()
    stop.wait()
    shm.close()


def spawn_workers(n, shm):
    if n == 0:
        return [], None
    ready_events = [mp.Event() for _ in range(n)]
    stop_event = mp.Event()
    workers = [
        mp.Process(target=worker_hold,
                   args=(shm.name, shm.size, ready_events[i], stop_event))
        for i in range(n)
    ]
    for w in workers:
        w.start()
    for ev in ready_events:
        ev.wait()
    return workers, stop_event


def join_workers(workers, stop_event):
    if workers:
        stop_event.set()
        for w in workers:
            w.join()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

print(f'Workload: {WORKLOAD_GIB} GiB array × {WORKLOAD_PASSES} passes | '
      f'{REPEATS} repeats per config')
print(f'Workers hold the same shared memory so monitor_mem.sh polls their smaps_rollup.')
print()

# Allocate workload array in shared memory so workers can map it
shm = SharedMemory(create=True, size=int(WORKLOAD_GIB * 1024**3))
arr = np.ndarray((N,), dtype=np.float64, buffer=shm.buf)
arr[:] = 1.0    # Touch all pages in main process

# Warmup (JIT, caches, etc.)
workload(arr)
workload(arr)


# ---------------------------------------------------------------------------
# Unmonitored baseline (no MemoryTimer subprocess)
# ---------------------------------------------------------------------------

t_unmon_list = []
for _ in range(REPEATS):
    t0 = default_timer()
    workload(arr)
    t_unmon_list.append(default_timer() - t0)

t_base = np.median(t_unmon_list)
print(f'Unmonitored baseline: median={t_base*1000:.1f}ms  '
      f'all={[f"{t*1000:.0f}ms" for t in t_unmon_list]}')
print()


# ---------------------------------------------------------------------------
# MemoryTimer overhead by worker count
# ---------------------------------------------------------------------------

hdr = (f'{"n_workers":>10}  {"t_base (ms)":>12}  {"t_mon med (ms)":>15}  '
       f'{"overhead":>10}  all (ms)')
print(hdr)
print('-' * 90)

for n_workers in WORKER_COUNTS:
    workers, stop_ev = spawn_workers(n_workers, shm)

    t_mon_list = []
    for _ in range(REPEATS):
        timers = MemoryTimer(silent=True)
        with timers('workload'):
            t0 = default_timer()
            workload(arr)
            t_inner = default_timer() - t0
        t_mon_list.append(t_inner)

    join_workers(workers, stop_ev)

    t_mon_med = np.median(t_mon_list)
    ratio = t_mon_med / t_base
    all_str = ', '.join(f'{t*1000:.0f}' for t in t_mon_list)
    print(f'{n_workers:>10}  {t_base*1000:>12.1f}  {t_mon_med*1000:>15.1f}  '
          f'{ratio:>9.2f}x  [{all_str}]')

shm.close()
shm.unlink()

print()
print('Interpretation:')
print('  overhead ratio ≈ 1.0 across all configs → monitoring is not causing the PCA slowdown;')
print('    look instead at: machine differences, thread count, NUMA topology between')
print('    interactive (login node) and job (compute node).')
print('  overhead ratio > 1.0 and scaling with n_workers → smaps_rollup polling is')
print('    contending with main-process memory operations; fix: increase POLLING_INTERVAL')
print('    or switch to a less intrusive measurement (e.g. /proc/PID/status VmRSS).')
