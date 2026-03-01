#!/usr/bin/env python3
"""
diag_memory.py: Reproduce and isolate the SHM double-counting bug in MemoryTimer.

monitor_mem.sh tracks two things that overlap for shared memory:
  1. PSS sum across main + direct children: proportional attribution means the sum
     across all processes = exactly 1 full copy (always correct).
  2. SHM_DELTA = CURRENT_/dev/shm - BASELINE_/dev/shm, where BASELINE is captured
     when the monitor subprocess starts (during the 100ms STARTUP_DELAY, before the
     measured operation begins).

The bug only fires when SharedMemory is allocated INSIDE the timer context, i.e.
AFTER the monitor has captured its baseline. SHM_DELTA then = the allocation size,
which is already fully counted in PSS → double-counting.

If SHM is allocated BEFORE the context (pre-allocated), the baseline includes it,
SHM_DELTA = 0, and there is no bug. The original test script pre-allocated in all
scenarios, masking the issue entirely.

In brisc with num_threads=-1, both the data loading into /dev/shm AND the worker
processes are created during the "Load data" operation (inside the context), which
triggers both mechanisms.

Four scenarios:
  1. Private array, allocated inside timer     → ratio ≈ 1.0 (correct reference)
  2. SHM pre-allocated before timer starts     → ratio ≈ 1.0 (SHM_DELTA = 0)
     This explains why the original test showed ~1.0 — the bug was not triggered.
  3. SHM allocated inside timer, no workers   → ratio ≈ 2.0 (SHM_DELTA double-counts)
     Minimal reproduction of the core bug.
  4. SHM inside timer + N workers w/ private  → ratio > 2.0 (SHM_DELTA + worker PSS)
     Workers are pre-spawned (pool exists before timer, as in brisc) but data is
     allocated inside the context. Represents the full "Load data" scenario.
"""

import sys
import os
import time
import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

sys.path.append('/home/karbabi')
sys.path.append('sc-benchmarking')
from utils_local import MemoryTimer

ALLOC_GIB   = 4    # GiB of data to allocate in each scenario
N_WORKERS   = 8    # Worker processes for scenarios 3-4
WORKER_GIB  = 0.5  # GiB of private memory per worker in scenario 4
ALLOC_BYTES  = int(ALLOC_GIB  * 1024**3)
WORKER_BYTES = int(WORKER_GIB * 1024**3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_pss_kb(pid):
    """Read PSS in KB from /proc/PID/smaps_rollup (mirrors monitor_mem.sh)."""
    try:
        with open(f'/proc/{pid}/smaps_rollup') as f:
            for line in f:
                if line.startswith('Pss:'):
                    return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return 0


def get_direct_children(pid):
    """Return direct child PIDs (same traversal as monitor_mem.sh)."""
    pids = []
    try:
        for task in os.listdir(f'/proc/{pid}/task'):
            try:
                with open(f'/proc/{pid}/task/{task}/children') as f:
                    for p in f.read().split():
                        if p.strip():
                            pids.append(int(p.strip()))
            except (FileNotFoundError, PermissionError):
                pass
    except (FileNotFoundError, PermissionError):
        pass
    return pids


def pss_sum_gib(pids):
    return sum(read_pss_kb(p) for p in pids) / 1024 / 1024


def worker_hold(shm_name, ready, stop):
    """Minimal worker: map shm, signal ready, hold."""
    shm = SharedMemory(name=shm_name)
    _ = memoryview(shm.buf)[0]
    ready.set()
    stop.wait()
    shm.close()


def worker_with_private(name_queue, priv_ready, shm_ready, stop):
    """
    Worker with private memory (mimics a live multiprocessing pool member):
      1. Allocates WORKER_GIB of private memory and signals priv_ready.
      2. Waits for the SHM name (sent inside the timer context when data is loaded).
      3. Maps the SHM, signals shm_ready, holds until stop.
    """
    priv = np.zeros(WORKER_BYTES // 8, dtype=np.float64)
    priv += 1.0           # Commit all private pages
    priv_ready.set()      # Private memory is live

    shm_name = name_queue.get()   # Block until main creates SHM inside timer
    shm = SharedMemory(name=shm_name)
    _ = memoryview(shm.buf)[0]
    shm_ready.set()
    stop.wait()
    shm.close()
    del priv


def print_header():
    print(f'\n{"Scenario":<52}  {"Alloc":>6}  {"PSS-only":>10}  '
          f'{"PSS/alloc":>10}  {"Reported":>10}  {"Rep/alloc":>10}  {"SHM_delta":>10}')
    print('-' * 118)


def print_row(label, alloc_gib, pss_gib, reported_gib):
    pss_ratio = pss_gib / alloc_gib if alloc_gib else float('nan')
    rep_ratio = reported_gib / alloc_gib if alloc_gib else float('nan')
    shm_delta = reported_gib - pss_gib
    print(f'{label:<52}  {alloc_gib:>5.1f}G  {pss_gib:>9.2f}G  '
          f'{pss_ratio:>9.2f}x  {reported_gib:>9.2f}G  {rep_ratio:>9.2f}x  {shm_delta:>9.2f}G')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print(f'Allocation: {ALLOC_GIB} GiB | Workers: {N_WORKERS} | '
      f'Worker private: {WORKER_GIB} GiB each')
print()
print('PSS/alloc ≈ 1.0 always → PSS sum is always the correct measure')
print('Rep/alloc > 1.0 only when SHM allocated inside timer → SHM_DELTA double-counts')
print_header()


# --- Scenario 1: Private array, inside timer (reference) ------------------
timers = MemoryTimer(silent=True)
with timers('s1'):
    arr = np.zeros(ALLOC_BYTES // 8, dtype=np.float64)
    arr += 1.0
    time.sleep(1.0)
    pids = [os.getpid()] + get_direct_children(os.getpid())
    pss_s1 = pss_sum_gib(pids)
del arr

print_row('1. Private array (inside timer)', ALLOC_GIB, pss_s1,
          timers.timings['s1']['memory'])


# --- Scenario 2: SHM pre-allocated before timer (original test pattern) ---
shm2 = SharedMemory(create=True, size=ALLOC_BYTES)
arr2 = np.ndarray((ALLOC_BYTES // 8,), dtype=np.float64, buffer=shm2.buf)
arr2 += 1.0             # Commit pages BEFORE timer starts

timers = MemoryTimer(silent=True)
with timers('s2'):
    time.sleep(1.0)
    pids = [os.getpid()] + get_direct_children(os.getpid())
    pss_s2 = pss_sum_gib(pids)
shm2.close(); shm2.unlink(); del arr2

print_row('2. SHM pre-allocated (SHM_DELTA=0, baseline captures it)',
          ALLOC_GIB, pss_s2, timers.timings['s2']['memory'])


# --- Scenario 3: SHM allocated INSIDE timer, no workers ------------------
timers = MemoryTimer(silent=True)
with timers('s3'):
    shm3 = SharedMemory(create=True, size=ALLOC_BYTES)   # allocated AFTER baseline
    arr3 = np.ndarray((ALLOC_BYTES // 8,), dtype=np.float64, buffer=shm3.buf)
    arr3 += 1.0         # SHM_DELTA now = ALLOC_GIB in every subsequent poll
    time.sleep(1.0)
    pids = [os.getpid()] + get_direct_children(os.getpid())
    pss_s3 = pss_sum_gib(pids)
    del arr3
shm3.close(); shm3.unlink()

print_row('3. SHM inside timer, no workers (SHM_DELTA > 0)',
          ALLOC_GIB, pss_s3, timers.timings['s3']['memory'])


# --- Scenario 4: SHM inside timer + N workers with private memory ---------
# Workers are pre-spawned (pool exists before the timer, as in brisc).
# Each holds WORKER_GIB of private memory. The SHM (data) is allocated
# inside the timer context, after the monitor baseline is captured.

name_queues  = [mp.Queue() for _ in range(N_WORKERS)]
priv_ready   = [mp.Event() for _ in range(N_WORKERS)]
shm_ready    = [mp.Event() for _ in range(N_WORKERS)]
stop_event   = mp.Event()

workers = [
    mp.Process(target=worker_with_private,
               args=(name_queues[i], priv_ready[i], shm_ready[i], stop_event))
    for i in range(N_WORKERS)
]
for w in workers:
    w.start()
for ev in priv_ready:
    ev.wait()           # All workers have committed their private memory

timers = MemoryTimer(silent=True)
with timers('s4'):
    shm4 = SharedMemory(create=True, size=ALLOC_BYTES)   # allocated AFTER baseline
    arr4 = np.ndarray((ALLOC_BYTES // 8,), dtype=np.float64, buffer=shm4.buf)
    arr4 += 1.0

    for q in name_queues:
        q.put(shm4.name)    # Workers now map the SHM
    for ev in shm_ready:
        ev.wait()

    time.sleep(1.0)
    pids = [os.getpid()] + get_direct_children(os.getpid())
    pss_s4 = pss_sum_gib(pids)
    del arr4

stop_event.set()
for w in workers:
    w.join()
shm4.close(); shm4.unlink()

# Expected denominator: data + genuine worker private footprint
expected_s4 = ALLOC_GIB + N_WORKERS * WORKER_GIB
print_row(
    f'4. SHM inside timer + {N_WORKERS} workers ({WORKER_GIB}G priv each)',
    expected_s4, pss_s4, timers.timings['s4']['memory'])

print()
print('Scenarios 1-2: SHM_delta = 0 → MemoryTimer is correct')
print('Scenario 3:    SHM_delta ≈ ALLOC_GIB → data double-counted by SHM_DELTA')
print('Scenario 4:    SHM_delta ≈ ALLOC_GIB + worker PSS inflation (both effects)')
print()
print('For brisc -1 (192 workers, ~172 GiB data allocated inside "Load data"):')
print('  Expected: reported = PSS-only  ≈ 172 GiB (data) + worker_private_overhead')
print('  Observed: reported ≈ 412 GiB  = PSS-only + SHM_DELTA (≈ 172 GiB double-counted)')
print('  Fix: remove SHM_DELTA addition from monitor_mem.sh; PSS sum alone is correct.')
