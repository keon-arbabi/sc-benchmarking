import os
import sys
import glob
sys.path.append('sc-benchmarking')
from utils_local import run_slurm

WORK_DIR = 'sc-benchmarking'
OUTPUT_DIR = os.path.join(WORK_DIR, 'output')
LOG_DIR = os.path.join(WORK_DIR, 'logs')
FIGURES_DIR = os.path.join(WORK_DIR, 'figures')

PYTHON_RAPIDS = '/home/karbabi/miniforge3/envs/rapids/bin/python'
NVIDIA_LIBS = ':'.join(sorted(set(
    os.path.dirname(p) for p in glob.glob(
        '/home/karbabi/miniforge3/envs/rapids/lib/python3.13'
        '/site-packages/nvidia/**/lib*/*.so*', recursive=True))))

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

d_name = 'SEAAD'
d_path = os.path.join('single-cell', 'SEAAD', 'SEAAD_raw.h5ad')
job_name = f'basic_rapids_{d_name}'

cmd = [
    PYTHON_RAPIDS,
    os.path.join(WORK_DIR, 'test_basic_rapids.py'),
    d_name, d_path,
    os.path.join(OUTPUT_DIR, f'{job_name}_timer.csv'),
    os.path.join(OUTPUT_DIR, f'{job_name}_embedding.csv'),
    os.path.join(OUTPUT_DIR, f'{job_name}_pcs.csv'),
    os.path.join(OUTPUT_DIR, f'{job_name}_neighbors.csv'),
]

run_slurm(
    ' '.join(cmd),
    job_name=job_name,
    log_file=os.path.join(LOG_DIR, f'{job_name}.log'),
    account='def-wainberg',
    CPUs=1,
    gpus_per_node=1,
    hours=2,
    env={'PIP_CONFIG_FILE': '/dev/null',
         'PYTHONPATH': os.getcwd(),
         'LD_LIBRARY_PATH': NVIDIA_LIBS + ':$LD_LIBRARY_PATH',
         'CUPY_CACHE_DIR': '$SCRATCH/.cupy'})
