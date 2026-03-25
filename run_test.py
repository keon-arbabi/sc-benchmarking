import os
import sys
sys.path.append('sc-benchmarking')
from utils_local import run_slurm

WORK_DIR = 'sc-benchmarking'
OUTPUT_DIR = os.path.join(WORK_DIR, 'output')
LOG_DIR = os.path.join(WORK_DIR, 'logs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

PYTHON = '/home/karbabi/miniforge3/bin/python'
DATA = 'single-cell/SEAAD/SEAAD_raw.h5ad'

BASIC_OUTPUTS = ['timer', 'embedding', 'pcs', 'neighbors']

def out(job, suffix):
    return os.path.join(OUTPUT_DIR, f'{job}_{suffix}.csv')

if __name__ == '__main__':

    for name in ['basic_scanpy', 'basic_rapids']:
        job = f'{name}_SEAAD'
        script = os.path.join(WORK_DIR, f'test_{name}.py')
        cmd = [PYTHON, script, 'SEAAD', DATA]
        if 'brisc' in name:
            cmd.append('-1')
            job += '_-1'
        cmd.extend(out(job, s) for s in BASIC_OUTPUTS)

        run_slurm(
            ' '.join(cmd), job_name=job,
            account='def-wainberg',
            log_file=os.path.join(LOG_DIR, f'{job}.log'),
            CPUs=112, GPUs=8, hours=2)

