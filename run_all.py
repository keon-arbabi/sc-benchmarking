import os
import sys
sys.path.append('sc-benchmarking')
from utils_local import run_slurm

DATA_DIR = 'single-cell'
WORK_DIR = 'sc-benchmarking'

OUTPUT_DIR = os.path.join(WORK_DIR, 'output')
LOG_DIR = os.path.join(WORK_DIR, 'logs')
FIGURES_DIR = os.path.join(WORK_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

RSCRIPT = '/home/wainberg/bin/Rscript-4.5.3'
PYTHON = '/home/wainberg/bin/python3.14'
PYTHON_RAPIDS = '/home/karbabi/miniforge3/bin/python'

DATASETS = {
    'SEAAD': os.path.join(DATA_DIR, 'SEAAD', 'SEAAD_raw.h5ad'),
    'PBMC': os.path.join(DATA_DIR, 'PBMC', 'Parse_PBMC_raw.h5ad'),
    'PANSCI': os.path.join(DATA_DIR, 'PanSci', 'PanSci_raw.h5ad'),
}

# (file, tool, task, thread_params)
CPU_SCRIPTS = [
    ('test_basic_brisc.py', 'brisc', 'basic', [-1, 1]),
    ('test_basic_scanpy.py', 'scanpy', 'basic', None),
    ('test_basic_seurat.R', 'seurat', 'basic', None),
    ('test_de_brisc.py', 'brisc', 'de', [-1, 1]),
    ('test_de_scanpy.py', 'scanpy', 'de', None),
    ('test_de_seurat.R', 'seurat', 'de', None),
    ('test_transfer_brisc.py', 'brisc', 'transfer', [-1, 1]),
    ('test_transfer_scanpy.py', 'scanpy', 'transfer', None),
    ('test_transfer_seurat.R', 'seurat', 'transfer', None),
    ('test_commands_brisc.py', 'brisc', 'commands', [-1, 1]),
    ('test_commands_scanpy.py', 'scanpy', 'commands', None),
    ('test_commands_seurat.R', 'seurat', 'commands', None),
]

GPU_SCRIPTS = [
    ('test_basic_rapids.py', 'rapids', 'basic', None),
    ('test_basic_brisc.py', 'brisc', 'basic', [-1, 1]),
]

OUTPUTS = {
    'basic': ['embedding', 'pcs', 'neighbors'],
    'de': ['de'],
    'transfer': ['accuracy'],
    'commands': [],
}

def out(job_name, suffix):
    return os.path.join(OUTPUT_DIR, f'{job_name}_{suffix}.csv')

def run_jobs(scripts, is_gpu=False):
    for script_file, tool, task, thread_params in scripts:
        if script_file.endswith('.R'):
            interpreter = RSCRIPT
        elif tool == 'rapids':
            interpreter = PYTHON_RAPIDS
        else:
            interpreter = PYTHON
        script_path = os.path.join(WORK_DIR, script_file)

        for d_name, d_path in DATASETS.items():
            for threads in (thread_params or [None]):
                parts = [f'{task}_{tool}', d_name]
                if threads is not None:
                    parts.append(str(threads))
                if is_gpu:
                    parts.append('gpu')
                job_name = '_'.join(parts)

                log = os.path.join(LOG_DIR, f'{job_name}.log')
                if os.path.exists(log):
                    with open(log) as f:
                        if 'Completed successfully' in f.read():
                            print(f'Skipping completed job: {job_name}')
                            continue

                cmd = [interpreter, script_path, d_name, d_path]
                if threads is not None:
                    cmd.append(str(threads))
                cmd.append(out(job_name, 'timer'))
                cmd.extend(out(job_name, s) for s in OUTPUTS[task])

                run_slurm(
                    ' '.join(cmd),
                    job_name=job_name, log_file=log,
                    CPUs=112 if is_gpu else 192,
                    GPUs=8 if is_gpu else 0,
                    hours=24)

if __name__ == '__main__':
    # run_jobs(CPU_SCRIPTS, is_gpu=False)
    run_jobs(GPU_SCRIPTS, is_gpu=True)
