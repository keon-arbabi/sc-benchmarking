import os
import sys
import glob
sys.path.append('sc-benchmarking')
from utils_local import run_slurm

DATA_DIR = 'single-cell'
WORK_DIR = 'sc-benchmarking'

OUTPUT_DIR = os.path.join(WORK_DIR, 'output')
LOG_DIR = os.path.join(WORK_DIR, 'logs')
FIGURES_DIR = os.path.join(WORK_DIR, 'figures')

PYTHON = '/home/wainberg/bin/python3.13'
PYTHON_RAPIDS = '/home/karbabi/miniforge3/envs/rapids/bin/python'
NVIDIA_LIBS = ':'.join(sorted(set(
    os.path.dirname(p) for p in glob.glob(
        '/home/karbabi/miniforge3/envs/rapids/lib/python3.13'
        '/site-packages/nvidia/**/lib*/*.so*', recursive=True))))
RSCRIPT = '/home/wainberg/bin/Rscript-4.5.1'

DATASETS = {
    'SEAAD': os.path.join(DATA_DIR, 'SEAAD', 'SEAAD_raw.h5ad'),
    'PBMC': os.path.join(DATA_DIR, 'PBMC', 'Parse_PBMC_raw.h5ad'),
    'PANSCI': os.path.join(DATA_DIR, 'PanSci', 'PanSci_raw.h5ad'),
}

# (file, tool, task, thread_params)
SCRIPTS = [
    ('test_basic_brisc.py', 'brisc', 'basic', [-1, 1]),
    ('test_basic_scanpy.py', 'scanpy', 'basic', None),
    ('test_basic_rapids.py', 'rapids', 'basic', None),
    ('test_basic_rapids_mg.py', 'rapids_mg', 'basic', None),
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

if __name__ == '__main__':

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for script_file, tool, task, thread_params in SCRIPTS:
        if script_file.endswith('.R'):
            interpreter = RSCRIPT
        elif tool in ('rapids', 'rapids_mg'):
            interpreter = PYTHON_RAPIDS
        else:
            interpreter = PYTHON
        script_path = os.path.join(WORK_DIR, script_file)
        is_basic = task == 'basic'
        is_de = task.startswith('de')
        is_transfer = task == 'transfer'

        for d_name, d_path in DATASETS.items():

            for threads in (thread_params or [None]):
                job_parts = [(task.replace('_', f'_{tool}_', 1) if '_' in task
                              else f'{task}_{tool}'), d_name]
                if threads is not None:
                    job_parts.append(str(threads))
                job_name = '_'.join(job_parts)

                log = os.path.join(LOG_DIR, f'{job_name}.log')
                if os.path.exists(log):
                    with open(log, 'r') as f:
                        if 'Completed successfully' in f.read():
                            print(f'Skipping completed job: {job_name}')
                            continue

                cmd = [interpreter, script_path, d_name, d_path]
                if threads is not None:
                    cmd.append(str(threads))
                cmd.append(os.path.join(OUTPUT_DIR, f'{job_name}_timer.csv'))
                if is_basic:
                    cmd.append(os.path.join(
                        OUTPUT_DIR, f'{job_name}_embedding.csv'))
                    cmd.append(os.path.join(
                        OUTPUT_DIR, f'{job_name}_pcs.csv'))
                    cmd.append(os.path.join(
                        OUTPUT_DIR, f'{job_name}_neighbors.csv'))
                if is_de:
                    cmd.append(os.path.join(
                        OUTPUT_DIR, f'{job_name}_de.csv'))
                if is_transfer:
                    cmd.append(os.path.join(
                        OUTPUT_DIR, f'{job_name}_accuracy.csv'))

                is_gpu = tool in ('rapids', 'rapids_mg')
                is_multi_gpu = tool == 'rapids_mg'
                env = {'PYTHONPATH': f'{os.getcwd()}:$PYTHONPATH'}
                if is_gpu:
                    env['PIP_CONFIG_FILE'] = '/dev/null'
                    env['PYTHONPATH'] = os.getcwd()
                    env['LD_LIBRARY_PATH'] = (
                        NVIDIA_LIBS + ':$LD_LIBRARY_PATH')
                    env['CUPY_CACHE_DIR'] = '$SCRATCH/.cupy'

                job_id = run_slurm(
                    ' '.join(cmd), job_name=job_name, log_file=log,
                    # account='def-wainberg',
                    account='def-shreejoy',
                    # account='rrg-shreejoy',
                    CPUs=96 if is_multi_gpu else (24 if is_gpu else 192),
                    gpus_per_node=4 if is_multi_gpu else (1 if is_gpu else 0),
                    hours=24,
                    env=env)
