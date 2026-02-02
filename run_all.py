import os
import sys
sys.path.append('sc-benchmarking')
from utils_local import run_slurm

DATA_DIR = 'single-cell'
WORK_DIR = 'sc-benchmarking'
OUTPUT_DIR = os.path.join(WORK_DIR, 'output')
LOG_DIR = os.path.join(WORK_DIR, 'logs')
FIGURES_DIR = os.path.join(WORK_DIR, 'figures')

PYTHON = '/home/wainberg/bin/python3.13'
RSCRIPT = '/home/wainberg/bin/Rscript-4.5.1'

DATASETS = {
    'SEAAD': os.path.join(DATA_DIR, 'SEAAD', 'SEAAD_raw_50K.h5ad'),
    'PBMC': os.path.join(DATA_DIR, 'PBMC', 'Parse_PBMC_raw.h5ad'),
}

# (file, tool, task, thread_params)
SCRIPTS = [
    ('test_basic_brisc.py', 'brisc', 'basic', [-1, 1]),
    ('test_basic_scanpy.py', 'scanpy', 'basic', None),
    ('test_basic_seurat.R', 'seurat', 'basic', None),
    ('test_de_brisc.py', 'brisc', 'de', [-1, 1]),
    ('test_de_scanpy.py', 'scanpy', 'de', None),
    ('test_de_seurat_deseq.R', 'seurat', 'de_deseq', None),
    ('test_de_seurat_wilcox.R', 'seurat', 'de_wilcox', None),
    ('test_transfer_brisc.py', 'brisc', 'transfer', [-1, 1]),
    ('test_transfer_scanpy.py', 'scanpy', 'transfer', None),
    ('test_transfer_seurat.R', 'seurat', 'transfer', None),
]

if __name__ == '__main__':

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    basic_job_ids = {}

    for script_file, tool, task, thread_params in SCRIPTS:
        interpreter = RSCRIPT if script_file.endswith('.R') else PYTHON
        script_path = os.path.join(WORK_DIR, script_file)
        uses_doublet_cache = tool in ('scanpy', 'seurat')
        is_basic = task == 'basic'
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
                        OUTPUT_DIR, f'{tool}_{d_name}_doublets.csv'))
                if is_transfer:
                    cmd.append(os.path.join(
                        OUTPUT_DIR, f'{job_name}_accuracy.csv'))
                if uses_doublet_cache and not is_basic:
                    cmd.append(os.path.join(
                        OUTPUT_DIR, f'{tool}_{d_name}_doublets.csv'))

                pythonpath = f'{os.getcwd()}:$PYTHONPATH'
                full_cmd = (
                    f'export PYTHONPATH={pythonpath} && '
                    f'{" ".join(cmd)}')

                dependency = (basic_job_ids.get((tool, d_name))
                              if uses_doublet_cache and not is_basic
                              else None)

                job_id = run_slurm(
                    full_cmd, job_name=job_name, log_file=log,
                    # account='def-wainberg',
                    account='rrg-shreejoy',
                    CPUs=192,
                    hours=24,
                    dependency=dependency)

                if uses_doublet_cache and is_basic and job_id:
                    basic_job_ids[(tool, d_name)] = job_id
