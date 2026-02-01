import os
import sys
sys.path.append('sc-benchmarking')
from utils_local import run_slurm

DATA_DIR = 'single-cell'
WORK_DIR = 'sc-benchmarking'
OUTPUT_DIR = os.path.join(WORK_DIR, 'output')
LOG_DIR = os.path.join(WORK_DIR, 'logs')
FIGURES_DIR = os.path.join(WORK_DIR, 'figures')

DATASETS = {
    'SEAAD': os.path.join(DATA_DIR, 'SEAAD', 'SEAAD_raw.h5ad'),
    'PBMC': os.path.join(DATA_DIR, 'PBMC', 'Parse_PBMC_raw.h5ad'),
}

SCRIPTS = [
    'test_basic_brisc.py',
    'test_basic_scanpy.py',
    'test_basic_seurat.R',
    'test_de_brisc.py',
    'test_de_scanpy.py',
    'test_de_seurat_deseq.R',
    'test_de_seurat_wilcox.R',
    'test_transfer_brisc.py',
    'test_transfer_scanpy.py',
    'test_transfer_seurat.R',
]

if __name__ == '__main__':

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for script_file in SCRIPTS:
        name, ext = os.path.splitext(script_file)
        python_path = '/home/wainberg/bin/python3.13'
        rscript_path = '/home/wainberg/bin/Rscript-4.5.1'
        interpreter = (rscript_path if ext == '.R' else python_path)
        script_path = os.path.join(WORK_DIR, script_file)

        for d_name, d_path in DATASETS.items():
            param_sets = [(-1,), (1,)] if 'brisc' in name else [()]

            for params_tuple in param_sets:
                short_name = name.removeprefix('test_')
                params_str = [str(p) for p in params_tuple]
                job_name = '_'.join([short_name, d_name] + params_str)
                output_timings = os.path.join(
                    OUTPUT_DIR, f'{job_name}_timer.csv')
                output_accuracy = os.path.join(
                    OUTPUT_DIR, f'{job_name}_accuracy.csv')
                log = os.path.join(LOG_DIR, f'{job_name}.log')

                if os.path.exists(log):
                    with open(log, 'r') as f:
                        if 'Completed successfully' in f.read():
                            print(f'Skipping completed job: {job_name}')
                            continue

                cmd = [interpreter, script_path, d_name, d_path]
                cmd.extend([str(p) for p in params_tuple])
                cmd.append(output_timings)
                if 'transfer' in name:
                    cmd.append(output_accuracy)

                pythonpath = f'{os.getcwd()}:$PYTHONPATH'
                full_cmd = (
                    f'export PYTHONPATH={pythonpath} && '
                    f'{" ".join(cmd)}')
                run_slurm(
                    full_cmd, job_name=job_name, log_file=log,
                    CPUs=192,
                    hours=24 if ext == '.R' else 12)
