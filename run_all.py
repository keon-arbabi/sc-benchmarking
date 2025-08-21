import itertools
import os
import utils

DATA_DIR = 'single-cell'
WORK_DIR = 'sc-benchmarking'
OUTPUT_DIR = os.path.join(WORK_DIR, 'output')
LOG_DIR = os.path.join(WORK_DIR, 'logs')
FIGURES_DIR = os.path.join(WORK_DIR, 'figures')

DATASETS = {
    'PBMC': {
        'data': os.path.join(DATA_DIR, 'PBMC', 'Parse_PBMC_raw.h5ad'),
        'ref': os.path.join(DATA_DIR, 'PBMC', 'ScaleBio_PBMC_reference.h5ad'),
    },
    'SEAAD': {
        'data': os.path.join(DATA_DIR, 'SEAAD', 'SEAAD_raw.h5ad'),
        'ref': os.path.join(DATA_DIR, 'SEAAD', 'SEAAD_ref.h5ad'),
    },
}

SCRIPTS = [
    'test_basic_brisc.py',
    'test_basic_scanpy.py',
    'test_basic_seurat.R',
    'test_de_brisc.py',
    'test_de_scanpy.py',
    'test_de_seurat.R',
    'test_transfer_brisc.py',
    'test_transfer_scanpy.py',
    'test_transfer_seurat.R',
]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for script_file in SCRIPTS:
        name, ext = os.path.splitext(script_file)
        interpreter = 'Rscript-4.5.1' if ext == '.R' else 'python3.13'
        script_path = os.path.join(WORK_DIR, script_file)

        for d_name, d_paths in DATASETS.items():
            param_sets = [()]
            if 'brisc' in name:
                threads = [-1, 1]
                param_sets = [(t,) for t in threads]

            for params_tuple in param_sets:
                job_name_parts = [name, d_name] + [str(p) for p in params_tuple]
                job_name = '_'.join(job_name_parts)
                output = os.path.join(OUTPUT_DIR, f'{job_name}.csv')
                log = os.path.join(LOG_DIR, f'{job_name}.log')

                if os.path.exists(log):
                    with open(log, 'r') as f:
                        if 'Completed successfully' in f.read():
                            print(f'Skipping completed job: {job_name}')
                            continue

                cmd = [interpreter, script_path, d_name]

                if 'transfer' in name:
                    cmd.extend([d_paths['data'], d_paths['ref']])
                else:
                    cmd.append(d_paths['data'])

                cmd.extend([str(p) for p in params_tuple])
                cmd.append(output)

                full_cmd = f'PYTHONPATH={os.getcwd()} {" ".join(cmd)}'
                utils.run_slurm(
                    full_cmd, job_name=job_name, log_file=log,
                    CPUs=192, 
                    hours=40 if interpreter == 'Rscript' else 5)

if __name__ == '__main__':
    main()