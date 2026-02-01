import os

LOG_DIR = os.path.join('sc-benchmarking', 'logs')

failed_jobs = []
for log_file in os.listdir(LOG_DIR):
    if log_file.endswith('.log'):
        with open(os.path.join(LOG_DIR, log_file), 'r') as f:
            content = f.read()
        if content.strip() and 'Completed successfully' not in content:
            failed_jobs.append(log_file[:-4])

if failed_jobs:
    print(f'Failed jobs ({len(failed_jobs)}):')
    for job in sorted(failed_jobs):
        print(f'  {job}')
else:
    print('No failed jobs found.')
