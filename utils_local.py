import gc
import io
import os
import time
import socket
import subprocess
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from timeit import default_timer
from contextlib import contextmanager

_MONITOR_MEM_SH_PATH = os.path.join(
    os.path.dirname(__file__), 'monitor_mem.sh')

class MemoryTimer:
    
    def __init__(self, silent=True):
        self.timings = {}
        self.silent = silent

    def __call__(self, message):
        start = default_timer()
        pid = os.getpid()

        @contextmanager
        def timer():
            if not self.silent:
                print(f'{message}...')
                
            # Start memory monitor with fast sampling
            monitor = subprocess.Popen(
                [_MONITOR_MEM_SH_PATH, '-p', str(pid), '-i', '0.1'],
                stdout=subprocess.PIPE, text=True
            )
            
            time.sleep(0.2)  # Startup delay
            try:
                yield
                aborted = False
            except Exception as e:
                aborted = True
                raise e
            finally:
                duration = default_timer() - start
                
                # Stop monitor and get output
                subprocess.run(['kill', str(monitor.pid)])
                output, _ = monitor.communicate()
                
                # Parse memory readings (memory_kb, percent)
                try:
                    data = np.loadtxt(io.StringIO(output), delimiter=',')
                except (ValueError, UserWarning) as e:
                    # Handle empty output - use minimal defaults
                    data = np.array([[0.0, 0.0]])
                
                if data.size == 0:
                    # Another safeguard for empty data
                    data = np.array([[0.0, 0.0]])
                elif data.ndim == 1:
                    data = data[None, :]
                    
                max_mem = np.max(data, axis=0)
                
                memory_gb = np.round(max_mem[0] / 1024 / 1024, 2)
                percent_mem = np.round(max_mem[1], 1)
                
                if not self.silent:
                    status = 'aborted after' if aborted else 'took'
                    time_str = self._format_time(duration)
                    print(f'{message} {status} {time_str} using '
                          f'{memory_gb} GiB\n')
                
                # Update or create timing entry
                if message in self.timings:
                    self.timings[message]['duration'] += duration
                    self.timings[message]['memory'] = max(
                        self.timings[message]['memory'], memory_gb
                    )
                    self.timings[message]['%mem'] = max(
                        self.timings[message]['%mem'], percent_mem
                    )
                    self.timings[message]['aborted'] = (
                        self.timings[message]['aborted'] or aborted
                    )
                else:
                    self.timings[message] = {
                        'duration': duration,
                        'memory': memory_gb,
                        '%mem': percent_mem,
                        'aborted': aborted,
                    }
                gc.collect()

        return timer()

    def print_summary(self, sort=True, unit=None):
        print('\n--- Timing Summary ---')
        items = (sorted(self.timings.items(), 
                       key=lambda x: x[1]['duration'], reverse=True)
                 if sort else list(self.timings.items()))
        
        total_time = sum(info['duration'] for _, info in items)
        
        for message, info in items:
            duration = info['duration']
            memory = info['memory']
            pct = (duration / total_time * 100) if total_time else 0
            status = 'aborted after' if info['aborted'] else 'took'
            time_str = self._format_time(duration, unit)
            print(f'{message} {status} {time_str} ({pct:.1f}%) '
                  f'using {memory} GiB ({info["%mem"]}%)')
        
        print(f'\nTotal time: {self._format_time(total_time, unit)}')

    def _format_time(self, duration, unit=None):
        if unit:
            # Fixed unit conversion
            conversions = {
                's': 1, 'ms': 1000, 'us': 1e6, 'µs': 1e6,
                'ns': 1e9, 'm': 1/60, 'h': 1/3600, 'd': 1/86400
            }
            if unit not in conversions:
                raise ValueError(f'Unsupported unit: {unit}')
            return f'{duration * conversions[unit]}{unit}'
        
        # Auto-format with appropriate units
        units = [
            (86400, 'd'), (3600, 'h'), (60, 'm'), (1, 's'),
            (0.001, 'ms'), (1e-6, 'µs'), (1e-9, 'ns')
        ]
        
        parts = []
        for threshold, suffix in units:
            if duration >= threshold or (not parts and suffix == 'ns'):
                if threshold >= 1:
                    value = int(duration // threshold)
                    duration %= threshold
                else:
                    value = int((duration / threshold) % 1000)
                if value > 0 or (not parts and suffix == 'ns'):
                    parts.append(f'{value}{suffix}')
                if len(parts) == 2:
                    break
        
        return ' '.join(parts) if parts else 'less than 1ns'

    def to_dataframe(self, sort=True, unit=None):
        if not self.timings:
            return pl.DataFrame({
                'operation': [], 'duration': [], 'duration_unit': [],
                'aborted': [], 'percentage': [], 'memory': [],
                'memory_unit': [], 'percent_mem': []
            })
        
        items = (sorted(self.timings.items(),
                       key=lambda x: x[1]['duration'], reverse=True)
                 if sort else list(self.timings.items()))
        
        total = sum(info['duration'] for info in self.timings.values())
        
        # Unit conversion
        conv = 1.0
        if unit:
            conversions = {
                's': 1, 'ms': 1000, 'us': 1e6, 'µs': 1e6,
                'ns': 1e9, 'm': 1/60, 'h': 1/3600, 'd': 1/86400
            }
            if unit not in conversions:
                raise ValueError(f'Unsupported unit: {unit}')
            conv = conversions[unit]
        
        ops, durs, aborts, pcts = [], [], [], []
        memory, memory_unit, percent_mem = [], [], []
        
        for msg, info in items:
            ops.append(msg)
            durs.append(info['duration'] * conv if unit else info['duration'])
            aborts.append(info['aborted'])
            pcts.append((info['duration'] / total * 100) if total else 0)
            memory.append(info['memory'])
            memory_unit.append('GiB')
            percent_mem.append(info['%mem'])
        
        return pl.DataFrame({
            'operation': ops,
            'duration': durs,
            'duration_unit': [unit or 's'] * len(ops),
            'aborted': aborts,
            'percentage': pcts,
            'memory': memory,
            'memory_unit': memory_unit,
            'percent_mem': percent_mem,
        })

def system_info():
    hostname = socket.gethostname()
    user = os.environ.get('USER', 'N/A')
    cpu_cores = (os.environ.get('SLURM_CPUS_PER_TASK') or 
                 os.environ.get('SLURM_CPUS_ON_NODE'))
    
    mem_gb = 'N/A'
    try:
        mem_mb = int(os.environ['SLURM_MEM_PER_NODE'])
        mem_gb = f'{mem_mb / 1024:.1f} GB'
    except (KeyError, ValueError):
        pass
    
    print(f'\n--- User Resource Allocation ---')
    print(f'Node: {hostname}')
    print(f'User: {user}')
    print(f'CPU Cores Allocated: {cpu_cores}')
    print(f'Memory Allocated: {mem_gb}')

def confusion_matrix_plot(sc_obs, orig_col, trans_col):
    cm = pd.crosstab(
        sc_obs[orig_col].to_pandas(),
        sc_obs[trans_col].to_pandas()
    )
    cm.to_csv('cell_type_confusion.csv')
    
    # Normalize by row
    norm_cm = cm.div(cm.sum(axis=1), axis=0).fillna(0)
    
    # Dynamic figure sizing
    height = 6 + norm_cm.shape[0] * 0.05
    width = 8 + norm_cm.shape[1] * 0.05
    
    plt.figure(figsize=(width, height))
    ax = sns.heatmap(
        norm_cm, square=False, linewidths=0.5,
        cmap='rocket_r', vmin=0, vmax=1,
        cbar_kws={'shrink': 0.7}
    )
    
    # Format colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.set_xlabel(f'Predicted: {trans_col}')
    ax.set_ylabel(f'Original: {orig_col}')
    plt.tight_layout()
    plt.savefig('cell_type_confusion.png', dpi=300)
    plt.show()
