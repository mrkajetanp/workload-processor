from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent.joinpath('config.yaml')
AGENDAS_PATH = Path(__file__).resolve().parent.parent.joinpath('agendas')

FULL_METRICS = [
    'power', 'idle', 'idle_miss', 'freq', 'overutil', 'pelt',
    'uclamp', 'adpf', 'thermal', 'wakeup-latency',
    'tasks-residency', 'tasks-activations',
    'cgroup-attach', 'wakeup-latency-cgroup', 'tasks-residency-cgroup',
    'energy-estimate',
]

SUPPORTED_WORKLOADS = ['drarm', 'geekbench', 'jankbench', 'speedometer']
