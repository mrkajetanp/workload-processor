from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent.joinpath('config.yaml')
AGENDAS_PATH = Path(__file__).resolve().parent.parent.joinpath('agendas')

FULL_METRICS = [
    'power', 'idle', 'idle-miss', 'freq', 'overutil', 'pelt', 'capacity',
    'uclamp', 'adpf', 'thermal', 'perf-trace-event', 'wakeup-latency',
    'tasks-residency', 'tasks-activations',
    'cgroup-attach', 'wakeup-latency-cgroup', 'tasks-residency-cgroup',
    'energy-estimate',
]

SUPPORTED_WORKLOADS = ['drarm', 'geekbench', 'jankbench', 'speedometer']

DEVICE_COMMANDS = ['status', 'disable-cpusets', 'disable-cpushares', 'menu', 'teo', 'latency-sensitive', 'powersave',
                   'performance', 'schedutil', 'sugov-rate-limit']
