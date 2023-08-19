from pathlib import Path

APP_NAME = 'workload-processor'
AGENDAS_PATH = Path(__file__).resolve().parent.parent.joinpath('agendas')

FULL_METRICS = [
    'power', 'idle', 'idle-miss', 'freq', 'overutil', 'pelt', 'capacity',
    'uclamp', 'adpf', 'thermal', 'fps', 'wakeup-latency',
    'tasks-activations', 'perf-trace-event', 'tasks-residency',
    'cgroup-attach', 'wakeup-latency-cgroup', 'tasks-residency-cgroup',
    'energy-estimate',
]

SUPPORTED_WORKLOADS = ['geekbench', 'jankbench', 'speedometer', 'drarm', 'fortnite']

DEVICE_COMMANDS = ['status', 'disable-cpusets', 'disable-cpushares', 'menu', 'teo', 'latency-sensitive', 'powersave',
                   'performance', 'schedutil', 'sugov-rate-limit', 'reload-module']
