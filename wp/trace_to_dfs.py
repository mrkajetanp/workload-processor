import pandas as pd
import logging as log
import lisa
from lisa.trace import TaskID
from wp.helpers import df_add_cluster, flatten, trim_task_comm

# TODO: pull this from platform info instead
CLUSTERS = {
    'little': [0, 1, 2, 3],
    'mid': [4, 5],
    'big': [6, 7],
}

CGROUPS = ['background', 'foreground', 'system-background']


def trace_pixel6_emeter_df(trace):
    return trace.ana.pixel6.df_power_meter()


def trace_energy_estimate_df(trace):
    em = trace.plat_info['nrg-model']
    df = em.estimate_from_trace(trace).reset_index()

    for cluster, cpus in CLUSTERS.items():
        df[cluster] = df[[str(cpu) for cpu in cpus]].sum(axis=1)

    df['total'] = df[[str(x) for x in (CLUSTERS['little'] + CLUSTERS['mid'] + CLUSTERS['big'])]].sum(axis=1)
    df = df.set_index('Time')
    return df


def trace_cpu_idle_df(trace):
    df = trace.ana.idle.df_cpus_idle()
    return df_add_cluster(df)


def trace_idle_residency_time_df(trace):
    def cluster_state_residencies(cluster, cpus):
        df = trace.ana.idle.df_cluster_idle_state_residency(cpus).reset_index().sort_values(by=['idle_state'])
        df['cluster'] = cluster
        return df

    return pd.concat([
        cluster_state_residencies(cluster, cpus)
        for cluster, cpus in CLUSTERS.items()
    ])


def trace_cpu_idle_miss_df(trace):
    df = trace.df_event("cpu_idle_miss").reset_index()
    return df_add_cluster(df, cpu_col='cpu_id')


def trace_frequency_df(trace):
    df = trace.ana.frequency.df_cpus_frequency().reset_index()
    df = df_add_cluster(df)
    return df[['Time', 'frequency', 'cluster', 'cpu']]


def trace_frequency_residency_df(trace):
    try:
        def frequency_residencies(cluster, cpu):
            df = trace.ana.frequency.df_domain_frequency_residency(cpu).reset_index()
            df['cluster'] = cluster
            return df

        return pd.concat([
            frequency_residencies(cluster, cpus[0])
            for cluster, cpus in CLUSTERS.items()
        ])
    except lisa.conf.ConfigKeyError as e:
        log.error("Platform info not provided, can't compute frequency residencies.")
        raise e


def trace_overutilized_df(trace):
    df = pd.DataFrame()
    time = trace.ana.status.get_overutilized_time()
    total_time = trace.time_range
    perc = round(time / total_time * 100, 2)
    # TODO: convert to concat
    return df.append({'time': time, 'total_time': total_time, 'percentage': perc}, ignore_index=True)


def trace_sched_pelt_cfs_df(trace):
    df = trace.df_event("sched_pelt_cfs").reset_index()
    df = df_add_cluster(df)
    return df


def trace_tasks_residency_time_df(trace):
    df = trace.ana.tasks.df_tasks_total_residency().reset_index()
    df['comm'] = df['index'].map(trim_task_comm).astype(str)
    for cluster, cpus in CLUSTERS.items():
        df[cluster] = df[[float(cpu) for cpu in cpus]].sum(axis=1)
    df = df.rename(columns={col: str(col) for col in df.columns})
    df = df.rename(columns={0.0: 'cpu0', 1.0: 'cpu1', 2.0: 'cpu2', 3.0: 'cpu3',
                            4.0: 'cpu4', 5.0: 'cpu5', 6.0: 'cpu6', 7.0: 'cpu7'})
    return df


def trace_cgroup_attach_task_df(trace):
    df = trace.df_event("cgroup_attach_task").reset_index()
    return df


def trace_wakeup_latency_cgroup_df(trace):
    df_events = trace.df_event("cgroup_attach_task").reset_index()

    def task_latencies(task, cgroup):
        try:
            latencies = trace.ana.latency.df_latency_wakeup((task.pid, task.comm))
        except Exception:
            return pd.DataFrame()
        latencies['pid'] = task.pid
        latencies['comm'] = task.comm
        latencies['cgroup'] = cgroup
        return latencies

    def cgroup_latencies(df, cgroup):
        df_cgroup = df.query(f"dst_path == '/{cgroup}'").apply(lambda x: TaskID(x['pid'], x['comm']), axis=1)
        try:
            df_cgroup = df_cgroup.unique()
        except Exception:
            pass

        return pd.concat([task_latencies(task, cgroup) for task in df_cgroup])

    return pd.concat([cgroup_latencies(df_events, cgroup) for cgroup in CGROUPS]).reset_index()


def trace_tasks_residency_cgroup_df(trace):
    df_events = trace.df_event("cgroup_attach_task").reset_index()

    def cgroup_residencies(df, cgroup):
        df_cgroup_tasks = df.query(f"dst_path == '/{cgroup}'").apply(lambda x: TaskID(x['pid'], x['comm']), axis=1)
        try:
            df_cgroup_tasks = df_cgroup_tasks.unique()
        except Exception:
            pass

        df_residencies = trace.ana.tasks.df_tasks_total_residency(list(df_cgroup_tasks))
        df_residencies['cgroup'] = cgroup
        return df_residencies

    df = pd.concat([cgroup_residencies(df_events, cgroup) for cgroup in CGROUPS]).reset_index()
    df['comm'] = df['index'].map(trim_task_comm).astype(str)
    for cluster, cpus in CLUSTERS.items():
        df[cluster] = df[[float(cpu) for cpu in cpus]].sum(axis=1)
    df = df.rename(columns={col: str(col) for col in df.columns})
    df = df.groupby(["comm", "cgroup"]).sum().sort_values(by='Total', ascending=False).reset_index()
    return df


def trace_task_wakeup_latency_df(trace, tasks):
    def task_latency(pid, comm):
        df = trace.ana.latency.df_latency_wakeup((pid, comm))
        df['pid'] = pid
        df['comm'] = comm
        return df

    return pd.concat([task_latency(pid, comm) for pid, comm in flatten(tasks)]).reset_index()


def trace_wakeup_latency_drarm_df(trace):
    tasks = [
        trace.get_task_ids('UnityMain'),
        trace.get_task_ids('UnityGfxDeviceW'),
        trace.get_task_ids('Thread-7'),
        trace.get_task_ids('Thread-5'),
        trace.get_task_ids('Thread-6'),
        trace.get_task_ids('Thread-4'),
        trace.get_task_ids('surfaceflinger'),
        trace.get_task_ids('mali-cmar-backe'),
        trace.get_task_ids('mali_jd_thread'),
        trace.get_task_ids('writer'),
        trace.get_task_ids('FastMixer'),
        trace.get_task_ids('RenderEngine'),
        trace.get_task_ids('Audio Mixer Thr'),
        trace.get_task_ids('UnityChoreograp'),
    ] + [
        trace.get_task_ids(task)
        for task in flatten(trace.get_tasks().values())
        if 'HwBinder' in task
    ]

    return trace_task_wakeup_latency_df(trace, tasks)


def trace_wakeup_latency_jankbench_df(trace):
    tasks = [
        trace.get_task_ids('RenderThread'),
        trace.get_task_ids('droid.benchmark'),
        trace.get_task_ids('surfaceflinger'),
        trace.get_task_ids('decon0_kthread'),
    ]

    return trace_task_wakeup_latency_df(trace, tasks)


def trace_wakeup_latency_geekbench_df(trace):
    tasks = [
        trace.get_task_ids('AsyncTask #1'),
        trace.get_task_ids('labs.geekbench5'),
        trace.get_task_ids('surfaceflinger'),
    ]

    return trace_task_wakeup_latency_df(trace, tasks)


def trace_wakeup_latency_speedometer_df(trace):
    tasks = [
        trace.get_task_ids('CrRendererMain'),
        trace.get_task_ids('ThreadPoolForeg'),
        trace.get_task_ids('.android.chrome'),
        trace.get_task_ids('CrGpuMain'),
        trace.get_task_ids('Compositor'),
        trace.get_task_ids('Chrome_IOThread'),
        trace.get_task_ids('surfaceflinger'),
        trace.get_task_ids('RenderThread'),
    ]

    return trace_task_wakeup_latency_df(trace, tasks)


def trace_task_activations_df(trace, tasks):
    def task_activations(pid, comm):
        df = trace.ana.tasks.df_task_activation((pid, comm))
        df['pid'] = pid
        df['comm'] = comm
        return df

    return pd.concat([task_activations(pid, comm) for pid, comm in flatten(tasks)]).reset_index()


def trace_tasks_activations_drarm_df(trace):
    tasks = [
        trace.get_task_ids('UnityMain'),
        trace.get_task_ids('UnityGfxDeviceW'),
        trace.get_task_ids('Thread-7'),
        trace.get_task_ids('Thread-5'),
        trace.get_task_ids('Thread-6'),
        trace.get_task_ids('Thread-4'),
        trace.get_task_ids('surfaceflinger'),
        trace.get_task_ids('mali-cmar-backe'),
        trace.get_task_ids('mali_jd_thread'),
        trace.get_task_ids('writer'),
        trace.get_task_ids('FastMixer'),
        trace.get_task_ids('RenderEngine'),
        trace.get_task_ids('Audio Mixer Thr'),
        trace.get_task_ids('UnityChoreograp'),
    ] + [
        trace.get_task_ids(task)
        for task in flatten(trace.get_tasks().values())
        if 'HwBinder' in task
    ]

    return trace_task_activations_df(trace, tasks)


def trace_tasks_activations_jankbench_df(trace):
    tasks = [
        trace.get_task_ids('RenderThread'),
        trace.get_task_ids('droid.benchmark'),
        trace.get_task_ids('surfaceflinger'),
        trace.get_task_ids('decon0_kthread'),
    ]

    return trace_task_activations_df(trace, tasks)


def trace_tasks_activations_geekbench_df(trace):
    tasks = [
        trace.get_task_ids('AsyncTask #1'),
        trace.get_task_ids('labs.geekbench5'),
        trace.get_task_ids('surfaceflinger'),
    ]

    return trace_task_activations_df(trace, tasks)


def trace_tasks_activations_speedometer_df(trace):
    tasks = [
        trace.get_task_ids('CrRendererMain'),
        trace.get_task_ids('ThreadPoolForeg'),
        trace.get_task_ids('.android.chrome'),
        trace.get_task_ids('CrGpuMain'),
        trace.get_task_ids('Compositor'),
        trace.get_task_ids('Chrome_IOThread'),
        trace.get_task_ids('surfaceflinger'),
        trace.get_task_ids('RenderThread'),
    ]

    return trace_task_activations_df(trace, tasks)


def trace_uclamp_df(trace):
    return trace.df_event('uclamp_update_tsk').reset_index()
