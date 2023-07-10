import polars as pl
import logging as log
import confuse

import lisa
from lisa.trace import TaskID

from wp.constants import APP_NAME
from wp.helpers import df_add_cluster, flatten, trim_task_comm, df_add_wa_output_tags

CONFIG = confuse.Configuration(APP_NAME, __name__)

CLUSTERS = CONFIG['target']['clusters'].get()
CGROUPS = CONFIG['target']['cgroups'].get()


class WorkloadAnalysisRunner:
    def __init__(self, processor):
        self.processor = processor

    def apply(self, trace_to_df):
        def analyse_iteration(trace, iteration):
            log.debug(f'Processing iteration {iteration} from {trace.trace_path} with {trace_to_df.__name__}')
            try:
                return trace_to_df(trace).with_columns(pl.lit(iteration).alias('iteration'))
            except Exception as e:
                log.error(f'Processing iteration {iteration} failed with {e}')
                if self.processor.allow_missing:
                    return None
                raise e

        log.debug(f'Applying analysis {trace_to_df.__name__}')
        dfs = [analyse_iteration(trace, iteration) for iteration, trace in self.processor.traces.items()]
        analysis_df = pl.concat([df for df in dfs if df is not None])
        return df_add_wa_output_tags(analysis_df, self.processor.wa_output)

    def trace_cpu_idle_df(self, trace):
        df = pl.from_pandas(trace.ana.idle.df_cpus_idle().reset_index())
        return df_add_cluster(df)

    def trace_idle_residency_time_df(self, trace):
        def cluster_state_residencies(cluster, cpus):
            df = pl.from_pandas(trace.ana.idle.df_cluster_idle_state_residency(cpus).reset_index()).sort('idle_state')
            return df.with_columns(pl.lit(cluster).alias('cluster'))

        return pl.concat([
            cluster_state_residencies(cluster, cpus)
            for cluster, cpus in CLUSTERS.items()
        ])

    def trace_cpu_idle_miss_df(self, trace):
        df = pl.from_pandas(trace.df_event("cpu_idle_miss").reset_index())
        return df_add_cluster(df, cpu_col='cpu_id')


def trace_pixel6_emeter_df(trace):
    return pl.from_pandas(trace.ana.pixel6.df_power_meter().reset_index())


def trace_energy_estimate_df(trace):
    em = trace.plat_info['nrg-model']
    df = pl.from_pandas(em.estimate_from_trace(trace).reset_index())

    return df.with_columns([
        pl.sum(pl.col([str(cpu) for cpu in cpus])).alias(cluster) for cluster, cpus in CLUSTERS.items()
    ]).with_columns([
        (pl.sum(pl.col([str(x) for x in (CLUSTERS['little'] + CLUSTERS['mid'] + CLUSTERS['big'])]))).alias('total')
    ])


def trace_frequency_df(trace):
    df = pl.from_pandas(trace.ana.frequency.df_cpus_frequency().reset_index())
    df = df_add_cluster(df)
    return df[['Time', 'frequency', 'cluster', 'cpu']]


def trace_frequency_residency_df(trace):
    try:
        def frequency_residencies(cluster, cpu):
            try:
                df = pl.from_pandas(trace.ana.frequency.df_domain_frequency_residency(cpu).reset_index())
                return df.with_columns(pl.lit(cluster).alias('cluster'))
            except ValueError:
                return None

        freqs = [frequency_residencies(cluster, cpus[0]) for cluster, cpus in CLUSTERS.items()]
        return pl.concat([df for df in freqs if df is not None])
    except lisa.conf.ConfigKeyError as e:
        log.error("Platform info not provided, can't compute frequency residencies.")
        raise e


def trace_overutilized_df(trace):
    time = trace.ana.status.get_overutilized_time()
    total_time = trace.time_range
    perc = round(time / total_time * 100, 2)
    return pl.DataFrame({'time': time, 'total_time': total_time, 'percentage': perc})


def trace_sched_pelt_cfs_df(trace):
    df = pl.from_pandas(trace.df_event("sched_pelt_cfs").reset_index())
    df = df_add_cluster(df)
    return df


def trace_tasks_residency_time_df(trace):
    df = pl.from_pandas(trace.ana.tasks.df_tasks_total_residency().reset_index())
    df = df.with_columns(
        [pl.col('index').apply(trim_task_comm).alias('comm')] + [
            pl.sum(pl.col([str(float(cpu)) for cpu in cpus])).alias(cluster) for cluster, cpus in CLUSTERS.items()
        ]
    )
    cpu_count = sum([len(lst) for lst in CONFIG['target']['clusters'].get().values()])
    return df.rename({**{col: str(col) for col in df.columns},
                      **{str(float(i)): "cpu"+str(i) for i in range(cpu_count)}})


def trace_task_wakeup_latency_df(trace, tasks):
    tasks = [trace.get_task_ids(task) for task in tasks]

    def task_latency(pid, comm):
        try:
            return pl.from_pandas(trace.ana.latency.df_latency_wakeup((pid, comm)).reset_index()).with_columns(
                pl.lit(pid).alias('pid'),
                pl.lit(comm).alias('comm'),
            )
        except ValueError:
            return pl.DataFrame()

    return pl.concat([task_latency(pid, comm) for pid, comm in flatten(tasks)])


def trace_wakeup_latency_drarm_df(trace):
    config = confuse.Configuration(APP_NAME, __name__)
    tasks = config['processor']['important_tasks']['drarm'].get()
    return trace_task_wakeup_latency_df(trace, tasks)


def trace_wakeup_latency_fortnite_df(trace):
    config = confuse.Configuration(APP_NAME, __name__)
    tasks = config['processor']['important_tasks']['fortnite'].get()
    return trace_task_wakeup_latency_df(trace, tasks)


def trace_wakeup_latency_jankbench_df(trace):
    config = confuse.Configuration(APP_NAME, __name__)
    tasks = config['processor']['important_tasks']['jankbench'].get()
    return trace_task_wakeup_latency_df(trace, tasks)


def trace_wakeup_latency_geekbench_df(trace):
    config = confuse.Configuration(APP_NAME, __name__)
    tasks = config['processor']['important_tasks']['geekbench'].get()
    return trace_task_wakeup_latency_df(trace, tasks)


def trace_wakeup_latency_speedometer_df(trace):
    config = confuse.Configuration(APP_NAME, __name__)
    tasks = config['processor']['important_tasks']['speedometer'].get()
    return trace_task_wakeup_latency_df(trace, tasks)


def trace_task_activations_df(trace, tasks):
    tasks = [trace.get_task_ids(task) for task in tasks]
    return pl.concat([
        pl.from_pandas(trace.ana.tasks.df_task_activation((pid, comm))).with_columns(
            pl.col('active', 'duration', 'duty_cycle').cast(pl.Float64),
            pl.lit(pid).alias('pid'),
            pl.lit(comm).alias('comm'),
        )
        for pid, comm in flatten(tasks)
    ])


def trace_tasks_activations_drarm_df(trace):
    config = confuse.Configuration(APP_NAME, __name__)

    tasks = config['processor']['important_tasks']['drarm'].get() + [
        task for task in flatten(trace.get_tasks().values()) if 'HwBinder' in task
    ]

    return trace_task_activations_df(trace, tasks)


def trace_tasks_activations_fortnite_df(trace):
    config = confuse.Configuration(APP_NAME, __name__)
    tasks = config['processor']['important_tasks']['fortnite'].get()
    return trace_task_activations_df(trace, tasks)


def trace_tasks_activations_jankbench_df(trace):
    config = confuse.Configuration(APP_NAME, __name__)
    tasks = config['processor']['important_tasks']['jankbench'].get()
    return trace_task_activations_df(trace, tasks)


def trace_tasks_activations_geekbench_df(trace):
    config = confuse.Configuration(APP_NAME, __name__)
    tasks = config['processor']['important_tasks']['geekbench'].get()
    return trace_task_activations_df(trace, tasks)


def trace_tasks_activations_speedometer_df(trace):
    config = confuse.Configuration(APP_NAME, __name__)
    tasks = config['processor']['important_tasks']['speedometer'].get()
    return trace_task_activations_df(trace, tasks)


def trace_cgroup_attach_task_df(trace):
    df = pl.from_pandas(trace.df_event("cgroup_attach_task").reset_index())
    return df


def trace_wakeup_latency_cgroup_df(trace):
    df_events = pl.from_pandas(trace.df_event("cgroup_attach_task").reset_index())

    def task_latencies(task, cgroup):
        try:
            latencies = pl.from_pandas(trace.ana.latency.df_latency_wakeup((task.pid, task.comm)).reset_index())
        except Exception:
            return None

        return latencies.with_columns(
            pl.lit(task.pid).alias('pid'),
            pl.lit(task.comm).alias('comm'),
            pl.lit(cgroup).alias('cgroup'),
        )

    def cgroup_latencies(df, cgroup):
        tasks = [
            TaskID(row['pid'], row['comm'])
            for row in df.filter(pl.col('dst_path') == f'/{cgroup}').unique(
                subset=['pid', 'comm']
            ).iter_rows(named=True)
        ]

        tasks_latencies_list = [task_latencies(task, cgroup) for task in tasks]
        return pl.concat([df for df in tasks_latencies_list if df is not None]) if tasks_latencies_list else None

    cgroup_latencies_list = [cgroup_latencies(df_events, cgroup) for cgroup in CGROUPS]
    return pl.concat([df for df in cgroup_latencies_list if df is not None])


def trace_tasks_residency_cgroup_df(trace):
    df_events = pl.from_pandas(trace.df_event("cgroup_attach_task").reset_index())

    def cgroup_residencies(df, cgroup):
        tasks = [
            TaskID(row['pid'], row['comm'])
            for row in df.filter(pl.col('dst_path') == f'/{cgroup}').unique(
                subset=['pid', 'comm']
            ).iter_rows(named=True)
        ]

        try:
            df_res = pl.from_pandas(trace.ana.tasks.df_tasks_total_residency(tasks).reset_index())
            return df_res.with_columns(pl.lit(cgroup).alias('cgroup')) if not df_res.is_empty() else None
        except ValueError:
            return None

    residencies = [cgroup_residencies(df_events, cgroup) for cgroup in CGROUPS]

    df = pl.concat([res for res in residencies if res is not None and not res.is_empty()])
    df = df.with_columns(
        [pl.col('index').apply(trim_task_comm).alias('comm')] + [
            pl.sum(pl.col([str(float(cpu)) for cpu in cpus])).alias(cluster) for cluster, cpus in CLUSTERS.items()
        ]
    )
    df = df.groupby(["comm", "cgroup"]).sum().sort('Total', descending=True)
    return df


def trace_uclamp_df(trace):
    df = pl.from_pandas(trace.df_event('uclamp_update_tsk').reset_index())
    tasks = trace.get_tasks()
    return df.with_columns(
        pl.col('pid').apply(lambda p: " ".join(tasks.get(p, ['<unknown>']))).alias('task'),
        # Normalise the timestamps across iterations
        (pl.col('Time') - trace.start).alias('time_it'),
    )


def trace_perf_counters_df(trace):
    df = pl.from_pandas(trace.df_event('perf_counter').reset_index()[['Time', 'cpu', 'counter_id', 'value']])

    def process_counter_group_df(counter, cpu, group_df):
        counter_name = CONFIG['target']['perf_counter_ids'].get()[int(counter)]
        group_df = group_df.with_columns(pl.col('value').diff().alias(f'{counter_name}')).rename(
            {'value': f'{counter_name}-Total'}
        ).drop_nulls()
        return group_df.melt(id_vars=['Time', 'cpu'], value_vars=[f'{counter_name}-Total', f'{counter_name}'])

    result = pl.concat([
        process_counter_group_df(counter, cpu, group_df)
        for (counter, cpu), group_df in df.groupby(['counter_id', 'cpu'])
    ])

    return result.sort(['variable', 'Time'])


def trace_capacity_df(trace):
    df = pl.from_pandas(trace.df_event('sched_cpu_capacity').reset_index())
    return df.with_columns(
        (pl.col('Time') - trace.start).alias('time_it'),
    )
