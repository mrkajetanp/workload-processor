import os
import subprocess
import logging as log
import pandas as pd

from lisa.platforms.platinfo import PlatformInfo
from lisa.trace import MissingTraceEventError
from devlib.exception import HostError

from wp import trace_to_dfs as tdfs
from wp.helpers import wa_output_to_mock_traces, traces_analysis, df_add_wa_output_tags, df_iterations_mean
from wp.helpers import df_sort_by_clusters
from wp.constants import FULL_METRICS


class WorkloadProcessor:
    def __init__(self, wa_output, init=False, plat_info_path=None):
        self.wa_output = wa_output

        # Initialise traces
        if init or self.needs_init():
            log.info('Parsing and initialising traces..')
            self.init_traces()

        # create a directory for the analysis
        self.analysis_path = os.path.join(wa_output.path, 'analysis')
        if not os.path.exists(self.analysis_path):
            log.debug('analysis directory not found, creating..')
            os.mkdir(self.analysis_path)
        else:
            log.debug('analysis directory exists, files might be overwritten')

        self.plat_info = None
        if plat_info_path is not None:
            self.plat_info = PlatformInfo.from_yaml_map(plat_info_path)
        self.traces = wa_output_to_mock_traces(wa_output, self.plat_info)

    def run_metrics(self, metrics):
        METRIC_TO_ANALYSIS = {
            'power': self.trace_pixel6_emeter_analysis,
            'idle': self.trace_cpu_idle_analysis,
            'idle_miss': self.trace_cpu_idle_miss_analysis,
            'freq': self.trace_frequency_analysis,
            'overutil': self.trace_overutilized_analysis,
            'pelt': self.trace_sched_pelt_cfs_analysis,
            'tasks-residency': self.trace_tasks_residency_time_analysis,
            'tasks-activations': self.trace_tasks_activations_analysis,
            'adpf': self.adpf_analysis,
            'thermal': self.thermal_analysis,
            'uclamp': self.trace_uclamp_analysis,
            'energy-estimate': self.trace_energy_estimate_analysis,
            'cgroup-attach': self.trace_cgroup_attach_task_analysis,
            'wakeup-latency': self.trace_wakeup_latency_analysis,
            'wakeup-latency-cgroup': self.trace_wakeup_latency_cgroup_analysis,
            'tasks-residency-cgroup': self.trace_tasks_residency_cgroup_analysis,
        }

        for metric in metrics:
            try:
                METRIC_TO_ANALYSIS[metric]()
            except (MissingTraceEventError, HostError) as e:
                log.error(e)

    # Parse and initialise the traces
    def init_traces(self):
        start_path = os.getcwd()

        trace_paths = [os.path.abspath(job.get_artifact_path('trace-cmd-bin'))
                       for job in self.wa_output.jobs]

        for trace_path in trace_paths:
            trace_dir = os.path.dirname(trace_path)
            trace_pq_path = os.path.join(trace_dir, 'trace-events')
            if not os.path.exists(trace_pq_path):
                log.info(f"trace-events does not exist in {trace_dir}, creating..")
                os.mkdir(trace_pq_path)
            else:
                log.warning(f"trace-events exists in {trace_dir}, overwriting..")
            os.chdir(trace_pq_path)

            # Parse the trace
            trace_parser = subprocess.run(['trace-parquet', trace_path], capture_output=True, encoding='utf-8')
            for line in trace_parser.stdout.splitlines():
                log.info(line)
            for line in trace_parser.stderr.splitlines():
                if line.startswith('FILE WRITTEN'):
                    log.info(line)

            # Remove unnecessary pq file
            vendor_cdev_path = os.path.join(trace_pq_path, 'vendor_cdev_update.pq')
            if os.path.exists(vendor_cdev_path):
                log.debug('vendor_cdev_update.pq found in parsed events, removing..')
                os.remove(vendor_cdev_path)

        # change back to the initial path
        os.chdir(start_path)

    def needs_init(self):
        trace_events_paths = [
            os.path.join(os.path.dirname(job.get_artifact_path('trace-cmd-bin')), 'trace-events')
            for job in self.wa_output.jobs
        ]

        return not all([os.path.exists(path) for path in trace_events_paths])

    def apply_analysis(self, trace_to_df):
        return df_add_wa_output_tags(traces_analysis(self.traces, trace_to_df), self.wa_output)

    def trace_pixel6_emeter_analysis(self):
        log.info('Collecting data from pixel6_emeter')
        power = self.apply_analysis(tdfs.trace_pixel6_emeter_df)
        power.to_parquet(os.path.join(self.analysis_path, 'pixel6_emeter.pqt'))
        print(power)
        power_mean = df_iterations_mean(power, other_cols=['channel'])
        power_mean.to_parquet(os.path.join(self.analysis_path, 'pixel6_emeter_mean.pqt'))
        print(power_mean)

    def trace_energy_estimate_analysis(self):
        log.info('Computing energy estimates')
        df = self.apply_analysis(tdfs.trace_energy_estimate_df)
        df.to_parquet(os.path.join(self.analysis_path, 'energy_estimate.pqt'))
        print(df)

        df = df_iterations_mean(df)
        df['metric'] = 'energy-estimate'
        df.to_parquet(os.path.join(self.analysis_path, 'energy_estimate_mean.pqt'))
        print(df)

    def trace_cpu_idle_analysis(self):
        log.info('Collecting cpu_idle events')
        idle = self.apply_analysis(tdfs.trace_cpu_idle_df)
        idle.to_parquet(os.path.join(self.analysis_path, 'cpu_idle.pqt'))
        print(idle)

        log.info('Computing idle residencies')
        idle_res = self.apply_analysis(tdfs.trace_idle_residency_time_df)
        idle_res.to_parquet(os.path.join(self.analysis_path, 'idle_residency.pqt'))
        print(idle_res)

    def trace_cpu_idle_miss_analysis(self):
        log.info('Collecting cpu_idle_miss events')
        idle_miss = self.apply_analysis(tdfs.trace_cpu_idle_miss_df)

        idle_miss.to_parquet(os.path.join(self.analysis_path, 'cpu_idle_miss.pqt'))
        print(idle_miss)

        idle_miss = idle_miss.groupby(['wa_path', 'kernel', 'iteration', 'cluster', 'below'], as_index=False).size()
        idle_miss = df_sort_by_clusters(idle_miss, value_cols=['below', 'size']).rename(
            columns={'size': 'count'}
        )

        idle_miss.to_parquet(os.path.join(self.analysis_path, 'cpu_idle_miss_counts.pqt'))
        print(idle_miss)

    def trace_frequency_analysis(self):
        log.info('Collecting frequency data')
        freq = self.apply_analysis(tdfs.trace_frequency_df)
        freq.to_parquet(os.path.join(self.analysis_path, 'freqs.pqt'))
        print(freq)

        freq_mean = df_sort_by_clusters(df_iterations_mean(freq, other_cols=['cluster']), value_cols=['frequency'])
        freq_mean['frequency'] = freq_mean['frequency'] / 1000
        freq_mean.to_parquet(os.path.join(self.analysis_path, 'freqs_mean.pqt'))
        print(freq_mean)

        log.info('Computing frequency residency')
        freq_res = self.apply_analysis(tdfs.trace_frequency_residency_df)
        freq_mean.to_parquet(os.path.join(self.analysis_path, 'freqs_residency.pqt'))
        print(freq_res)

    def trace_overutilized_analysis(self):
        log.info('Collecting overutilized data')
        overutil = self.apply_analysis(tdfs.trace_overutilized_df).reset_index(drop=True)
        overutil.to_parquet(os.path.join(self.analysis_path, 'overutilized.pqt'))
        print(overutil)

        overutil = overutil.groupby(['wa_path']).mean().reset_index()
        overutil['metric'] = 'overutilized'
        overutil = overutil[['metric', 'wa_path', 'time', 'total_time', 'percentage']]
        overutil['percentage'] = round(overutil['percentage'], 2)
        overutil['time'] = round(overutil['time'], 2)
        overutil['total_time'] = round(overutil['total_time'], 2)

        overutil.to_parquet(os.path.join(self.analysis_path, 'overutilized_mean.pqt'))
        print(overutil)

    def trace_sched_pelt_cfs_analysis(self):
        log.info('Collecting sched_pelt_cfs data')
        pelt = self.apply_analysis(tdfs.trace_sched_pelt_cfs_df)
        pelt.to_parquet(os.path.join(self.analysis_path, 'sched_pelt_cfs.pqt'))
        print(pelt)

        pelt = pelt.query("path == '/'")[['cluster', 'load', 'util', 'iteration', 'wa_path', 'kernel']]
        pelt = df_iterations_mean(pelt, other_cols=['cluster']).sort_values(by=['wa_path', 'iteration'])
        pelt.to_parquet(os.path.join(self.analysis_path, 'sched_pelt_cfs_mean.pqt'))
        print(pelt)

    def trace_tasks_residency_time_analysis(self):
        log.info('Collecting task residency data')
        tasks = self.apply_analysis(tdfs.trace_tasks_residency_time_df)
        tasks.to_parquet(os.path.join(self.analysis_path, 'tasks_residency.pqt'))
        print(tasks)

        tasks = tasks.groupby(['wa_path', 'kernel', 'iteration', "comm"]).sum().sort_values(
            by='Total', ascending=False
        ).reset_index()
        tasks.to_parquet(os.path.join(self.analysis_path, 'tasks_residency_total.pqt'))
        print(tasks)

        tasks = tasks.query("not comm.str.startswith('swapper')")
        tasks = tasks.groupby(['wa_path', 'kernel', 'iteration']).sum().reset_index()
        tasks.to_parquet(os.path.join(self.analysis_path, 'tasks_residency_cpu_total.pqt'))
        print(tasks)

    def adpf_analysis(self):
        log.info('Collecting ADPF report data')

        def report_to_df(path, iteration):
            df = pd.read_csv(path).rename(
                columns={'time since start': 'time'}
            ).set_index('time')

            df['iteration'] = iteration
            return df

        reports = [
            report_to_df(job.get_artifact_path('adpf'), job.iteration)
            for job in self.wa_output.jobs
        ]

        reports = df_add_wa_output_tags(pd.concat(reports), self.wa_output)
        reports.to_parquet(os.path.join(self.analysis_path, 'adpf.pqt'))
        print(reports)

        aggregates = reports[['# frame count', 'average fps', 'iteration', 'kernel', 'wa_path']]
        report_aggs = df_iterations_mean(aggregates)[['average fps', 'iteration', 'kernel', 'wa_path']]
        report_aggs['frame count'] = aggregates.groupby([
            'wa_path', 'kernel', 'iteration'
        ]).sum().reset_index()['# frame count']
        report_aggs = report_aggs[['average fps', 'frame count', 'iteration', 'kernel', 'wa_path']]

        report_aggs.to_parquet(os.path.join(self.analysis_path, 'adpf_totals.pqt'))
        print(report_aggs)

    def thermal_analysis(self):
        log.info('Collecting thermal data')

        def process_thermal(df, iteration):
            df['iteration'] = iteration
            return df

        thermals = df_add_wa_output_tags(pd.concat([
            process_thermal(pd.read_csv(job.get_artifact_path('poller-output')), job.iteration)
            for job in self.wa_output.jobs
        ]), self.wa_output).sort_values(by=['iteration']).reset_index(drop=True).rename(
            columns={"thermal_zone0-temp": "big",
                     "thermal_zone1-temp": "mid",
                     "thermal_zone2-temp": "little",
                     "time": "Time"}
        ).set_index('Time')

        thermals.to_parquet(os.path.join(self.analysis_path, 'thermal.pqt'))
        print(thermals)

    def trace_cgroup_attach_task_analysis(self):
        log.info('Collecting cgroup_attach_task events')
        df = self.apply_analysis(tdfs.trace_cgroup_attach_task_df)
        df.to_parquet(os.path.join(self.analysis_path, 'cgroup_attach_task.pqt'))
        print(df)

    def trace_wakeup_latency_analysis(self):
        log.info('Collecting task wakeup latencies')

        label_to_analysis = {
            'jankbench': tdfs.trace_wakeup_latency_jankbench_df,
            'drarm': tdfs.trace_wakeup_latency_drarm_df,
            'geekbench': tdfs.trace_wakeup_latency_geekbench_df,
            'speedometer': tdfs.trace_wakeup_latency_speedometer_df,
        }

        label = self.wa_output.jobs[0].label
        if label not in label_to_analysis:
            log.error(f'Workload {label} does not yet support task wakeup latency analysis')
            return
        df = self.apply_analysis(label_to_analysis[label])

        df.to_parquet(os.path.join(self.analysis_path, 'wakeup_latency.pqt'))
        print(df)

        df = df.query("not comm.str.startswith('Hw')")
        df = df_iterations_mean(df, other_cols=['comm'])
        df = df[['wa_path', 'iteration', 'comm', 'wakeup_latency']]
        df = df.sort_values(by=['iteration', 'wa_path', 'comm'], ascending=[True, True, True])
        df.to_parquet(os.path.join(self.analysis_path, 'wakeup_latency_mean.pqt'))
        print(df)

    def trace_tasks_activations_analysis(self):
        log.info('Collecting task activations')

        label_to_analysis = {
            'jankbench': tdfs.trace_tasks_activations_jankbench_df,
            'drarm': tdfs.trace_tasks_activations_drarm_df,
            'geekbench': tdfs.trace_tasks_activations_geekbench_df,
            'speedometer': tdfs.trace_tasks_activations_speedometer_df,
        }

        label = self.wa_output.jobs[0].label
        if label not in label_to_analysis:
            log.error(f'Workload {label} does not yet support task activation analysis')
            return
        df = self.apply_analysis(label_to_analysis[label])

        df.to_parquet(os.path.join(self.analysis_path, 'task_activations.pqt'))
        print(df)

    def trace_wakeup_latency_cgroup_analysis(self):
        log.info('Collecting per-cgroup task wakeup latency')
        df = self.apply_analysis(tdfs.trace_wakeup_latency_cgroup_df)
        df.to_parquet(os.path.join(self.analysis_path, 'wakeup_latency_cgroup.pqt'))
        print(df)

        df = df_iterations_mean(df, other_cols=['cgroup'])
        df = df[['wa_path', 'iteration', 'cgroup', 'wakeup_latency']]
        df = df.sort_values(by=['iteration', 'wa_path', 'cgroup'], ascending=[True, True, True])

        df.to_parquet(os.path.join(self.analysis_path, 'wakeup_latency_cgroup_mean.pqt'))
        print(df)

    def trace_tasks_residency_cgroup_analysis(self):
        log.info('Collecting per-cgroup tasks residency')
        df = self.apply_analysis(tdfs.trace_tasks_residency_cgroup_df)
        df.to_parquet(os.path.join(self.analysis_path, 'tasks_residency_cgroup.pqt'))
        print(df)

        df = df.groupby(["wa_path", "cgroup", "iteration"]).sum().reset_index().sort_values(
            by=['iteration', 'wa_path', 'cgroup', 'Total'], ascending=[True, True, True, False]
        )

        df.to_parquet(os.path.join(self.analysis_path, 'tasks_residency_cgroup_total.pqt'))
        print(df)

    def trace_uclamp_analysis(self):
        log.info('Collecting uclamp data')
        df = self.apply_analysis(tdfs.trace_uclamp_df)

        df.to_parquet(os.path.join(self.analysis_path, 'uclamp_updates.pqt'))
        print(df)
