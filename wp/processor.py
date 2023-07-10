import os
import subprocess
import shutil
import time
import confuse
import logging as log
import polars as pl

from lisa.wa import WAOutput
from lisa.platforms.platinfo import PlatformInfo
from lisa.trace import MissingTraceEventError
from devlib.exception import HostError

from wp import analysis as ana
from wp.analysis import WorkloadAnalysisRunner
from wp.constants import APP_NAME
from wp.helpers import wa_output_to_mock_traces, wa_output_to_traces
from wp.helpers import df_sort_by_clusters, df_add_wa_output_tags, df_iterations_mean
from wp.helpers import cpu_cluster


class WorkloadProcessor:
    def __init__(self, output_path, config=None):
        self.config = confuse.Configuration(APP_NAME, __name__) if config is None else config

        pl.toggle_string_cache(True)
        pl.Config.set_tbl_formatting('ASCII_MARKDOWN')
        pl.Config.set_tbl_hide_column_data_types(True)
        pl.Config.set_tbl_rows(10)

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"WA output path '{output_path}' not found.")
        self.wa_output = WAOutput(output_path)

        # create a directory for the analysis
        self.analysis_path = os.path.join(self.wa_output.path, 'analysis')
        if not os.path.exists(self.analysis_path):
            log.debug('analysis directory not found, creating..')
            os.mkdir(self.analysis_path)
        else:
            log.debug('analysis directory exists, files might be overwritten')

        self.plat_info = None
        plat_info_path = os.path.expanduser(self.config['target']['plat_info'].get(str))
        if plat_info_path is not None:
            self.plat_info = PlatformInfo.from_yaml_map(plat_info_path)

        trace_parquet_found = shutil.which('trace-parquet') is not None
        no_parser = self.config['no_parser'].get(False)
        init = self.config['init'].get(False)

        # Initialise traces
        if (trace_parquet_found and not no_parser) and (init or self.needs_init()):
            if not self.config['skip_validation'].get(False):
                self.validate_traces()
            log.info('Parsing and initialising traces..')
            self.init_traces()

        self.allow_missing = self.config['allow_missing'].get(False)

        # Trace parquet not found or fallback requested
        traces_start = time.time()
        if no_parser or not trace_parquet_found:
            self.traces = wa_output_to_traces(self.wa_output, self.plat_info)
        else:
            self.traces = wa_output_to_mock_traces(self.wa_output, self.plat_info)
        log.debug(f"trace loading complete, took {round(time.time() - traces_start, 2)}s")

        # initialise the analysis runner
        self.analysis = WorkloadAnalysisRunner(self)

    def run_metrics(self, metrics):
        METRIC_TO_ANALYSIS = {
            'power': self.trace_pixel6_emeter_analysis,
            'idle': self.trace_cpu_idle_analysis,
            'idle-miss': self.trace_cpu_idle_miss_analysis,
            'freq': self.trace_frequency_analysis,
            'overutil': self.trace_overutilized_analysis,
            'pelt': self.trace_sched_pelt_cfs_analysis,
            'capacity': self.trace_capacity_analysis,
            'tasks-residency': self.trace_tasks_residency_time_analysis,
            'tasks-activations': self.trace_tasks_activations_analysis,
            'adpf': self.adpf_analysis,
            'thermal': self.thermal_analysis,
            'perf-trace-event': self.trace_perf_event_analysis,
            'uclamp': self.trace_uclamp_analysis,
            'energy-estimate': self.trace_energy_estimate_analysis,
            'cgroup-attach': self.trace_cgroup_attach_task_analysis,
            'wakeup-latency': self.trace_wakeup_latency_analysis,
            'wakeup-latency-cgroup': self.trace_wakeup_latency_cgroup_analysis,
            'tasks-residency-cgroup': self.trace_tasks_residency_cgroup_analysis,
        }

        for metric in metrics:
            try:
                analysis_start = time.time()
                METRIC_TO_ANALYSIS[metric]()
                log.debug(f"{metric} analysis complete, took {round(time.time() - analysis_start, 2)}s")
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

    def validate_traces(self):
        log.info('Validating traces..')
        for trace in self.wa_output['trace'].traces.values():
            trace.start
        log.info('Traces validated successfully')

    def trace_pixel6_emeter_analysis(self):
        log.info('Collecting data from pixel6_emeter')
        power = self.analysis.apply(ana.trace_pixel6_emeter_df)
        power.write_parquet(os.path.join(self.analysis_path, 'pixel6_emeter.pqt'))
        print(power)
        power_mean = df_iterations_mean(power, other_cols=['channel'])
        power_mean.write_parquet(os.path.join(self.analysis_path, 'pixel6_emeter_mean.pqt'))
        print(power_mean)

    def trace_energy_estimate_analysis(self):
        log.info('Computing energy estimates')
        df = self.analysis.apply(ana.trace_energy_estimate_df)
        df.write_parquet(os.path.join(self.analysis_path, 'energy_estimate.pqt'))
        print(df)

        df = df_iterations_mean(df).with_columns(
            pl.lit('energy-estimate').alias('metric')
        )
        df.write_parquet(os.path.join(self.analysis_path, 'energy_estimate_mean.pqt'))
        print(df)

    def trace_cpu_idle_analysis(self):
        log.info('Collecting cpu_idle events')
        idle = self.analysis.apply(self.analysis.trace_cpu_idle_df)
        idle.write_parquet(os.path.join(self.analysis_path, 'cpu_idle.pqt'))
        print(idle)

        log.info('Computing idle residencies')
        idle_res = self.analysis.apply(self.analysis.trace_idle_residency_time_df)
        idle_res.write_parquet(os.path.join(self.analysis_path, 'idle_residency.pqt'))
        print(idle_res)

    def trace_cpu_idle_miss_analysis(self):
        log.info('Collecting cpu_idle_miss events')
        idle_miss = self.analysis.apply(self.analysis.trace_cpu_idle_miss_df)

        idle_miss.write_parquet(os.path.join(self.analysis_path, 'cpu_idle_miss.pqt'))
        print(idle_miss)

        idle_miss = idle_miss.groupby(['wa_path', 'kernel', 'iteration', 'cluster', 'below']).count()
        idle_miss = df_sort_by_clusters(idle_miss, value_cols=['below', 'count'])

        idle_miss.write_parquet(os.path.join(self.analysis_path, 'cpu_idle_miss_counts.pqt'))
        print(idle_miss)

    def trace_frequency_analysis(self):
        log.info('Collecting frequency data')
        freq = self.analysis.apply(ana.trace_frequency_df)
        freq.write_parquet(os.path.join(self.analysis_path, 'freqs.pqt'))
        print(freq)

        freq_mean = df_sort_by_clusters(df_iterations_mean(freq, other_cols=['cluster']), value_cols=['frequency'])
        freq_mean = freq_mean.with_columns(pl.col('frequency') / 1000)
        freq_mean.write_parquet(os.path.join(self.analysis_path, 'freqs_mean.pqt'))
        print(freq_mean)

        log.info('Computing frequency residency')
        freq_res = self.analysis.apply(ana.trace_frequency_residency_df)
        freq_mean.write_parquet(os.path.join(self.analysis_path, 'freqs_residency.pqt'))
        print(freq_res)

    def trace_overutilized_analysis(self):
        log.info('Collecting overutilized data')
        overutil = self.analysis.apply(ana.trace_overutilized_df)
        overutil.write_parquet(os.path.join(self.analysis_path, 'overutilized.pqt'))
        print(overutil)

        overutil = overutil.groupby(['wa_path']).mean().with_columns(
            pl.lit('overutilized').alias('metric'),
            pl.col('percentage').apply(lambda x: round(x, 2)),
            pl.col('time').apply(lambda x: round(x, 2)),
            pl.col('total_time').apply(lambda x: round(x, 2)),
        )[['metric', 'wa_path', 'time', 'total_time', 'percentage']]

        overutil.write_parquet(os.path.join(self.analysis_path, 'overutilized_mean.pqt'))
        print(overutil)

    def trace_sched_pelt_cfs_analysis(self):
        log.info('Collecting sched_pelt_cfs data')
        pelt = self.analysis.apply(ana.trace_sched_pelt_cfs_df)
        pelt.write_parquet(os.path.join(self.analysis_path, 'sched_pelt_cfs.pqt'))
        print(pelt)

        pelt = pelt.filter(pl.col('path') == '/')
        pelt.write_parquet(os.path.join(self.analysis_path, 'sched_pelt_cfs_root.pqt'))
        print(pelt)

        pelt = df_iterations_mean(pelt[['cluster', 'load', 'util', 'iteration', 'wa_path', 'kernel']],
                                  other_cols=['cluster']).sort(['wa_path', 'iteration'])
        pelt.write_parquet(os.path.join(self.analysis_path, 'sched_pelt_cfs_mean.pqt'))
        print(pelt)

    def trace_tasks_residency_time_analysis(self):
        log.info('Collecting task residency data')
        tasks = self.analysis.apply(ana.trace_tasks_residency_time_df)
        tasks.write_parquet(os.path.join(self.analysis_path, 'tasks_residency.pqt'))
        print(tasks)

        tasks = tasks.groupby(['wa_path', 'kernel', 'iteration', "comm"]).sum().sort('Total', descending=True)
        tasks.write_parquet(os.path.join(self.analysis_path, 'tasks_residency_total.pqt'))
        print(tasks)

        tasks = tasks.filter(pl.col('comm').str.starts_with('swapper').is_not()).groupby(
            ['wa_path', 'kernel', 'iteration']
        ).sum()
        tasks.write_parquet(os.path.join(self.analysis_path, 'tasks_residency_cpu_total.pqt'))
        print(tasks)

    def adpf_analysis(self):
        log.info('Collecting ADPF report data')

        def report_to_df(path, iteration):
            report = pl.read_csv(path, dtypes=[pl.Float64]*39).rename({'time since start': 'time'})
            return report.with_columns([pl.lit(iteration).alias('iteration')])

        reports = [
            report_to_df(job.get_artifact_path('adpf'), job.iteration)
            for job in self.wa_output.jobs
        ]

        reports = df_add_wa_output_tags(pl.concat(reports), self.wa_output)
        reports.write_parquet(os.path.join(self.analysis_path, 'adpf.pqt'))
        print(reports)

        aggregates = reports[['# frame count', 'average fps', 'iteration', 'kernel', 'wa_path']]
        report_aggs = df_iterations_mean(aggregates).with_columns(
            pl.lit(aggregates.groupby(['wa_path', 'kernel', 'iteration']).tail(1)['# frame count']).alias(
                'frame count'
            )
        )[['average fps', 'frame count', 'iteration', 'kernel', 'wa_path']]

        report_aggs.write_parquet(os.path.join(self.analysis_path, 'adpf_totals.pqt'))
        print(report_aggs)

    def thermal_analysis(self):
        log.info('Collecting thermal data')

        thermals = df_add_wa_output_tags(pl.concat([
            pl.read_csv(job.get_artifact_path('poller-output')).with_columns(pl.lit(job.iteration).alias('iteration'))
            for job in self.wa_output.jobs
        ]), self.wa_output).sort('iteration').rename({
            "thermal_zone0-temp": "big",
            "thermal_zone1-temp": "mid",
            "thermal_zone2-temp": "little",
            "time": "Time"
        })

        thermals.write_parquet(os.path.join(self.analysis_path, 'thermal.pqt'))
        print(thermals)

    def trace_wakeup_latency_analysis(self):
        log.info('Collecting task wakeup latencies')

        label_to_analysis = {
            'jankbench': ana.trace_wakeup_latency_jankbench_df,
            'geekbench': ana.trace_wakeup_latency_geekbench_df,
            'speedometer': ana.trace_wakeup_latency_speedometer_df,
            'drarm': ana.trace_wakeup_latency_drarm_df,
            'fortnite': ana.trace_wakeup_latency_fortnite_df,
        }

        label = self.wa_output.jobs[0].label
        if label not in label_to_analysis:
            log.error(f'Workload {label} does not yet support task wakeup latency analysis')
            return
        df = self.analysis.apply(label_to_analysis[label])

        df.write_parquet(os.path.join(self.analysis_path, 'wakeup_latency.pqt'))
        print(df)

        df = df.filter(pl.col('comm').str.starts_with('Hw').is_not())
        df = df_iterations_mean(df, other_cols=['comm'])
        df = df[['wa_path', 'iteration', 'comm', 'wakeup_latency']]
        df = df.sort(['iteration', 'wa_path', 'comm'], descending=[False, False, False])
        df.write_parquet(os.path.join(self.analysis_path, 'wakeup_latency_mean.pqt'))
        print(df)

    def trace_tasks_activations_analysis(self):
        log.info('Collecting task activations')

        label_to_analysis = {
            'jankbench': ana.trace_tasks_activations_jankbench_df,
            'geekbench': ana.trace_tasks_activations_geekbench_df,
            'speedometer': ana.trace_tasks_activations_speedometer_df,
            'drarm': ana.trace_tasks_activations_drarm_df,
            'fortnite': ana.trace_tasks_activations_fortnite_df,
        }

        label = self.wa_output.jobs[0].label
        if label not in label_to_analysis:
            log.error(f'Workload {label} does not yet support task activation analysis')
            return
        df = self.analysis.apply(label_to_analysis[label])

        df.write_parquet(os.path.join(self.analysis_path, 'task_activations.pqt'))
        print(df)

        df = df.filter(pl.col('active') == 1).groupby(['kernel', 'wa_path', 'iteration', 'cpu', 'comm']).agg(
            pl.col('duration').count().alias('count'), pl.col('duration').sum().alias('duration')
        ).with_columns(
            pl.col('cpu').apply(cpu_cluster).alias('cluster')
        )[['kernel', 'wa_path', 'iteration', 'cpu', 'cluster', 'comm', 'count', 'duration']]

        df.write_parquet(os.path.join(self.analysis_path, 'task_activations_stats.pqt'))
        print(df)

        df = df_sort_by_clusters(
            df.groupby(['kernel', 'wa_path', 'iteration', 'cluster', 'comm']).sum(),
            value_cols=['comm', 'count', 'duration']
        )

        df.write_parquet(os.path.join(self.analysis_path, 'task_activations_stats_cluster.pqt'))
        print(df)

    def trace_cgroup_attach_task_analysis(self):
        log.info('Collecting cgroup_attach_task events')
        df = self.analysis.apply(ana.trace_cgroup_attach_task_df)
        df.write_parquet(os.path.join(self.analysis_path, 'cgroup_attach_task.pqt'))
        print(df)

    def trace_wakeup_latency_cgroup_analysis(self):
        log.info('Collecting per-cgroup task wakeup latency')
        df = self.analysis.apply(ana.trace_wakeup_latency_cgroup_df)
        df.write_parquet(os.path.join(self.analysis_path, 'wakeup_latency_cgroup.pqt'))
        print(df)

        df = df.groupby(['wa_path', 'kernel', 'iteration', 'cgroup'], maintain_order=True).mean()
        df = df[['wa_path', 'iteration', 'cgroup', 'wakeup_latency']]
        df = df.sort(['iteration', 'wa_path', 'cgroup'], descending=False)

        df.write_parquet(os.path.join(self.analysis_path, 'wakeup_latency_cgroup_mean.pqt'))
        print(df)

    def trace_tasks_residency_cgroup_analysis(self):
        log.info('Collecting per-cgroup tasks residency')
        df = self.analysis.apply(ana.trace_tasks_residency_cgroup_df)
        df.write_parquet(os.path.join(self.analysis_path, 'tasks_residency_cgroup.pqt'))
        print(df)

        df = df.groupby(["wa_path", "cgroup", "iteration"]).sum().sort(
            ['iteration', 'wa_path', 'cgroup', 'Total'], descending=[False, False, False, True]
        )

        df.write_parquet(os.path.join(self.analysis_path, 'tasks_residency_cgroup_total.pqt'))
        print(df)

    def trace_uclamp_analysis(self):
        log.info('Collecting uclamp data')
        df = self.analysis.apply(ana.trace_uclamp_df)

        df.write_parquet(os.path.join(self.analysis_path, 'uclamp_updates.pqt'))
        print(df)

    def trace_perf_event_analysis(self):
        log.info('Collecting perf counter event data')
        df = self.analysis.apply(ana.trace_perf_counters_df)

        log.debug('Saving the perf counter event analysis file')
        df.write_parquet(os.path.join(self.analysis_path, 'perf_counters.pqt'))
        print(df)

    def trace_capacity_analysis(self):
        log.info('Collecting capacity data')
        df = self.analysis.apply(ana.trace_capacity_df)

        df.write_parquet(os.path.join(self.analysis_path, 'capacity.pqt'))
        print(df)
