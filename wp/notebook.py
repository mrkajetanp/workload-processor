import os
import pandas as pd
import polars as pl
import scipy as sp
import logging as log
import confuse
import shutil

from tabulate import tabulate
from functools import lru_cache, cached_property

from IPython.display import display
import plotly.express as px
import holoviews as hv
from holoviews import opts

from lisa.wa import WAOutput
from lisa.stats import Stats
from lisa.platforms.platinfo import PlatformInfo
from lisa.utils import LazyMapping

from wp.helpers import wa_output_to_mock_traces, wa_output_to_traces, flatten, cpu_cluster
from wp.constants import APP_NAME


def trim_number(x):
    if x > 1000000000:
        return f"{round(x / 1000000000, 3)}B"
    if x > 1000000:
        return f"{round(x / 1000000, 3)}M"
    if x > 10000:
        return f"{round(x / 1000, 2)}k"
        return str(x)
    if x != 0 and x < 0.01:
        return f"{round(x * 1000000, 2)}μ"
    return str(x)


def format_percentage(vals, perc, pvals, pval_threshold=0.02):
    result = round(perc, 2).astype(str).apply(
        lambda s: f"({'' if s.startswith('-') or (s == '0.0') else '+'}{s}%)"
    ).to_frame()
    result['vals'] = vals.apply(lambda x: trim_number(x))
    result['pvals'] = pvals
    result['pval_marker'] = pvals.apply(lambda x: "* " if x < pval_threshold else "")
    result['value'] = result['vals'] + " " + result['pval_marker'] + result['value']
    return result['value']


def ptable(df):
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False, floatfmt=".3f"))


def trim_wa_path(path):
    return "_".join(path.split("_")[1:-2])


class WorkloadNotebookAnalysis:
    def __init__(self, benchmark_path, benchmark_dirs, label=None):
        self.benchmark_path = benchmark_path
        self.benchmark_dirs = benchmark_dirs

        self.wa_outputs = [WAOutput(os.path.join(benchmark_path, benchmark_dir)) for benchmark_dir in benchmark_dirs]
        self.results = [wa_output['results'].df for wa_output in self.wa_outputs]
        self.kernels = [
            output._jobs[os.path.basename(output.path)][0].target_info.kernel_version.release
            for output in self.wa_outputs
        ]
        self.wa_paths = [
            trim_wa_path(os.path.basename(output.path))
            for output in self.wa_outputs
        ]
        self.results = pd.concat(self.results)

        if not self.results.empty:
            if 'scaled from(%)' in self.results.columns:
                self.results = self.results.drop(columns=['scaled from(%)'])
            self.results['wa_path'] = self.results['wa_path'].map(trim_wa_path)

            # separate perf results from benchmark results
            self.results_perf = self.results[
                self.results['metric'].str.contains('perf')
            ].reset_index(drop=True).query("value != 0")
            self.results_perf['metric'] = self.results_perf['metric'].str[7:]
            self.results = self.results[
                ~self.results['metric'].str.contains('perf')
            ].reset_index(drop=True).query("value != 0")
        else:
            self.results_perf = pd.DataFrame()

        self.analysis = dict()
        self.summary = dict()

        self.CPUS = [str(float(x)) for x in flatten(self.config['target']['clusters'].get().values())]
        self.CLUSTERS = list(self.config['target']['clusters'].get().keys())
        self.CLUSTERS_TOTAL = self.CLUSTERS + ['total']

        self.label = self.wa_outputs[0].jobs[0].label.capitalize() if not label else label
        self.workload_label = self.wa_outputs[0].jobs[0].label
        self.plot = WorkloadNotebookPlotter(self)

    @cached_property
    def plat_info(self):
        self.plat_info = None
        plat_info_path = os.path.expanduser(self.config['target']['plat_info'].get(str))
        if plat_info_path is not None:
            return PlatformInfo.from_yaml_map(plat_info_path)
        return None

    @cached_property
    def traces(self):
        trace_parquet_found = shutil.which('trace-parquet') is not None
        trace_function = wa_output_to_mock_traces if trace_parquet_found else wa_output_to_traces

        return LazyMapping({
            trim_wa_path(os.path.basename(wa_output.path)): lru_cache()(
                lambda k: trace_function(wa_output, self.plat_info)
            ) for wa_output in self.wa_outputs
        })

    def show(self):
        display(self.results)
        print('benchmark_dirs:', self.benchmark_dirs)
        print('wa_paths:', self.wa_paths)
        print('kernels:', self.kernels)

    @property
    def config(self):
        return confuse.Configuration(APP_NAME, __name__)

    def load_combined_analysis(self, name, trim_path=True, preprocess=lambda d: d,
                               postprocess=None, allow_missing=False):

        def load_parquet(benchmark):
            try:
                return preprocess(pd.read_parquet(os.path.join(self.benchmark_path, benchmark, 'analysis', name)))
            except FileNotFoundError as e:
                if allow_missing:
                    log.debug(e)
                    return pd.DataFrame()
                log.error(e)

        dfs = [load_parquet(benchmark) for benchmark in self.benchmark_dirs]
        result = pd.concat(dfs)
        if trim_path:
            result['wa_path'] = result['wa_path'].map(trim_wa_path)
        if postprocess is not None:
            result = postprocess(result)
        self.analysis[name.split('.')[0]] = result

    def plot_gmean_bars(self, df, x='stat', y='value', facet_col='metric', facet_col_wrap=3, title='',
                        width=None, height=600, gmean_round=1, include_columns=[], table_sort=None,
                        order_cluster=False, sort_ascending=False, include_total=False, debug=False,
                        percentage=True):

        shown_clusters = self.CLUSTERS if not include_total else self.CLUSTERS_TOTAL
        if 'unit' not in df.columns:
            df['unit'] = 'x'
        if 'metric' not in df.columns:
            df['metric'] = 'gmean'

        if debug:
            import pdb
            pdb.set_trace()

        # prepare the sort list
        sort_list = ['metric']
        if order_cluster:
            sort_list.append('order_cluster')
        sort_list.append('order_kernel')

        # prepare percentage differences & pvalues
        if percentage:
            # compute percentage differences
            stats_perc = Stats(df, ref_group={'wa_path': self.wa_paths[0]}, value_col=y,
                               agg_cols=['iteration'], stats={'gmean': sp.stats.gmean}).df
            # re-add stub a_wa_path
            stats_perc_vals_temp = stats_perc.query(f"wa_path == '{self.wa_paths[1]}'")
            stats_perc_vals_temp['wa_path'] = self.wa_paths[0]
            stats_perc_vals_temp['value'] = 0
            # re-combine a df with percentage differences
            stats_perc_vals = pd.concat([stats_perc_vals_temp, stats_perc])
            stats_perc_vals['order_kernel'] = stats_perc_vals['wa_path'].map(lambda x: self.wa_paths.index(x))

            if order_cluster:
                stats_perc_vals['order_cluster'] = stats_perc_vals['cluster'].map(lambda x: shown_clusters.index(x))
            # split into dfs with percentages and pvalues
            stats_perc_pvals = stats_perc_vals.query("stat == 'ks2samp_test'").sort_values(
                by=sort_list
            ).reset_index(drop=True)
            stats_perc_vals = stats_perc_vals.query("stat == 'gmean'").sort_values(by=sort_list).reset_index(drop=True)

        # compute absolute gmeans
        gmeans = Stats(df, agg_cols=['iteration'], stats={'gmean': sp.stats.gmean, 'std': None, 'sem': None}).df
        if gmean_round > 0:
            gmeans['value'] = round(gmeans['value'], gmean_round)
        gmeans['order_kernel'] = gmeans['wa_path'].map(lambda x: self.wa_paths.index(x))

        if order_cluster:
            gmeans['order_cluster'] = gmeans['cluster'].map(lambda x: shown_clusters.index(x))

        gmeans_mean = gmeans.query("stat == 'gmean'").sort_values(by=sort_list).reset_index(drop=True)

        # prepare the data table
        data_table_cols = [col for col in gmeans_mean.columns
                           if col in ([
                               'wa_path', 'value', 'test_name', 'variable', 'metric', 'chan_name', 'comm'
                           ] + include_columns)]
        data_table = gmeans_mean[data_table_cols].rename(columns={'wa_path': 'kernel'})
        if percentage:
            data_table['perc_diff'] = stats_perc_vals['value'].map(lambda x: str(round(x, 2)) + '%')
        data_table['value'] = data_table['value'].apply(lambda x: trim_number(x))
        if table_sort is not None:
            data_table = data_table.sort_values(by=table_sort)
        ptable(data_table)

        # prepare the plot labels
        plot_text = format_percentage(
            gmeans_mean['value'], stats_perc_vals['value'], stats_perc_pvals['value']
        ) if percentage else gmeans_mean['value']

        # plot bars
        fig = px.bar(gmeans_mean, x=x, y=y, color='wa_path', facet_col=facet_col, facet_col_wrap=facet_col_wrap,
                     barmode='group', title=title, width=width, height=height,
                     text=plot_text)
        fig.update_traces(textposition='outside')
        fig.update_yaxes(matches=None)
        if sort_ascending:
            fig.update_xaxes(categoryorder='total ascending')
        fig.show(renderer='iframe')

        return data_table

    def plot_lines_px(self, df, x='iteration', y='value', color='wa_path', facet_col=None, facet_col_wrap=2,
                      height=600, width=None, title=None, scale_y=False, renderer='iframe'):
        fig = px.line(df, x=x, y=y, color=color, facet_col=facet_col, facet_col_wrap=facet_col_wrap,
                      height=height, width=width, title=title)
        if not scale_y:
            fig.update_yaxes(matches=None)
        fig.show(renderer=renderer)


class WorkloadNotebookPlotter:
    def __init__(self, notebook_analysis):
        self.ana = notebook_analysis

    # TODO: detect missing and call processor?
    def _check_load_analysis(self, names, loader):
        if any([d not in self.ana.analysis for d in names]):
            log.debug(f'{names} not found in analysis, trying to load combined analysis')
            loader()

        if any([d not in self.ana.analysis for d in names]):
            log.error(f"{names} failed to load into analysis using {loader.__name__}")

    # -------- Results --------
    def results_line(self, metrics, height=600, width=900, columns=2,
                     title='Benchmark score per-iteration', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        metrics = sorted(metrics)

        ds = hv.Dataset(
            self.ana.results,
            ['iteration', hv.Dimension('wa_path', values=self.ana.wa_paths),
             hv.Dimension('metric', values=metrics)], 'value'
        )
        layout = ds.select(metric=metrics).to(hv.Curve, 'iteration', 'value').overlay('wa_path').opts(
            legend_position='bottom'
        ).layout('metric').opts(shared_axes=False, title=title).cols(columns)
        layout.opts(
            opts.Curve(height=height, width=width),
        )
        return layout

    def results_bar(self, metrics, height=600, width=None, columns=2,
                    title='gmean benchmark score', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        data = pl.from_pandas(self.ana.results).filter(
            pl.col('metric').is_in(metrics)
        ).to_pandas()

        self.ana.summary['results'] = self.ana.plot_gmean_bars(
            data, x='stat', y='value', facet_col='metric',
            facet_col_wrap=columns, title=title, width=width, height=height
        )

    # -------- Power Meter (pixel6_emeter) --------

    def _load_power_meter(self):
        def postprocess_pixel6_emeter_means(df):
            df_total = df.groupby(['wa_path', 'kernel', 'iteration']).sum(numeric_only=True).reset_index()
            df_total['channel'] = 'Total'

            df_cpu_total = df.query("channel.str.startswith('CPU')").groupby(
                ['wa_path', 'kernel', 'iteration']
            ).sum(numeric_only=True).reset_index()
            df_cpu_total['channel'] = 'CPU'
            return pd.concat([df, df_cpu_total, df_total])[['wa_path', 'kernel', 'iteration', 'channel', 'power']]

        self.ana.load_combined_analysis('pixel6_emeter.pqt')
        log.info('Loaded pixel6_emeter into analysis')
        self.ana.load_combined_analysis('pixel6_emeter_mean.pqt', postprocess=postprocess_pixel6_emeter_means)
        log.info('Loaded pixel6_emeter_mean into analysis')

    def power_meter_line(self, height=1000, width=None,
                         title='Mean power usage across iterations [mW]', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['pixel6_emeter', 'pixel6_emeter_mean'], self._load_power_meter)

        self.ana.plot_lines_px(
            self.ana.analysis['pixel6_emeter_mean'], y='power', facet_col='channel',
            facet_col_wrap=3, height=height, width=width, title=title
        )

    def power_meter_bar(self, height=600, width=None, title='Gmean power usage [mW]', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['pixel6_emeter', 'pixel6_emeter_mean'], self._load_power_meter)

        self.ana.summary['power_usage'] = self.ana.plot_gmean_bars(
            self.ana.analysis['pixel6_emeter_mean'].rename(columns={'power': 'value'}),
            x='channel', y='value', facet_col='metric', facet_col_wrap=5, title=title,
            height=height, width=width, include_total=True, include_columns=['channel']
        )

    # -------- Overutilized --------

    def _load_overutilized(self):
        def postprocess_overutil(df):
            df['time'] = round(df['time'], 2)
            df['total_time'] = round(df['total_time'], 2)
            return df

        self.ana.load_combined_analysis('overutilized.pqt', postprocess=postprocess_overutil)
        log.info('Loaded overutilized into analysis')
        self.ana.load_combined_analysis('overutilized_mean.pqt')
        log.info('Loaded overutilized_mean into analysis')

    def overutilized_line(self, height=600, width=None,
                          title='Overutilized percentage per-iteration', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['overutilized', 'overutilized_mean'], self._load_overutilized)
        ptable(self.ana.analysis['overutilized_mean'])
        self.ana.plot_lines_px(self.ana.analysis['overutilized'], y='percentage',
                               title=title, height=height, width=width)

    # -------- Frequency --------

    def _load_frequency(self):
        def postprocess_freq(df):
            df['unit'] = 'MHz'
            df['metric'] = 'frequency'
            df['order'] = df['cluster'].replace('little', 0).replace('mid', 1).replace('big', 2)
            return df.sort_values(by=['iteration', 'order']).rename(columns={'frequency': 'value'})

        self.ana.load_combined_analysis('freqs_mean.pqt', postprocess=postprocess_freq)
        log.info('Loaded freqs_mean into analysis')

    def frequency_line(self, height=600, width=None,
                       title='Mean cluster frequency across iterations', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['freqs_mean'], self._load_frequency)

        self.ana.plot_lines_px(self.ana.analysis['freqs_mean'], facet_col='cluster',
                               facet_col_wrap=3, title=title, height=height, width=width)

    def frequency_bar(self, height=600, width=None, title='Gmean frequency per cluster', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['freqs_mean'], self._load_frequency)

        self.ana.summary['frequency'] = self.ana.plot_gmean_bars(
            self.ana.analysis['freqs_mean'], x='metric', y='value', facet_col='cluster', facet_col_wrap=3,
            title=title, width=width, height=height, order_cluster=True,
            include_columns=['cluster'])

    # -------- Thermal --------

    def _load_thermal(self):
        def preprocess_thermal(df):
            return df.groupby(['iteration', 'kernel', 'wa_path']).mean().reset_index()

        def postprocess_thermal(df):
            for col in [c for c in df.columns if c not in ['time', 'iteration', 'kernel', 'wa_path']]:
                df[col] = df[col] / 1000
            df = round(df, 2)
            return df

        self.ana.load_combined_analysis('thermal.pqt', preprocess=preprocess_thermal, postprocess=postprocess_thermal)
        log.info('Loaded thermal into analysis')
        self.ana.analysis['thermal_melt'] = pd.melt(self.ana.analysis['thermal'],
                                                    id_vars=['iteration', 'wa_path', 'kernel'],
                                                    value_vars=['little', 'mid', 'big']
                                                    ).rename(columns={'variable': 'cluster'})
        log.info('Loaded thermal_melt into analysis')

    def thermal_line(self, height=600, width=None,
                     title='Mean cluster temperature across iterations', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['thermal', 'thermal_melt'], self._load_thermal)

        self.ana.plot_lines_px(self.ana.analysis['thermal_melt'], facet_col='cluster', height=height, width=width,
                               facet_col_wrap=3, title=title)

    def thermal_bar(self, height=600, width=None, title='Gmean temperature', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['thermal', 'thermal_melt'], self._load_thermal)

        self.ana.summary['thermal'] = self.ana.plot_gmean_bars(
            self.ana.analysis['thermal_melt'], x='cluster', y='value',
            facet_col='metric', facet_col_wrap=2, title=title,
            width=width, height=height, order_cluster=True, include_columns=['cluster']
        )

    # -------- Perf --------

    def perf_line(self, counters=None, height=340, width=600,
                  title='Perf counters across iterations', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        if not counters:
            counters = self.ana.config['notebook']['perf_counters'].get()

        ds = hv.Dataset(self.ana.results_perf, [
            'iteration', hv.Dimension('wa_path', values=self.ana.wa_paths), hv.Dimension('metric', values=counters)
        ], 'value')
        layout = ds.select(metric=counters).to(hv.Curve, 'iteration', 'value').overlay('wa_path').opts(
            legend_position='bottom'
        ).layout('metric').opts(shared_axes=False, title=title).cols(3)
        layout.opts(
            opts.Curve(width=width, height=height),
            opts.Overlay(legend_position='bottom'),
        )

        return layout

    def perf_bar(self, counters=None, height=900, width=None, title='Gmean perf counters', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        if not counters:
            counters = self.ana.config['notebook']['perf_counters'].get()

        self.ana.plot_gmean_bars(
            self.ana.results_perf.query("metric in @counters")[['kernel', 'wa_path', 'iteration', 'metric', 'value']],
            x='stat', y='value', facet_col='metric', facet_col_wrap=5, title=title, width=width, height=height
        )

    # -------- Idle --------

    def _load_idle_residency(self):
        self.ana.load_combined_analysis('idle_residency.pqt')
        self.ana.analysis['idle_residency'] = self.ana.analysis['idle_residency'].groupby(
            ['wa_path', 'cluster', 'idle_state'], sort=False
        ).mean(numeric_only=True).reset_index()[['wa_path', 'cluster', 'idle_state', 'time']]
        self.ana.analysis['idle_residency']['time'] = round(self.ana.analysis['idle_residency']['time'], 2)
        log.info('Loaded idle_residency into analysis')

    def idle_residency_bar(self, height=600, width=None, title='Idle state residencies', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['idle_residency'], self._load_idle_residency)

        fig = px.bar(
            self.ana.analysis['idle_residency'], x='idle_state', y='time', color='wa_path',
            facet_col='cluster', barmode='group', text=self.ana.analysis['idle_residency']['time'],
            width=width, height=height, title=title
        )
        fig.update_traces(textposition='outside')
        fig.show(renderer='iframe')

    def _load_idle_miss(self):
        def preprocess_cpu_idle_miss_df(df):
            df = df.groupby(['wa_path', 'kernel', 'cluster', 'below']).sum().reset_index()
            wa_path = trim_wa_path(df['wa_path'].iloc[0])
            if not wa_path:
                return df
            wakeup_count = len(self.ana.analysis['cpu_idle'].query("wa_path == @wa_path and state == -1"))
            df['count_perc'] = round(df['count'] / wakeup_count * 100, 3)
            return df

        def postprocess_cpu_idle_miss_df(df):
            df['type'] = df['below'].replace(0, 'too deep').replace(1, 'too shallow')
            df['order'] = df['cluster'].replace('little', 0).replace('mid', 1).replace('big', 2)
            df = df.sort_values(by=['wa_path', 'kernel', 'order', 'type'])
            return df

        self.ana.load_combined_analysis('cpu_idle.pqt')
        log.info('Loaded cpu_idle into analysis')
        self.ana.load_combined_analysis(
            'cpu_idle_miss_counts.pqt', preprocess=preprocess_cpu_idle_miss_df, postprocess=postprocess_cpu_idle_miss_df
        )
        log.info('Loaded cpu_idle_miss_counts into analysis')

    def idle_miss_bar(self, height=600, width=None,
                      title='CPUIdle misses as percentage of all wakeups', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['cpu_idle_miss_counts'], self._load_idle_miss)

        ptable(self.ana.analysis['cpu_idle_miss_counts'].groupby(['wa_path', 'type']).sum().reset_index()[
            ['wa_path', 'type', 'count_perc']
        ])
        fig = px.bar(
            self.ana.analysis['cpu_idle_miss_counts'], x='type', y='count_perc', color='wa_path',
            facet_col='cluster', barmode='group', text=self.ana.analysis['cpu_idle_miss_counts']['count_perc'],
            width=width, height=height, title=title
        )
        fig.show(renderer='iframe')

    # -------- Energy Estimate --------

    def _load_energy_estimate(self):
        self.ana.load_combined_analysis('energy_estimate_mean.pqt')
        log.info('Loaded energy_estimate_mean into analysis')
        self.ana.analysis['energy_estimate_melt'] = pd.melt(
            self.ana.analysis['energy_estimate_mean'], id_vars=['iteration', 'wa_path'],
            value_vars=['little', 'mid', 'big', 'total']
        ).rename(columns={'variable': 'cluster'})
        log.info('Loaded energy_estimate_melt into analysis')

    def energy_estimate_line(self, height=1000, width=None,
                             title='Mean energy estimate across iterations [bW]', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['energy_estimate_melt'], self._load_energy_estimate)

        self.ana.plot_lines_px(
            self.ana.analysis['energy_estimate_melt'], facet_col='cluster',
            height=height, width=width, title=title
        )

    def energy_estimate_bar(self, height=600, width=None,
                            title='Gmean energy estimate [bW]', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['energy_estimate_melt'], self._load_energy_estimate)

        self.ana.summary['energy_estimate'] = self.ana.plot_gmean_bars(
            self.ana.analysis['energy_estimate_melt'], x='cluster', y='value', facet_col='metric',
            facet_col_wrap=5, title=title, width=width, height=height,
            include_columns=['cluster'], order_cluster=True, include_total=True
        )

    # -------- CFS signals --------

    def _load_sched_pelt_cfs(self):
        self.ana.load_combined_analysis('sched_pelt_cfs_mean.pqt')
        log.info('Loaded sched_pelt_cfs_mean into analysis')
        self.ana.analysis['sched_pelt_cfs_melt'] = pd.melt(
            self.ana.analysis['sched_pelt_cfs_mean'],
            id_vars=['iteration', 'wa_path', 'kernel', 'cluster'], value_vars=['util', 'load']
        )
        log.info('Loaded sched_pelt_cfs_melt into analysis')

    def sched_pelt_cfs_line(self, height=400, width=700,
                            title='Mean cluster', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['sched_pelt_cfs_mean'], self._load_sched_pelt_cfs)

        signals = ['util', 'load']
        ds = hv.Dataset(
            self.ana.analysis['sched_pelt_cfs_mean'],
            ['iteration', hv.Dimension('wa_path', values=self.ana.wa_paths),
             hv.Dimension('cluster', values=self.ana.CLUSTERS)],
            signals
        )
        layout = hv.Layout([
            ds.to(hv.Curve, 'iteration', signal).overlay('wa_path').opts(legend_position='bottom').layout(
                'cluster'
            ).opts(title=f"{title} {signal}", framewise=True)
            for signal in signals
        ]).cols(1)
        layout.opts(
            opts.Curve(width=width, height=height, framewise=True),
        )
        return layout

    def sched_pelt_cfs_bar(self, height=1000, width=None,
                           title='Gmean cfs signals', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['sched_pelt_cfs_melt'], self._load_sched_pelt_cfs)

        self.ana.summary['cfs_signals'] = self.ana.plot_gmean_bars(
            self.ana.analysis['sched_pelt_cfs_melt'], x='cluster', y='value',
            facet_col='variable', facet_col_wrap=1, title=title,
            width=width, height=height, order_cluster=True, include_columns=['cluster']
        )

    # -------- Wakeup latency --------

    def _load_wakeup_latency(self):
        def postprocess_wakeup_latency_mean(df):
            df = df.rename(columns={'wakeup_latency': 'value'})
            df['order'] = df['wa_path'].map(lambda x: self.ana.wa_paths.index(x))
            df['unit'] = 'x'
            return df

        self.ana.load_combined_analysis('wakeup_latency_mean.pqt', postprocess=postprocess_wakeup_latency_mean)
        log.info('Loaded wakeup_latency_mean into analysis')

        def postprocess_wakeup_latency(df):
            df['order'] = df['wa_path'].map(lambda x: self.ana.wa_paths.index(x))
            df['cluster'] = df['cpu'].copy().apply(cpu_cluster)
            df['order_cluster'] = df['cluster'].map(lambda x: self.ana.CLUSTERS.index(x))
            df['target_cluster'] = df['target_cpu'].copy().apply(cpu_cluster)
            df['order_target_cluster'] = df['target_cluster'].map(lambda x: self.ana.CLUSTERS.index(x))
            return df

        self.ana.load_combined_analysis('wakeup_latency.pqt', postprocess=postprocess_wakeup_latency)
        log.info('Loaded wakeup_latency into analysis')

        self.ana.analysis['wakeup_latency_quantiles'] = self.ana.analysis['wakeup_latency'].groupby([
            'comm', 'wa_path', 'iteration'
        ]).quantile([0.9, 0.95, 0.99], numeric_only=True).reset_index()[
            ['comm', 'wa_path', 'level_3', 'iteration', 'wakeup_latency', 'order']
        ].rename(columns={'level_3': 'quantile'}).sort_values(by=['comm', 'order'])
        log.info('Loaded wakeup_latency_quantiles into analysis')

        self.ana.analysis['wakeup_latency_execution_cluster'] = self.ana.analysis['wakeup_latency'].groupby([
            'comm', 'wa_path', 'cluster'
        ]).mean(numeric_only=True).reset_index().sort_values(by=['comm', 'order_cluster', 'order'])[
            ['comm', 'wa_path', 'cluster', 'wakeup_latency']
        ]
        log.info('Loaded wakeup_latency_execution_cluster into analysis')

        self.ana.analysis['wakeup_latency_target_cluster'] = self.ana.analysis['wakeup_latency'].groupby([
            'comm', 'wa_path', 'target_cluster'
        ]).mean(numeric_only=True).reset_index().sort_values(by=['comm', 'order_target_cluster', 'order'])[
            ['comm', 'wa_path', 'target_cluster', 'wakeup_latency']
        ]
        log.info('Loaded wakeup_latency_target_cluster into analysis')

    def wakeup_latency_line(self, height=600, width=None,
                            title='Task wakeup latencies across iterations', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"
        self._check_load_analysis(['wakeup_latency_mean'], self._load_wakeup_latency)

        self.ana.plot_lines_px(
            self.ana.analysis['wakeup_latency_mean'], facet_col='comm',
            facet_col_wrap=3, height=height, width=width, title=title
        )

    def wakeup_latency_bar(self, height=600, width=None,
                           title='Gmean task wakeup latency', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['wakeup_latency_mean'], self._load_wakeup_latency)

        self.ana.summary['wakeup_latency'] = self.ana.plot_gmean_bars(
            self.ana.analysis['wakeup_latency_mean'], x='metric', y='value', facet_col='comm', facet_col_wrap=3,
            title=title, table_sort=['comm', 'kernel'], gmean_round=0, width=width, height=height
        )

    def wakeup_latency_quantiles_bar(self, height=1300, width=None,
                                     title='Gmean latency quantile', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['wakeup_latency_quantiles'], self._load_wakeup_latency)

        self.ana.summary['wakeup_latency_quantiles'] = self.ana.plot_gmean_bars(
            self.ana.analysis['wakeup_latency_quantiles'].rename(columns={'wakeup_latency': 'value'}),
            x='quantile', y='value', facet_col='comm', facet_col_wrap=1, title=title,
            width=width, height=height, include_columns=['quantile'], table_sort=['quantile', 'comm'], gmean_round=0
        )

    def wakeup_latency_execution_cluster_bar(self, height=1300, width=None, include_label=True,
                                             title='Mean task wakeup latency per execution cluster'):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['wakeup_latency_execution_cluster'], self._load_wakeup_latency)

        fig = px.bar(
            self.ana.analysis['wakeup_latency_execution_cluster'], x='cluster', y='wakeup_latency', color='wa_path',
            facet_col='comm', barmode='group', facet_col_wrap=1, width=width, height=height, title=title,
            text=self.ana.analysis['wakeup_latency_execution_cluster']['wakeup_latency'].apply(trim_number),
        )
        fig.update_traces(textposition='outside')
        self.ana.analysis['wakeup_latency_execution_cluster']['wakeup_latency'] = self.ana.analysis[
            'wakeup_latency_execution_cluster'
        ]['wakeup_latency'].apply(trim_number)
        ptable(self.ana.analysis['wakeup_latency_execution_cluster'])
        fig.show(renderer='iframe')

    def wakeup_latency_target_cluster_bar(self, height=1300, width=None, include_label=True,
                                          title='Mean task wakeup latency per target cluster'):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['wakeup_latency_target_cluster'], self._load_wakeup_latency)

        fig = px.bar(
            self.ana.analysis['wakeup_latency_target_cluster'], x='target_cluster', y='wakeup_latency', color='wa_path',
            facet_col='comm', barmode='group', facet_col_wrap=1, width=width, height=height, title=title,
            text=self.ana.analysis['wakeup_latency_target_cluster']['wakeup_latency'].apply(trim_number),
        )
        fig.update_traces(textposition='outside')
        self.ana.analysis['wakeup_latency_target_cluster']['wakeup_latency'] = self.ana.analysis[
            'wakeup_latency_target_cluster'
        ]['wakeup_latency'].apply(trim_number)
        ptable(self.ana.analysis['wakeup_latency_target_cluster'])
        fig.show(renderer='iframe')

    # -------- Wakeup latency - cgroup --------

    def _load_wakeup_latency_cgroup(self):
        def postprocess_cgroup_latency(df):
            df = df.rename(columns={'wakeup_latency': 'value'})
            df['order'] = df['wa_path'].map(lambda x: self.ana.wa_paths.index(x))
            return df

        self.ana.load_combined_analysis('wakeup_latency_cgroup.pqt', postprocess=postprocess_cgroup_latency)
        log.info('Loaded wakeup_latency_cgroup into analysis')

        self.ana.analysis['wakeup_latency_cgroup_mean'] = self.ana.analysis['wakeup_latency_cgroup'].groupby(
            ["wa_path", "cgroup", "iteration", "order"]
        ).mean(numeric_only=True).reset_index().sort_values(by=["order", "cgroup", "iteration"])[
            ['wa_path', 'cgroup', 'iteration', 'value', 'order']
        ]
        log.info('Loaded wakeup_latency_cgroup_mean into analysis')

        self.ana.analysis['wakeup_latency_cgroup_quantiles'] = self.ana.analysis['wakeup_latency_cgroup'].groupby(
            ['cgroup', 'wa_path', 'iteration']
        ).quantile([0.9, 0.95, 0.99], numeric_only=True).reset_index()[
            ['cgroup', 'wa_path', 'level_3', 'iteration', 'value', 'order']
        ].rename(columns={'level_3': 'quantile'}).sort_values(by=['cgroup', 'order'])
        log.info('Loaded wakeup_latency_cgroup_quantiles into analysis')

    def wakeup_latency_cgroup_line(self, height=600, width=None,
                                   title='cgroup wakeup latencies across iterations', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['wakeup_latency_cgroup_mean'], self._load_wakeup_latency_cgroup)

        self.ana.plot_lines_px(
            self.ana.analysis['wakeup_latency_cgroup_mean'], facet_col='cgroup', facet_col_wrap=3,
            height=height, width=width, title=title
        )

    def wakeup_latency_cgroup_bar(self, height=600, width=None,
                                  title='Gmean task wakeup latency per-cgroup', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['wakeup_latency_cgroup_mean'], self._load_wakeup_latency_cgroup)

        self.ana.summary['wakeup_latency_cgroup'] = self.ana.plot_gmean_bars(
            self.ana.analysis['wakeup_latency_cgroup_mean'], x='metric', y='value', facet_col='cgroup',
            title=title, include_columns=['cgroup'], table_sort=['cgroup'], gmean_round=0, width=width, height=height
        )

    def wakeup_latency_cgroup_quantiles_bar(self, height=1400, width=None,
                                            title='Gmean latency quantile per-cgroup', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['wakeup_latency_cgroup_mean'], self._load_wakeup_latency_cgroup)

        self.ana.summary['wakeup_latency_cgroup_quantiles'] = self.ana.plot_gmean_bars(
            self.ana.analysis['wakeup_latency_cgroup_quantiles'], x='quantile', y='value', facet_col='cgroup',
            facet_col_wrap=1, title=title, include_columns=['cgroup', 'quantile'],
            table_sort=['quantile', 'cgroup'], width=width, height=height, gmean_round=0
        )

    # -------- Task CPU residency --------

    def _load_tasks_cpu_residency(self):
        cpus_prefix = [f"cpu{int(float(x))}" for x in self.ana.CPUS]

        def postprocess_tasks_residency_cpu_total(df):
            df = df.rename(columns={'Total': 'total'})
            return df

        self.ana.load_combined_analysis('tasks_residency_cpu_total.pqt',
                                        postprocess=postprocess_tasks_residency_cpu_total)
        log.info('Loaded tasks_residency_cpu_total into analysis')

        self.ana.analysis['tasks_residency_cpu_total_melt'] = pd.melt(
            self.ana.analysis['tasks_residency_cpu_total'],
            id_vars=['iteration', 'wa_path', 'kernel'], value_vars=cpus_prefix
        ).rename(columns={'variable': 'cpu'}).sort_values(['wa_path', 'kernel', 'iteration', 'cpu'])
        log.info('Loaded tasks_residency_cpu_total_melt into analysis')

        # TODO: sort by order of clusters
        self.ana.analysis['tasks_residency_cpu_total_cluster_melt'] = pd.melt(
            self.ana.analysis['tasks_residency_cpu_total'], id_vars=['iteration', 'wa_path', 'kernel'],
            value_vars=self.ana.CLUSTERS_TOTAL
        ).rename(columns={'variable': 'cluster'}).sort_values(['wa_path', 'kernel', 'iteration', 'cluster'])
        log.info('Loaded tasks_residency_cpu_total_cluster_melt into analysis')

        def postprocess_tasks_residency_total(df):
            tasks_important = self.ana.config['processor']['important_tasks'][self.ana.workload_label].get()
            if not tasks_important:
                return df
            df = df.rename(columns={'Total': 'total'}).query("comm in @tasks_important")
            return df

        self.ana.load_combined_analysis('tasks_residency_total.pqt', postprocess=postprocess_tasks_residency_total)
        log.info('Loaded tasks_residency_total into analysis')

        self.ana.analysis['tasks_residency_total_melt'] = pd.melt(
            self.ana.analysis['tasks_residency_total'], id_vars=['iteration', 'wa_path', 'kernel', 'comm'],
            value_vars=cpus_prefix
        ).rename(columns={'variable': 'cpu'}).sort_values(['wa_path', 'kernel', 'iteration', 'cpu'])
        log.info('Loaded tasks_residency_total_melt into analysis')

        self.ana.analysis['tasks_residency_total_cluster_melt'] = pd.melt(
            self.ana.analysis['tasks_residency_total'], id_vars=['iteration', 'wa_path', 'kernel', 'comm'],
            value_vars=self.ana.CLUSTERS_TOTAL
        ).rename(columns={'variable': 'cluster'}).sort_values(['wa_path', 'kernel', 'iteration', 'cluster'])
        log.info('Loaded tasks_residency_total_cluster_melt into analysis')

        def postprocess_tasks_residency(df):
            tasks_important = self.ana.config['processor']['important_tasks'][self.ana.workload_label].get()
            if not tasks_important:
                return df
            df = df.rename(columns={'Total': 'total'}).query("comm in @tasks_important")
            return df

        self.ana.load_combined_analysis('tasks_residency.pqt', postprocess=postprocess_tasks_residency)
        log.info('Loaded tasks_residency into analysis')

        self.ana.analysis['tasks_residency_cluster_melt'] = pd.melt(
            self.ana.analysis['tasks_residency'], id_vars=['iteration', 'wa_path', 'kernel', 'comm'],
            value_vars=self.ana.CLUSTERS_TOTAL
        ).rename(columns={'variable': 'cluster'})
        log.info('Loaded tasks_residency_cluster_melt into analysis')

    def tasks_cpu_residency_cluster_line(self, height=600, width=None,
                                         title='Mean cluster CPU residency', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['tasks_residency_cpu_total_cluster_melt'], self._load_tasks_cpu_residency)

        self.ana.plot_lines_px(
            self.ana.analysis['tasks_residency_cpu_total_cluster_melt'], facet_col='cluster',
            title=title, height=height, width=width, facet_col_wrap=4
        )

    def tasks_cpu_residency_cluster_bar(self, height=800, width=None,
                                        title='Gmean cluster CPU residency', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['tasks_residency_cpu_total_cluster_melt'], self._load_tasks_cpu_residency)

        self.ana.summary['tasks_cpu_residency_cluster'] = self.ana.plot_gmean_bars(
            self.ana.analysis['tasks_residency_cpu_total_cluster_melt'], x='cluster', y='value', facet_col='metric',
            facet_col_wrap=1, title=title, include_columns=['cluster'], height=height, width=width, order_cluster=True,
            include_total=True
        )

    def tasks_cpu_residency_per_task_bar(self, height=1200, width=None,
                                         title='Gmean cluster per-task CPU residency', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['tasks_residency_total_cluster_melt'], self._load_tasks_cpu_residency)

        self.ana.summary['tasks_cpu_residency_per_task'] = self.ana.plot_gmean_bars(
            self.ana.analysis['tasks_residency_total_cluster_melt'], x='cluster', y='value', facet_col='comm',
            facet_col_wrap=1, title=title, include_columns=['cluster'], height=height,
            width=width, order_cluster=True, include_total=True
        )

    # TODO: CPUs line plot
    def tasks_cpu_residency_cpu_bar(self, height=1400, width=None,
                                    title='Gmean CPU residency', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['tasks_residency_total_melt'], self._load_tasks_cpu_residency)

        self.ana.summary['tasks_cpu_residency_cpus'] = self.ana.plot_gmean_bars(
            self.ana.analysis['tasks_residency_total_melt'], x='cpu', y='value', facet_col='comm',
            facet_col_wrap=1, title=title, width=width, height=height
        )

    # -------- cgroup CPU residency --------

    def _load_cgroup_cpu_residency(self):
        def postprocess_cgroup_task_residency(df):
            df = df.rename(columns={'Total': 'total'})[
                ['wa_path', 'cgroup', 'iteration', 'total', 'little', 'mid', 'big'] + self.ana.CPUS
            ]
            return df

        self.ana.load_combined_analysis('tasks_residency_cgroup_total.pqt',
                                        postprocess=postprocess_cgroup_task_residency)
        log.info('Loaded tasks_residency_cgroup_total into analysis')

        self.ana.analysis['cgroup_residency_total_melt'] = pd.melt(
            self.ana.analysis['tasks_residency_cgroup_total'], id_vars=['iteration', 'wa_path', 'cgroup'],
            value_vars=self.ana.CPUS
        ).rename(columns={'variable': 'cpu'})
        log.info('Loaded cgroup_residency_total_melt into analysis')

        self.ana.analysis['cgroup_residency_total_cluster_melt'] = pd.melt(
            self.ana.analysis['tasks_residency_cgroup_total'], id_vars=['iteration', 'wa_path', 'cgroup'],
            value_vars=self.ana.CLUSTERS_TOTAL
        ).rename(columns={'variable': 'cluster'})
        log.info('Loaded cgroup_residency_total_cluster_melt into analysis')

    def cgroup_cpu_residency_cluster_bar(self, height=1100, width=None,
                                         title='Gmean cluster CPU residency per-cgroup', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['cgroup_residency_total_cluster_melt'], self._load_cgroup_cpu_residency)

        self.ana.summary['cgroup_cpu_residency_cluster'] = self.ana.plot_gmean_bars(
            self.ana.analysis['cgroup_residency_total_cluster_melt'], x='cluster', y='value', facet_col='cgroup',
            facet_col_wrap=1, title=title, width=width, height=height,
            include_columns=['cgroup', 'cluster'], table_sort=['cgroup', 'cluster'],
            order_cluster=True, include_total=True
        )

    def cgroup_cpu_residency_cpu_bar(self, height=1100, width=None,
                                     title='Gmean cgroup CPU residency', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self._check_load_analysis(['cgroup_residency_total_melt'], self._load_cgroup_cpu_residency)

        self.ana.summary['cgroup_cpu_residency_cpu'] = self.ana.plot_gmean_bars(
            self.ana.analysis['cgroup_residency_total_melt'], x='cpu', y='value', facet_col='cgroup',
            facet_col_wrap=1, title='', width=width, height=height, include_columns=['cgroup', 'cpu']
        )
