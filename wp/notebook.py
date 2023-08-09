"""Notebook analysis module. Intended to be used inside a Jupyter notebook on workload runs processed with
`wp.processor`."""

import os
import pandas as pd
import polars as pl
import scipy as sp
import logging as log
import confuse
import shutil
import functools
import plotly

from typing import Dict, List, Callable, Optional
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
from lisa.datautils import series_mean

from wp.helpers import wa_output_to_mock_traces, wa_output_to_traces, flatten, cpu_cluster
from wp.constants import APP_NAME


class WorkloadNotebookAnalysis:
    """
    Container class for analysis of different runs (potentially with many iterations) of a single workload.

    ```python
    gb5 = WorkloadNotebookAnalysis('/home/user/tmp/geekbench/', [
        'geekbench_baseline_3_3101',
        'geekbench_ufc_feec_all_cpus_3_3001',
    ], label='Geekbench 5')
    ```
    """

    def __init__(self, benchmark_path: str, benchmark_dirs: List[str], label: Optional[str] = None):
        """
        Create a notebook analysis object.
        :param benchmark_path: Absolute path to where the benchmark runs can be found.
        :param benchmark_dirs: Directories of the benchmark runs to be included in analysis.
        :param label: Label to display on plots. Defaults to the capitalised workload name.
        """
        self.benchmark_path: str = benchmark_path
        """Absolute path to where the benchmark runs can be found."""
        self.benchmark_dirs: List[str] = benchmark_dirs
        """Directories of the benchmark runs to be included in analysis."""

        self.wa_outputs: List[WAOutput] = [
            WAOutput(os.path.join(benchmark_path, benchmark_dir)) for benchmark_dir in benchmark_dirs
        ]
        """List of `lisa.wa.WAOutput` corresponding to the runs in the analysis"""
        self.kernels: List[str] = [
            output._jobs[os.path.basename(output.path)][0].target_info.kernel_version.release
            for output in self.wa_outputs
        ]
        """Kernel versions of the runs included in analysis"""
        self.tags: List[str] = [
            trim_wa_path(os.path.basename(output.path))
            for output in self.wa_outputs
        ]
        """Tags of the runs included in analysis"""
        self.label: str = self.wa_outputs[0].jobs[0].label.capitalize() if not label else label
        """Label displayed on the plots"""
        self.workload_label = self.wa_outputs[0].jobs[0].label
        """Label used internally to distinguish workloads (the same as the WA  joblabel)"""
        self.results: pd.DataFrame = pd.concat([wa_output['results'].df for wa_output in self.wa_outputs])
        """`workload-automation` results dataframe"""

        self.results_perf: pd.DataFrame = None
        """`perf` dataframe if `perf` was enabled for the runs"""
        if not self.results.empty:
            if 'scaled from(%)' in self.results.columns:
                self.results = self.results.drop(columns=['scaled from(%)'])
            self.results['tag'] = self.results['wa_path'].map(trim_wa_path)

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

        self.analysis: Dict[str, pd.DataFrame] = dict()
        """Loaded analysis metrics data"""
        self.summary: Dict[str, pd.DataFrame] = dict()
        """Summary data used by `WorkloadNotebookPlotter.summary`"""
        self.plot: WorkloadNotebookPlotter = WorkloadNotebookPlotter(self)
        """Proxy for accessing plotting functions"""
        self.px_figures: Dict[str, plotly.graph_objs._figure.Figure] = dict()
        """Figures plotted using `plotly.express`"""
        self.hv_figures: Dict = dict()
        """Figures plotted using `holoviews`"""

        self.CPUS: List[str] = [str(float(x)) for x in flatten(self.config['target']['clusters'].get().values())]
        """List of CPUs on the target device used internally. Automatically pulled from config."""
        self.CLUSTERS: List[str] = list(self.config['target']['clusters'].get().keys())
        """List of cluster names on the target device used internally. Automatically pulled from config."""
        self.CLUSTERS_TOTAL: List[str] = self.CLUSTERS + ['total']
        """List of clusters + 'total'. Used internally."""

    @property
    def config(self) -> confuse.Configuration:
        """Config object accessor"""
        return confuse.Configuration(APP_NAME, __name__)

    @cached_property
    def plat_info(self) -> PlatformInfo:
        """Platform info of the target device if available from the config, otherwise None"""
        self.plat_info = None
        plat_info_path = os.path.expanduser(self.config['target']['plat_info'].get(str))
        if plat_info_path is not None:
            return PlatformInfo.from_yaml_map(plat_info_path)
        return None

    @cached_property
    def traces(self) -> LazyMapping:
        """
        Traces of each of the iterations in each of the runs, indexed by workload tag & iteration number
        ```python
        power_event = gb6.traces['baseline'][1].df_event('pixel6_emeter')
        ```
        """
        trace_parquet_found = shutil.which('trace-parquet') is not None
        trace_function = wa_output_to_mock_traces if trace_parquet_found else wa_output_to_traces

        return LazyMapping({
            trim_wa_path(os.path.basename(wa_output.path)): lru_cache()(
                lambda _: trace_function(wa_output, self.plat_info)
            ) for wa_output in self.wa_outputs
        })

    def show(self):
        """Print the results dataframe, benchmark_dirs, tags and kernels versions"""
        display(self.results)
        print('benchmark_dirs:', self.benchmark_dirs)
        print('tags:', self.tags)
        print('kernels:', self.kernels)

    def load_combined_analysis(self, name: str, allow_missing: bool = False,
                               preprocess: Callable[[pd.DataFrame], pd.DataFrame] = lambda d: d,
                               postprocess: Callable[[pd.DataFrame], pd.DataFrame] = None):
        """
        Load analysis generated by `workload-processor process` from `analysis/` into the notebook's analysis dict.

        The resulting dataframe will be added as e.g. `gb5.analysis['overutilized']` using the base filename as key.

        :param name: Filename of the analysis pqt
        :param allow_missing: Replace missing data with empty dataframes instead of raising an error
        :param preprocess: Function applied onto each analysis dataframe before they're concatenated
        :param postprocess: Function applied onto the resulting concatenated dataframe
        """
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
        result['tag'] = result['wa_path'].map(trim_wa_path)
        if postprocess is not None:
            result = postprocess(result)
        self.analysis[name.split('.')[0]] = result

    def _title_to_filename(self, title, suffix):
        return "".join([lt for lt in title.lower() if lt.isalnum() or lt == ' ']).replace(' ', '_').replace(
            '__', '_'
        ) + '__' + "__".join(self.tags) + suffix

    def save_image_plots(self, directory: str, extension: str = 'png', width: int = 1800):
        """
        Save all image plots contained in `px_figures` and `hv_figures` into a directory.

        :param directory: Directory to save the plots into
        :param extension: File extension of the resulting plots
        :param width: Width of the resulting plots
        """
        for name, fig in self.px_figures.items():
            filename = f"{directory}/{name}.{extension}"
            self.px_figures[name].write_image(filename, width=width)
            log.debug(f'Successfully saved px figure {filename}')
        log.debug('All px image plots saved successfully')

        for name, fig in self.hv_figures.items():
            filename = f"{directory}/{name}.{extension}"
            hv.save(self.hv_figures[name], filename, fmt=extension)
            log.debug(f'Successfully saved hv figure {filename}')
        log.debug('All hv image plots saved successfully')
        log.info('All image plots saved successfully')


class WorkloadNotebookPlotter:
    """
    Container class for plotting methods that operate on data from `WorkloadNotebookAnalysis`.
    """
    def __init__(self, notebook_analysis: WorkloadNotebookAnalysis):
        """
        Create a new `WorkloadNotebookPlotter`. This will automatically be done when initialising
        `WorkloadNotebookAnalysis` and so most likely this should not be called outside of that context.

        :param notebook_analysis: Notebook analysis object with the data to plot
        """
        self.ana: WorkloadNotebookAnalysis = notebook_analysis
        """Handle for accessing analysis data when creating the plots"""

    def analysis_to_loader(self, analysis: str) -> Callable:
        """
        Convert an analysis name to the loader function used for loading it. The names should match what can be
        found in `WorkloadNotebookAnalysis.analysis` after the loader function has executed successfully.

        :return: Loader function that takes no arguments and loads the requested analysis
        """
        mapping = {
            'pixel6_emeter': self._load_power_meter,
            'pixel6_emeter_mean': self._load_power_meter,
            'overutilized': self._load_overutilized,
            'overutilized_mean': self._load_overutilized,
            'freqs_mean': self._load_frequency,
            'thermal': self._load_thermal,
            'thermal_melt': self._load_thermal,
            'idle_residency': self._load_idle_residency,
            'cpu_idle_miss_counts': self._load_idle_miss,
            'energy_estimate_melt': self._load_energy_estimate,
            'sched_pelt_cfs_mean': self._load_sched_pelt_cfs,
            'wakeup_latency_mean': self._load_wakeup_latency,
            'wakeup_latency_quantiles': self._load_wakeup_latency,
            'wakeup_latency_execution_cluster': self._load_wakeup_latency,
            'wakeup_latency_target_cluster': self._load_wakeup_latency,
            'wakeup_latency_cgroup_mean': self._load_wakeup_latency_cgroup,
            'wakeup_latency_cgroup_quantiles': self._load_wakeup_latency_cgroup,
            'tasks_residency_cpu_total_cluster_melt': self._load_tasks_cpu_residency,
            'tasks_residency_total_cluster_melt': self._load_tasks_cpu_residency,
            'tasks_residency_total_melt': self._load_tasks_cpu_residency,
            'cgroup_residency_total_cluster_melt': self._load_cgroup_cpu_residency,
            'cgroup_residency_total_melt': self._load_cgroup_cpu_residency,
            'jb_max_frame_duration': self._load_jankbench,
            'jb_mean_frame_duration': self._load_jankbench,
            'jankbench': self._load_jankbench,
            'jankbench_percs': self._load_jankbench,
            'adpf': self._load_drarm_adpf,
            'adpf_totals': self._load_drarm_adpf,
            'adpf_totals_melt': self._load_drarm_adpf,
            'task_activations_stats_cluster': self._load_task_activations_stats,
            'task_activations_stats_cluster_melt': self._load_task_activations_stats,
            'uclamp_updates': self._load_uclamp_updates,
        }
        return mapping[analysis]

    # TODO: detect missing and call processor?
    def requires_analysis(names: List[str]):
        """
        Decorator function used to make sure the required analysis is loaded before a plotting function is called.

        Usage:
        ```python
        @requires_analysis(['jankbench'])
        def jankbench_frame_durations_hist():
            # (...)
        ```

        :param names: List of analysis metrics that are required for the plot

        """
        def wrapper(func):
            @functools.wraps(func)
            def inner(self, *args, **kwargs):
                if any([d not in self.ana.analysis for d in names]):
                    log.debug(f'{names} not found in analysis, trying to load combined analysis')
                    loader = self.analysis_to_loader(names[0])
                    loader()

                if any([d not in self.ana.analysis for d in names]):
                    log.error(f"Failed to load {names} into analysis using {loader.__name__}")

                return func(self, *args, **kwargs)
            return inner
        return wrapper

    # -------- Results --------

    def results_line(self, metrics: List[str], height=600, width=900, columns=2,
                     title='Benchmark score per-iteration', include_label=True) -> hv.Layout:
        """
        Plot metrics from `WorkloadNotebookAnalysis.results` per-iteration overlaid by tag with
        the specific metrics as facet plots.

        .. note:: This plot uses HoloViews. The resulting figure will be saved to `WorkloadNotebookAnalysis.hv_figures`.

        :param metrics: List of metrics present in the `results` df to plot
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        metrics = sorted(metrics)

        ds = hv.Dataset(
            self.ana.results,
            ['iteration', hv.Dimension('tag', values=self.ana.tags),
             hv.Dimension('metric', values=metrics)], 'value'
        )
        layout = ds.select(metric=metrics).to(hv.Curve, 'iteration', 'value').overlay('tag').opts(
            legend_position='bottom'
        ).layout('metric').opts(shared_axes=False, title=title).cols(columns)
        layout.opts(
            opts.Curve(height=height, width=width),
        )

        self.ana.hv_figures[self.ana._title_to_filename(title, '__line')] = layout
        return layout

    def results_bar(self, metrics: List[str], height=600, width=None, columns=2, percentage=True,
                    title='gmean benchmark score', include_label=True):
        """
        Plot gmean metrics from `WorkloadNotebookAnalysis.results` with the runs as columns and
        the specific metrics as facet plots.

        .. note:: This plot uses plotly.express.
            The resulting figure will be saved to `WorkloadNotebookAnalysis.px_figures`.

        :param metrics: List of metrics present in the `results` df to plot
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        data = pl.from_pandas(self.ana.results).filter(
            pl.col('metric').is_in(metrics)
        ).to_pandas()

        self.ana.summary['results'] = self.gmean_bars(
            data, x='stat', y='value', facet_col='metric', percentage=percentage,
            facet_col_wrap=columns, title=title, width=width, height=height
        )

    # -------- Jankbench --------

    def _load_jankbench(self):
        self.ana.analysis['jankbench'] = pd.concat([wa_output['jankbench'].df for wa_output in self.ana.wa_outputs])
        self.ana.analysis['jankbench']['tag'] = self.ana.analysis['jankbench']['wa_path'].map(trim_wa_path)
        log.info('Loaded jankbench into analysis')

        self.ana.analysis['jb_max_frame_duration'] = self.ana.analysis['jankbench'].query(
            "variable == 'total_duration'"
        )[["tag", "iteration", "value"]].groupby(["tag"]).max().reset_index()
        self.ana.analysis['jb_max_frame_duration']['variable'] = 'max_frame_duration'
        log.info('Loaded jb_max_frame_duration into analysis')

        self.ana.analysis['jb_mean_frame_duration'] = self.ana.analysis['jankbench'].query(
            "variable == 'total_duration'"
        )[["tag", "iteration", "value"]].groupby(
            ["tag", "iteration"]
        ).agg(lambda x: series_mean(x)).reset_index()
        self.ana.analysis['jb_mean_frame_duration']['variable'] = 'mean_frame_duration'
        log.info('Loaded jb_mean_frame_duration into analysis')

        self.ana.analysis['jankbench_percs'] = self.ana.analysis['jankbench'].query("variable == 'jank_frame'").groupby(
            ['tag', 'iteration']
        ).size().reset_index().rename(columns={0: 'count'})
        self.ana.analysis['jankbench_percs']['jank_count'] = self.ana.analysis['jankbench'].query(
            "variable == 'jank_frame' and value == 1.0"
        ).groupby(['tag', 'iteration']).size().reset_index().rename(columns={0: 'count'})['count']
        self.ana.analysis['jankbench_percs']['perc'] = round(
            self.ana.analysis['jankbench_percs']['jank_count'] / self.ana.analysis['jankbench_percs']['count'] * 100, 2
        )
        self.ana.analysis['jankbench_percs']['variable'] = 'jank_percentage'
        log.info('Loaded jankbench_percs into analysis')

    @requires_analysis(['jb_max_frame_duration'])
    def jankbench_max_frame_durations(self, height=600, width=1000, columns=2,
                                      title='Max frame durations', include_label=True):
        """
        Plot max frame durations collected by Jankbench as a bar plot.

        .. note:: This plot uses plotly.express.
            The resulting figure will be saved to `WorkloadNotebookAnalysis.px_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['jb_max_frame_duration'] = self.gmean_bars(
            self.ana.analysis['jb_max_frame_duration'], x='variable', y='value',
            title=title, width=width, height=height
        )

    @requires_analysis(['jb_mean_frame_duration'])
    def jankbench_mean_frame_durations_line(self, height=600, width=1500,
                                            title='Mean frame duration per-iteration', include_label=True):
        """
        Plot mean frame durations collected by Jankbench as a per-iteration line plot.

        .. note:: This plot uses HoloViews.
            The resulting figure will be saved to `WorkloadNotebookAnalysis.hv_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        ds = hv.Dataset(self.ana.analysis['jb_mean_frame_duration'], [
            'iteration', hv.Dimension('tag', values=self.ana.tags)
        ], 'value')
        layout = ds.to(hv.Curve, 'iteration',
                       'value').overlay('tag').opts(legend_position='bottom').opts(shared_axes=False, title=title)
        layout.opts(
            opts.Curve(height=height, width=width, axiswise=True, shared_axes=False),
        )
        self.ana.hv_figures[self.ana._title_to_filename(title, '__line')] = layout
        return layout

    @requires_analysis(['jb_mean_frame_duration'])
    def jankbench_mean_frame_durations_bar(self, height=600, width=1000, percentage=True,
                                           title='gmean frame durations', include_label=True):
        """
        Plot gmean mean frame durations collected by Jankbench as a bar plot.

        .. note:: This plot uses plotly.express.
            The resulting figure will be saved to `WorkloadNotebookAnalysis.px_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['jb_mean_frame_duration'] = self.gmean_bars(
            self.ana.analysis['jb_mean_frame_duration'], x='variable', y='value',
            title=title, width=width, height=height, percentage=percentage,
        )

    @requires_analysis(['jankbench'])
    def jankbench_frame_durations_hist(self, height=800, width=None,
                                       title='Frame duration histogram', include_label=True):
        """
        Plot a histogram of frame durations collected by Jankbench.

        .. note:: This plot uses plotly.express.
            The resulting figure will be saved to `WorkloadNotebookAnalysis.px_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        fig = px.histogram(
            self.ana.analysis['jankbench'].query("variable == 'total_duration'"), x='value',
            color='tag', barmode='group', nbins=40, height=height, width=width, title=title
        )
        self.ana.px_figures[self.ana._title_to_filename(title, "__bar")] = fig
        fig.show(renderer='iframe')

    @requires_analysis(['jankbench'])
    def jankbench_frame_durations_ecdf(self, height=800, width=None,
                                       title='Frame duration ecdf', include_label=True):
        """
        Plot an ecdf of frame durations collected by Jankbench.

        .. note:: This plot uses plotly.express.
            The resulting figure will be saved to `WorkloadNotebookAnalysis.px_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        fig = px.ecdf(
            self.ana.analysis['jankbench'].query("variable == 'total_duration'"),
            x='value', color='tag', height=height, width=width, title=title
        )
        self.ana.px_figures[self.ana._title_to_filename(title, "__bar")] = fig
        fig.show(renderer='iframe')

    @requires_analysis(['jankbench_percs'])
    def jankbench_jank_percentage_line(self, height=600, width=1500,
                                       title='jank percentage per-iteration', include_label=True):
        """
        Plot per-iteration jank percentage collected by Jankbench.

        .. note:: This plot uses HoloViews. The resulting figure will be saved to `WorkloadNotebookAnalysis.hv_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        ds = hv.Dataset(self.ana.analysis['jankbench_percs'],
                        ['iteration', hv.Dimension('tag', values=self.ana.tags)], 'perc')
        layout = ds.to(hv.Curve, 'iteration',
                       'perc').overlay('tag').opts(legend_position='bottom').opts(shared_axes=False, title=title)
        layout.opts(
            opts.Curve(height=height, width=width, axiswise=True, shared_axes=False),
        )
        self.ana.hv_figures[self.ana._title_to_filename(title, '__line')] = layout
        return layout

    @requires_analysis(['jankbench_percs'])
    def jankbench_jank_percentage_bar(self, height=600, width=1000, percentage=True,
                                      title='gmean jank percentage', include_label=True):
        """
        Plot gmean jank percentage collected by Jankbench as a bar plot.

        .. note:: This plot uses plotly.express.
            The resulting figure will be saved to `WorkloadNotebookAnalysis.px_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['jankbench_percs'] = self.gmean_bars(
            self.ana.analysis['jankbench_percs'].rename(columns={'perc': 'value'})[
                ['tag', 'iteration', 'value', 'variable']
            ], x='variable', y='value', title=title, width=width, height=height, percentage=percentage
        )

    def jankbench_metric_line(self, height=600, width=500, columns=3,
                              title='Metric per-iteration', include_label=True):
        """
        Plot per-iteration jankbench-specific metrics.

        .. note:: This plot uses HoloViews. The resulting figure will be saved to `WorkloadNotebookAnalysis.hv_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        ds = hv.Dataset(self.ana.results, [
            'iteration', hv.Dimension('tag', values=self.ana.tags), 'test_name', 'metric'
        ], 'value')
        layout = ds.to(hv.Curve, 'iteration', 'value').overlay('tag').opts(
            legend_position='bottom'
        ).layout('test_name').opts(shared_axes=False, title=title).cols(columns)
        layout.opts(
            opts.Curve(height=height, width=width, axiswise=True, shared_axes=False),
        )
        self.ana.hv_figures[self.ana._title_to_filename(title, '__line')] = layout
        return layout

    def jankbench_jank_percentage_metric_bar(self, height=1000, width=None, columns=4, percentage=True,
                                             title='gmean jank percentage per-metric', include_label=True):
        """
        Plot gmean jank percentage per-metric collected by Jankbench as a bar plot.

        .. note:: This plot uses plotly.express.
            The resulting figure will be saved to `WorkloadNotebookAnalysis.px_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.gmean_bars(
            self.ana.results.query("metric == 'jank_p'"),
            x='stat', y='value', facet_col='test_name', facet_col_wrap=columns,
            title=title, width=width, height=height, percentage=percentage
        )

    def jankbench_mean_duration_metric_bar(self, height=1000, width=None, columns=4, percentage=True,
                                           title='gmean frame duration per-metric', include_label=True):
        """
        Plot gmean mean frame duration per-metric collected by Jankbench as a bar plot.

        .. note:: This plot uses plotly.express.
            The resulting figure will be saved to `WorkloadNotebookAnalysis.px_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.gmean_bars(
            self.ana.results.query("metric == 'mean'"),
            x='stat', y='value', facet_col='test_name', facet_col_wrap=columns,
            title=title, width=width, height=height, percentage=percentage
        )

    # -------- DrArm ADPF --------

    def _load_drarm_adpf(self):
        def preproces_adpf(df):
            return df.reset_index().rename(columns={'index': 'ts'})

        self.ana.load_combined_analysis('adpf.pqt', preprocess=preproces_adpf)
        log.info('Loaded adpf into analysis')
        self.ana.load_combined_analysis('adpf_totals.pqt')
        log.info('Loaded adpf_totals into analysis')
        self.ana.analysis['adpf_totals_melt'] = pd.melt(
            self.ana.analysis['adpf_totals'], id_vars=['iteration', 'tag'],
            value_vars=['average fps', 'frame count']
        )

        self.ana.ds_adpf = hv.Dataset(
            self.ana.analysis['adpf'].reset_index(), ['ts', hv.Dimension('tag', values=self.ana.tags)], [
                'average fps', 'sigma fps', 'thermal status',
                'Adaptive Batching', 'sn_Adaptive Batching', 'Adaptive Decals',
                'sn_Adaptive Decals', 'Adaptive Framerate', 'sn_Adaptive Framerate',
                'Adaptive LOD', 'sn_Adaptive LOD', 'Adaptive Lut', 'sn_Adaptive Lut',
                'Adaptive MSAA', 'sn_Adaptive MSAA', 'Adaptive Resolution',
                'sn_Adaptive Resolution', 'Adaptive Shadow Cascade',
                'sn_Adaptive Shadow Cascade', 'Adaptive Shadow Distance',
                'sn_Adaptive Shadow Distance', 'Adaptive Shadowmap Resolution',
                'sn_Adaptive Shadowmap Resolution', 'Adaptive Shadow Quality',
                'sn_Adaptive Shadow Quality', 'Adaptive Transparency',
                'sn_Adaptive Transparency', 'Adaptive View Distance',
                'sn_Adaptive View Distance', 'Adaptive Sorting', 'sn_Adaptive Sorting',
                'Adaptive Physics', 'sn_Adaptive Physics', 'Adaptive Layer Culling',
                'sn_Adaptive Layer Culling', 'Adaptive Fog', 'sn_Adaptive Fog'
            ]
        )

    @requires_analysis(['adpf_totals_melt'])
    def drarm_adpf_fps_line(self, height=600, width=None,
                            title='FPS & frame count across iterations', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.lines_px(
            self.ana.analysis['adpf_totals_melt'], facet_col='variable', title=title,
            height=height, width=width
        )

    @requires_analysis(['adpf_totals_melt'])
    def drarm_adpf_fps_bar(self, height=600, width=None, percentage=True,
                           title='Gmean iteration FPS & total frames', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['drarm_adpf_fps'] = self.gmean_bars(
            self.ana.analysis['adpf_totals_melt'], x='metric', y='value', facet_col='variable',
            facet_col_wrap=5, title=title, height=height, width=width, include_columns=['variable'],
            table_sort=['variable', 'tag'], percentage=percentage
        )

    @requires_analysis(['adpf_totals_melt'])
    def drarm_adpf_metric_line(self, metrics, height=600, width=1000, title='ADPF', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        layout = hv.Layout([
            self.ana.ds_adpf.to(hv.Curve, 'ts', metrics[0]).overlay('tag').opts(
                legend_position='bottom'
            ).opts(shared_axes=False, title=f"{title} {metric}")
            for metric in metrics
        ])

        layout.opts(
            opts.Curve(height=height, width=width, title=title),
        )
        return layout

    # -------- TLDR --------

    def summary(self, rename_cols: Dict[str, str] = {}) -> pd.DataFrame:
        """
        Print the summary table containing all the relevant summary information from all the plots generated in the
        notebook.

        :param rename_cols: Mapping to optionally rename the workload tags in column headers
        """
        parts: List[pd.DataFrame] = []

        # --- Results ---

        if 'results' in self.ana.summary:
            scores = self.ana.summary['results'].copy()
            scores['perc_diff'] = scores['perc_diff'].apply(lambda s: f"({s})")
            scores['value'] = scores['value'] + " " + scores['perc_diff']
            scores = scores.pivot(values='value', columns='tag', index='metric').reset_index()[
                ['metric'] + self.ana.tags
            ]
            parts.append(scores)

        # --- Jankbench ---

        if 'jb_mean_frame_duration' in self.ana.summary:
            mean_durations = self.ana.summary['jb_mean_frame_duration'].copy()
            mean_durations['perc_diff'] = mean_durations['perc_diff'].apply(lambda s: f"({s})")
            mean_durations['value'] = mean_durations['value'] + " " + mean_durations['perc_diff']
            mean_durations = mean_durations.pivot(
                values='value', columns='tag', index='variable'
            ).reset_index().rename(columns={'variable': 'metric'})[['metric'] + self.ana.tags]
            parts.append(mean_durations)

        if 'jankbench_percs' in self.ana.summary:
            jankbench_percs = self.ana.summary['jankbench_percs'].copy()
            jankbench_percs['perc_diff'] = jankbench_percs['perc_diff'].apply(lambda s: f"({s})")
            jankbench_percs['value'] = jankbench_percs['value'] + " " + jankbench_percs['perc_diff']
            jankbench_percs = jankbench_percs.pivot(
                values='value', columns='tag', index='variable'
            ).reset_index().rename(columns={'variable': 'metric'})[['metric'] + self.ana.tags]
            parts.append(jankbench_percs)

        if 'jb_max_frame_duration' in self.ana.summary:
            summary_max_durations = self.ana.summary['jb_max_frame_duration'].copy()
            summary_max_durations = summary_max_durations.pivot(
                values='value', columns='tag', index='variable'
            ).reset_index().rename(columns={'variable': 'metric'})[['metric'] + self.ana.tags]
            parts.append(summary_max_durations)

        # --- Trace metrics ---

        if 'power_usage' in self.ana.summary:
            power_usage = self.ana.summary['power_usage'].copy().query("channel == 'CPU'")
            power_usage['perc_diff'] = power_usage['perc_diff'].apply(lambda s: f"({s})")
            power_usage['value'] = power_usage['value'] + " " + power_usage['perc_diff']
            power_usage['channel'] = 'CPU_total_power'
            power_usage = power_usage.pivot(values='value', columns='tag', index='channel').reset_index().rename(
                columns={'channel': 'metric'}
            )[['metric'] + self.ana.tags]
            parts.append(power_usage)

        if 'overutilized_mean' in self.ana.analysis:
            ou = self.ana.analysis['overutilized_mean'].copy()
            ou['percentage'] = ou['percentage'].apply(lambda x: f"{x}%")
            ou = ou.pivot(values='percentage', columns='tag', index='metric').reset_index()[
                ['metric'] + self.ana.tags
            ]
            parts.append(ou)

        if 'thermal' in self.ana.summary:
            thermal = self.ana.summary['thermal'].copy()
            thermal['perc_diff'] = thermal['perc_diff'].apply(lambda s: f"({s})")
            thermal['value'] = thermal['value'] + " " + thermal['perc_diff']
            thermal = thermal.pivot(values='value', columns='tag', index='cluster').reset_index().rename(
                columns={'cluster': 'metric'}
            )[['metric'] + self.ana.tags]
            thermal['metric'] = "thermal (" + thermal['metric'] + ")"
            parts.append(thermal)

        if 'wakeup_latency' in self.ana.summary:
            wakeup_latency = self.ana.summary['wakeup_latency'].copy()
            wakeup_latency['perc_diff'] = wakeup_latency['perc_diff'].apply(lambda s: f"({s})")
            wakeup_latency['comm'] = "latency (" + wakeup_latency['comm'] + ")"
            wakeup_latency['value'] = wakeup_latency['value'] + " " + wakeup_latency['perc_diff']
            wakeup_latency = wakeup_latency.pivot(values='value', columns='tag', index='comm').reset_index().rename(
                columns={'comm': 'metric'}
            )[['metric'] + self.ana.tags]
            parts.append(wakeup_latency)

        if 'tasks_cpu_residency_per_task' in self.ana.summary:
            task_cpu_res = self.ana.summary['tasks_cpu_residency_per_task'].copy().query("cluster == 'total'")
            task_cpu_res['perc_diff'] = task_cpu_res['perc_diff'].apply(lambda s: f"({s})")
            task_cpu_res['value'] = task_cpu_res['value'] + " " + task_cpu_res['perc_diff']
            task_cpu_res = task_cpu_res.pivot(values='value', columns='tag', index='comm').reset_index().rename(
                columns={'comm': 'metric'}
            )[['metric'] + self.ana.tags]
            task_cpu_res['metric'] = "CPU residency (" + task_cpu_res['metric'] + ")"
            parts.append(task_cpu_res)

        summary = pd.concat(parts).reset_index(drop=True).rename(columns=rename_cols)

        print(self.ana.label)
        print_table(summary)

        return summary

    # -------- Power Meter (pixel6_emeter) --------

    def _load_power_meter(self):
        def postprocess_pixel6_emeter_means(df):
            df_total = df.groupby(['tag', 'kernel', 'iteration']).sum(numeric_only=True).reset_index()
            df_total['channel'] = 'Total'

            df_cpu_total = df.query("channel.str.startswith('CPU')").groupby(
                ['tag', 'kernel', 'iteration']
            ).sum(numeric_only=True).reset_index()
            df_cpu_total['channel'] = 'CPU'
            return pd.concat([df, df_cpu_total, df_total])[['tag', 'kernel', 'iteration', 'channel', 'power']]

        self.ana.load_combined_analysis('pixel6_emeter.pqt')
        log.info('Loaded pixel6_emeter into analysis')
        self.ana.load_combined_analysis('pixel6_emeter_mean.pqt', postprocess=postprocess_pixel6_emeter_means)
        log.info('Loaded pixel6_emeter_mean into analysis')

    @requires_analysis(['pixel6_emeter_mean'])
    def power_meter_line(self, height=1000, width=None,
                         title='Mean power usage across iterations [mW]', include_label=True):
        """
        Plot mean power meter readings as a per-iteration line plot.

        .. note:: This plot uses plotly.express.
            The resulting figure will be saved to `WorkloadNotebookAnalysis.px_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.lines_px(
            self.ana.analysis['pixel6_emeter_mean'], y='power', facet_col='channel',
            facet_col_wrap=3, height=height, width=width, title=title
        )

    @requires_analysis(['pixel6_emeter_mean'])
    def power_meter_bar(self, height=600, width=None, channels: Optional[List[str]] = None, percentage: bool = True,
                        title: str = 'Gmean power usage [mW]', include_label: bool = True):
        """
        Plot gmean power meter readings as a bar plot.

        .. note:: This plot uses plotly.express.
            The resulting figure will be saved to `WorkloadNotebookAnalysis.px_figures`.

        :param channels: List of power meter channels to include in the plot
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        data = pl.from_pandas(self.ana.analysis['pixel6_emeter_mean']).rename({'power': 'value'})
        channels = channels if channels else list(data['channel'].unique())
        data = data.filter(pl.col('channel').is_in(channels))

        self.ana.summary['power_usage'] = self.gmean_bars(
            data.to_pandas(), x='channel', y='value', facet_col='metric', facet_col_wrap=5, percentage=percentage,
            title=title, height=height, width=width, include_total=True, include_columns=['channel']
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

    @requires_analysis(['overutilized', 'overutilized_mean'])
    def overutilized_line(self, height=600, width=None,
                          title='Overutilized percentage per-iteration', include_label=True):
        """
        Plot overutilized time percentage as a per-iteration line plot.

        .. note:: This plot uses plotly.express.
            The resulting figure will be saved to `WorkloadNotebookAnalysis.px_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        print_table(self.ana.analysis['overutilized_mean'][['metric', 'tag', 'time', 'total_time', 'percentage']])
        self.lines_px(self.ana.analysis['overutilized'], y='percentage',
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

    @requires_analysis(['freqs_mean'])
    def frequency_line(self, height=600, width=None,
                       title='Mean cluster frequency across iterations', include_label=True):
        """
        Plot the mean cluster frequency as a per-iteration line plot.

        .. note:: This plot uses plotly.express.
            The resulting figure will be saved to `WorkloadNotebookAnalysis.px_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.lines_px(self.ana.analysis['freqs_mean'], facet_col='cluster',
                               facet_col_wrap=3, title=title, height=height, width=width)

    @requires_analysis(['freqs_mean'])
    def frequency_bar(self, height=600, width=None, include_label=True, percentage=True,
                      title='Gmean frequency per cluster'):
        """
        Plot the gmean cluster frequency as a bar plot.

        .. note:: This plot uses plotly.express.
            The resulting figure will be saved to `WorkloadNotebookAnalysis.px_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['frequency'] = self.gmean_bars(
            self.ana.analysis['freqs_mean'], x='metric', y='value', facet_col='cluster', facet_col_wrap=3,
            title=title, width=width, height=height, order_cluster=True, percentage=percentage,
            include_columns=['cluster']
        )

    # -------- Thermal --------

    def _load_thermal(self):
        def preprocess_thermal(df):
            return df.groupby(['iteration', 'kernel', 'wa_path']).mean().reset_index()

        def postprocess_thermal(df):
            for col in [c for c in df.columns if c not in ['time', 'iteration', 'kernel', 'tag', 'wa_path']]:
                df[col] = df[col] / 1000
            df = round(df, 2)
            return df

        self.ana.load_combined_analysis('thermal.pqt', preprocess=preprocess_thermal, postprocess=postprocess_thermal)
        log.info('Loaded thermal into analysis')
        self.ana.analysis['thermal_melt'] = pd.melt(self.ana.analysis['thermal'],
                                                    id_vars=['iteration', 'tag', 'kernel'],
                                                    value_vars=['little', 'mid', 'big']
                                                    ).rename(columns={'variable': 'cluster'})
        log.info('Loaded thermal_melt into analysis')

    @requires_analysis(['thermal_melt'])
    def thermal_line(self, height=600, width=None,
                     title='Mean cluster temperature across iterations', include_label=True):
        """
        Plot the mean thermal zone temperature as a per-iteration bar plot.

        .. note:: This plot uses plotly.express.
            The resulting figure will be saved to `WorkloadNotebookAnalysis.px_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.lines_px(self.ana.analysis['thermal_melt'], facet_col='cluster', height=height, width=width,
                               facet_col_wrap=3, title=title)

    @requires_analysis(['thermal_melt'])
    def thermal_bar(self, height=600, width=None, percentage=True,
                    title='Gmean temperature', include_label=True):
        """
        Plot the gmean thermal zone temperature as a bar plot.

        .. note:: This plot uses plotly.express.
            The resulting figure will be saved to `WorkloadNotebookAnalysis.px_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['thermal'] = self.gmean_bars(
            self.ana.analysis['thermal_melt'], x='cluster', y='value',
            facet_col='metric', facet_col_wrap=2, title=title,
            width=width, height=height, order_cluster=True, include_columns=['cluster'], percentage=percentage
        )

    # -------- Perf --------

    def perf_line(self, counters=None, height=340, width=600,
                  title='Perf counters across iterations', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        if not counters:
            counters = self.ana.config['notebook']['perf_counters'].get()

        ds = hv.Dataset(self.ana.results_perf, [
            'iteration', hv.Dimension('tag', values=self.ana.tags), hv.Dimension('metric', values=counters)
        ], 'value')
        layout = ds.select(metric=counters).to(hv.Curve, 'iteration', 'value').overlay('tag').opts(
            legend_position='bottom'
        ).layout('metric').opts(shared_axes=False, title=title).cols(3)
        layout.opts(
            opts.Curve(width=width, height=height),
            opts.Overlay(legend_position='bottom'),
        )

        return layout

    def perf_bar(self, counters=None, height=900, width=None, percentage=True,
                 title='Gmean perf counters', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        if not counters:
            counters = self.ana.config['notebook']['perf_counters'].get()

        self.gmean_bars(
            self.ana.results_perf.query("metric in @counters")[['kernel', 'tag', 'iteration', 'metric', 'value']],
            x='stat', y='value', facet_col='metric', facet_col_wrap=5, title=title, width=width, height=height,
            percentage=percentage
        )

    # -------- Idle --------

    def _load_idle_residency(self):
        self.ana.load_combined_analysis('idle_residency.pqt')
        self.ana.analysis['idle_residency'] = self.ana.analysis['idle_residency'].groupby(
            ['tag', 'cluster', 'idle_state'], sort=False
        ).mean(numeric_only=True).reset_index()[['tag', 'cluster', 'idle_state', 'time']]
        self.ana.analysis['idle_residency']['time'] = round(self.ana.analysis['idle_residency']['time'], 2)
        log.info('Loaded idle_residency into analysis')

    @requires_analysis(['idle_residency'])
    def idle_residency_bar(self, height=600, width=None, title='Idle state residencies', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['idle_resdiency'] = self.ana.analysis['idle_residency']
        print_table(self.ana.analysis['idle_residency'])
        fig = px.bar(
            self.ana.analysis['idle_residency'], x='idle_state', y='time', color='tag',
            facet_col='cluster', barmode='group', text=self.ana.analysis['idle_residency']['time'],
            width=width, height=height, title=title
        )
        fig.update_traces(textposition='outside')
        fig.show(renderer='iframe')

    def _load_idle_miss(self):
        def preprocess_cpu_idle_miss_df(df):
            df = df.groupby(['wa_path', 'kernel', 'cluster', 'below']).sum().reset_index()
            wa_path = df['wa_path'].iloc[0]
            if not wa_path:
                return df
            wakeup_count = len(self.ana.analysis['cpu_idle'].query("wa_path == @wa_path and state == -1"))
            df['count_perc'] = round(df['count'] / wakeup_count * 100, 3)
            return df

        def postprocess_cpu_idle_miss_df(df):
            df['type'] = df['below'].replace(0, 'too deep').replace(1, 'too shallow')
            df['order'] = df['cluster'].replace('little', 0).replace('mid', 1).replace('big', 2)
            df = df.sort_values(by=['tag', 'kernel', 'order', 'type'])
            return df

        self.ana.load_combined_analysis('cpu_idle.pqt')
        log.info('Loaded cpu_idle into analysis')
        self.ana.load_combined_analysis(
            'cpu_idle_miss_counts.pqt', preprocess=preprocess_cpu_idle_miss_df, postprocess=postprocess_cpu_idle_miss_df
        )
        log.info('Loaded cpu_idle_miss_counts into analysis')

    @requires_analysis(['cpu_idle_miss_counts'])
    def idle_miss_bar(self, height=600, width=None,
                      title='CPUIdle misses as percentage of all wakeups', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['idle_miss'] = self.ana.analysis['cpu_idle_miss_counts'].groupby(
            ['tag', 'type']
        ).sum().reset_index()[['tag', 'type', 'count_perc']]

        print_table(self.ana.summary['idle_miss'])
        fig = px.bar(
            self.ana.analysis['cpu_idle_miss_counts'], x='type', y='count_perc', color='tag',
            facet_col='cluster', barmode='group', text=self.ana.analysis['cpu_idle_miss_counts']['count_perc'],
            width=width, height=height, title=title
        )
        fig.show(renderer='iframe')

    # -------- Energy Estimate --------

    def _load_energy_estimate(self):
        self.ana.load_combined_analysis('energy_estimate_mean.pqt')
        log.info('Loaded energy_estimate_mean into analysis')
        self.ana.analysis['energy_estimate_melt'] = pd.melt(
            self.ana.analysis['energy_estimate_mean'], id_vars=['iteration', 'tag'],
            value_vars=['little', 'mid', 'big', 'total']
        ).rename(columns={'variable': 'cluster'})
        log.info('Loaded energy_estimate_melt into analysis')

    @requires_analysis(['energy_estimate_melt'])
    def energy_estimate_line(self, height=1000, width=None,
                             title='Mean energy estimate across iterations [bW]', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.lines_px(
            self.ana.analysis['energy_estimate_melt'], facet_col='cluster',
            height=height, width=width, title=title
        )

    @requires_analysis(['energy_estimate_melt'])
    def energy_estimate_bar(self, height=600, width=None, percentage=True,
                            title='Gmean energy estimate [bW]', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['energy_estimate'] = self.gmean_bars(
            self.ana.analysis['energy_estimate_melt'], x='cluster', y='value', facet_col='metric',
            facet_col_wrap=5, title=title, width=width, height=height,
            include_columns=['cluster'], order_cluster=True, include_total=True, percentage=percentage
        )

    # -------- CFS signals --------

    def _load_sched_pelt_cfs(self):
        self.ana.load_combined_analysis('sched_pelt_cfs_mean.pqt')
        log.info('Loaded sched_pelt_cfs_mean into analysis')
        self.ana.analysis['sched_pelt_cfs_melt'] = pd.melt(
            self.ana.analysis['sched_pelt_cfs_mean'],
            id_vars=['iteration', 'tag', 'kernel', 'cluster'], value_vars=['util', 'load']
        )
        log.info('Loaded sched_pelt_cfs_melt into analysis')

    @requires_analysis(['sched_pelt_cfs_mean'])
    def sched_pelt_cfs_line(self, height=400, width=700,
                            title='Mean cluster', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        signals = ['util', 'load']
        ds = hv.Dataset(
            self.ana.analysis['sched_pelt_cfs_mean'],
            ['iteration', hv.Dimension('tag', values=self.ana.tags),
             hv.Dimension('cluster', values=self.ana.CLUSTERS)],
            signals
        )
        layout = hv.Layout([
            ds.to(hv.Curve, 'iteration', signal).overlay('tag').opts(legend_position='bottom').layout(
                'cluster'
            ).opts(title=f"{title} {signal}", framewise=True)
            for signal in signals
        ]).cols(1)
        layout.opts(
            opts.Curve(width=width, height=height, framewise=True),
        )
        return layout

    @requires_analysis(['sched_pelt_cfs_mean'])
    def sched_pelt_cfs_bar(self, height=1000, width=None, percentage=True,
                           title='Gmean cfs signals', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['cfs_signals'] = self.gmean_bars(
            self.ana.analysis['sched_pelt_cfs_melt'], x='cluster', y='value',
            facet_col='variable', facet_col_wrap=1, title=title, percentage=percentage,
            width=width, height=height, order_cluster=True, include_columns=['cluster'],
        )

    # -------- Wakeup latency --------

    def _load_wakeup_latency(self):
        def postprocess_wakeup_latency_mean(df):
            df = df.rename(columns={'wakeup_latency': 'value'})
            df['order'] = df['tag'].map(lambda x: self.ana.tags.index(x))
            df['unit'] = 'x'
            return df

        self.ana.load_combined_analysis('wakeup_latency_mean.pqt', postprocess=postprocess_wakeup_latency_mean)
        log.info('Loaded wakeup_latency_mean into analysis')

        def postprocess_wakeup_latency(df):
            df['order'] = df['tag'].map(lambda x: self.ana.tags.index(x))
            df['cluster'] = df['cpu'].copy().apply(cpu_cluster)
            df['order_cluster'] = df['cluster'].map(lambda x: self.ana.CLUSTERS.index(x))
            df['target_cluster'] = df['target_cpu'].copy().apply(cpu_cluster)
            df['order_target_cluster'] = df['target_cluster'].map(lambda x: self.ana.CLUSTERS.index(x))
            return df

        self.ana.load_combined_analysis('wakeup_latency.pqt', postprocess=postprocess_wakeup_latency)
        log.info('Loaded wakeup_latency into analysis')

        self.ana.analysis['wakeup_latency_quantiles'] = self.ana.analysis['wakeup_latency'].groupby([
            'comm', 'tag', 'iteration'
        ]).quantile([0.9, 0.95, 0.99], numeric_only=True).reset_index()[
            ['comm', 'tag', 'level_3', 'iteration', 'wakeup_latency', 'order']
        ].rename(columns={'level_3': 'quantile'}).sort_values(by=['comm', 'order'])
        log.info('Loaded wakeup_latency_quantiles into analysis')

        self.ana.analysis['wakeup_latency_execution_cluster'] = self.ana.analysis['wakeup_latency'].groupby([
            'comm', 'tag', 'cluster'
        ]).mean(numeric_only=True).reset_index().sort_values(by=['comm', 'order_cluster', 'order'])[
            ['comm', 'tag', 'cluster', 'wakeup_latency']
        ]
        log.info('Loaded wakeup_latency_execution_cluster into analysis')

        self.ana.analysis['wakeup_latency_target_cluster'] = self.ana.analysis['wakeup_latency'].groupby([
            'comm', 'tag', 'target_cluster'
        ]).mean(numeric_only=True).reset_index().sort_values(by=['comm', 'order_target_cluster', 'order'])[
            ['comm', 'tag', 'target_cluster', 'wakeup_latency']
        ]
        log.info('Loaded wakeup_latency_target_cluster into analysis')

    @requires_analysis(['wakeup_latency_mean'])
    def wakeup_latency_line(self, height=600, width=None, columns=3,
                            title='Task wakeup latencies across iterations', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.lines_px(
            self.ana.analysis['wakeup_latency_mean'], facet_col='comm',
            facet_col_wrap=columns, height=height, width=width, title=title
        )

    @requires_analysis(['wakeup_latency_mean'])
    def wakeup_latency_bar(self, height=600, width=None, columns=3, percentage=True,
                           title='Gmean task wakeup latency', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['wakeup_latency'] = self.gmean_bars(
            self.ana.analysis['wakeup_latency_mean'], x='metric', y='value', facet_col='comm', facet_col_wrap=columns,
            title=title, table_sort=['comm', 'tag'], gmean_round=0, width=width, height=height, percentage=percentage
        )

    @requires_analysis(['wakeup_latency_quantiles'])
    def wakeup_latency_quantiles_bar(self, height=1300, width=None, columns=1, percentage=True,
                                     title='Gmean latency quantile', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['wakeup_latency_quantiles'] = self.gmean_bars(
            self.ana.analysis['wakeup_latency_quantiles'].rename(columns={'wakeup_latency': 'value'}),
            x='quantile', y='value', facet_col='comm', facet_col_wrap=columns, title=title, percentage=percentage,
            width=width, height=height, include_columns=['quantile'], table_sort=['quantile', 'comm'], gmean_round=0,
        )

    @requires_analysis(['wakeup_latency_execution_cluster'])
    def wakeup_latency_execution_cluster_bar(self, height=1300, width=None, include_label=True,
                                             title='Mean task wakeup latency per execution cluster'):
        if include_label:
            title = f"{self.ana.label} - {title}"

        fig = px.bar(
            self.ana.analysis['wakeup_latency_execution_cluster'], x='cluster', y='wakeup_latency', color='tag',
            facet_col='comm', barmode='group', facet_col_wrap=1, width=width, height=height, title=title,
            text=self.ana.analysis['wakeup_latency_execution_cluster']['wakeup_latency'].apply(trim_number),
        )
        fig.update_traces(textposition='outside')
        self.ana.analysis['wakeup_latency_execution_cluster']['wakeup_latency'] = self.ana.analysis[
            'wakeup_latency_execution_cluster'
        ]['wakeup_latency'].apply(trim_number)
        print_table(self.ana.analysis['wakeup_latency_execution_cluster'])
        fig.show(renderer='iframe')

    @requires_analysis(['wakeup_latency_target_cluster'])
    def wakeup_latency_target_cluster_bar(self, height=1300, width=None, include_label=True,
                                          title='Mean task wakeup latency per target cluster'):
        if include_label:
            title = f"{self.ana.label} - {title}"

        fig = px.bar(
            self.ana.analysis['wakeup_latency_target_cluster'], x='target_cluster', y='wakeup_latency', color='tag',
            facet_col='comm', barmode='group', facet_col_wrap=1, width=width, height=height, title=title,
            text=self.ana.analysis['wakeup_latency_target_cluster']['wakeup_latency'].apply(trim_number),
        )
        fig.update_traces(textposition='outside')
        self.ana.analysis['wakeup_latency_target_cluster']['wakeup_latency'] = self.ana.analysis[
            'wakeup_latency_target_cluster'
        ]['wakeup_latency'].apply(trim_number)
        print_table(self.ana.analysis['wakeup_latency_target_cluster'])
        fig.show(renderer='iframe')

    # -------- Wakeup latency - cgroup --------

    def _load_wakeup_latency_cgroup(self):
        def postprocess_cgroup_latency(df):
            df = df.rename(columns={'wakeup_latency': 'value'})
            df['order'] = df['tag'].map(lambda x: self.ana.tags.index(x))
            return df

        self.ana.load_combined_analysis('wakeup_latency_cgroup.pqt', postprocess=postprocess_cgroup_latency)
        log.info('Loaded wakeup_latency_cgroup into analysis')

        self.ana.analysis['wakeup_latency_cgroup_mean'] = self.ana.analysis['wakeup_latency_cgroup'].groupby(
            ["tag", "cgroup", "iteration", "order"]
        ).mean(numeric_only=True).reset_index().sort_values(by=["order", "cgroup", "iteration"])[
            ['tag', 'cgroup', 'iteration', 'value', 'order']
        ]
        log.info('Loaded wakeup_latency_cgroup_mean into analysis')

        self.ana.analysis['wakeup_latency_cgroup_quantiles'] = self.ana.analysis['wakeup_latency_cgroup'].groupby(
            ['cgroup', 'tag', 'iteration']
        ).quantile([0.9, 0.95, 0.99], numeric_only=True).reset_index()[
            ['cgroup', 'tag', 'level_3', 'iteration', 'value', 'order']
        ].rename(columns={'level_3': 'quantile'}).sort_values(by=['cgroup', 'order'])
        log.info('Loaded wakeup_latency_cgroup_quantiles into analysis')

    @requires_analysis(['wakeup_latency_cgroup_mean'])
    def wakeup_latency_cgroup_line(self, height=600, width=None,
                                   title='cgroup wakeup latencies across iterations', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.lines_px(
            self.ana.analysis['wakeup_latency_cgroup_mean'], facet_col='cgroup', facet_col_wrap=3,
            height=height, width=width, title=title
        )

    @requires_analysis(['wakeup_latency_cgroup_mean'])
    def wakeup_latency_cgroup_bar(self, height=600, width=None, percentage=True,
                                  title='Gmean task wakeup latency per-cgroup', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['wakeup_latency_cgroup'] = self.gmean_bars(
            self.ana.analysis['wakeup_latency_cgroup_mean'], x='metric', y='value', facet_col='cgroup',
            title=title, include_columns=['cgroup'], table_sort=['cgroup'], gmean_round=0, width=width, height=height,
            percentage=percentage
        )

    @requires_analysis(['wakeup_latency_cgroup_quantiles'])
    def wakeup_latency_cgroup_quantiles_bar(self, height=1400, width=None, percentage=True,
                                            title='Gmean latency quantile per-cgroup', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['wakeup_latency_cgroup_quantiles'] = self.gmean_bars(
            self.ana.analysis['wakeup_latency_cgroup_quantiles'], x='quantile', y='value', facet_col='cgroup',
            facet_col_wrap=1, title=title, include_columns=['cgroup', 'quantile'], percentage=percentage,
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
            id_vars=['iteration', 'tag', 'kernel'], value_vars=cpus_prefix
        ).rename(columns={'variable': 'cpu'}).sort_values(['tag', 'kernel', 'iteration', 'cpu'])
        log.info('Loaded tasks_residency_cpu_total_melt into analysis')

        # TODO: sort by order of clusters
        self.ana.analysis['tasks_residency_cpu_total_cluster_melt'] = pd.melt(
            self.ana.analysis['tasks_residency_cpu_total'], id_vars=['iteration', 'tag', 'kernel'],
            value_vars=self.ana.CLUSTERS_TOTAL
        ).rename(columns={'variable': 'cluster'}).sort_values(['tag', 'kernel', 'iteration', 'cluster'])
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
            self.ana.analysis['tasks_residency_total'], id_vars=['iteration', 'tag', 'kernel', 'comm'],
            value_vars=cpus_prefix
        ).rename(columns={'variable': 'cpu'}).sort_values(['tag', 'kernel', 'iteration', 'cpu'])
        log.info('Loaded tasks_residency_total_melt into analysis')

        self.ana.analysis['tasks_residency_total_cluster_melt'] = pd.melt(
            self.ana.analysis['tasks_residency_total'], id_vars=['iteration', 'tag', 'kernel', 'comm'],
            value_vars=self.ana.CLUSTERS_TOTAL
        ).rename(columns={'variable': 'cluster'}).sort_values(['tag', 'kernel', 'iteration', 'cluster'])
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
            self.ana.analysis['tasks_residency'], id_vars=['iteration', 'tag', 'kernel', 'comm'],
            value_vars=self.ana.CLUSTERS_TOTAL
        ).rename(columns={'variable': 'cluster'})
        log.info('Loaded tasks_residency_cluster_melt into analysis')

    @requires_analysis(['tasks_residency_cpu_total_cluster_melt'])
    def tasks_cpu_residency_cluster_line(self, height=600, width=None, columns=4,
                                         title='Mean cluster CPU residency', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.lines_px(
            self.ana.analysis['tasks_residency_cpu_total_cluster_melt'], facet_col='cluster',
            title=title, height=height, width=width, facet_col_wrap=columns
        )

    @requires_analysis(['tasks_residency_cpu_total_cluster_melt'])
    def tasks_cpu_residency_cluster_bar(self, height=800, width=None, columns=1, percentage=True,
                                        title='Gmean cluster CPU residency', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['tasks_cpu_residency_cluster'] = self.gmean_bars(
            self.ana.analysis['tasks_residency_cpu_total_cluster_melt'], x='cluster', y='value', facet_col='metric',
            facet_col_wrap=columns, title=title, include_columns=['cluster'], height=height, width=width,
            order_cluster=True, include_total=True, percentage=percentage
        )

    @requires_analysis(['tasks_residency_total_cluster_melt'])
    def tasks_cpu_residency_per_task_bar(self, height=1200, width=None, columns=1, percentage=True,
                                         title='Gmean cluster per-task CPU residency', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['tasks_cpu_residency_per_task'] = self.gmean_bars(
            self.ana.analysis['tasks_residency_total_cluster_melt'], x='cluster', y='value', facet_col='comm',
            facet_col_wrap=columns, title=title, include_columns=['cluster'], height=height,
            width=width, order_cluster=True, include_total=True, percentage=percentage
        )

    # TODO: CPUs line plot
    @requires_analysis(['tasks_residency_total_melt'])
    def tasks_cpu_residency_cpu_bar(self, height=1400, width=None, columns=1, percentage=True,
                                    title='Gmean CPU residency', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['tasks_cpu_residency_cpus'] = self.gmean_bars(
            self.ana.analysis['tasks_residency_total_melt'], x='cpu', y='value', facet_col='comm',
            facet_col_wrap=columns, title=title, width=width, height=height, percentage=percentage
        )

    # -------- cgroup CPU residency --------

    def _load_cgroup_cpu_residency(self):
        def postprocess_cgroup_task_residency(df):
            df = df.rename(columns={'Total': 'total'})[
                ['tag', 'cgroup', 'iteration', 'total', 'little', 'mid', 'big'] + self.ana.CPUS
            ]
            return df

        self.ana.load_combined_analysis('tasks_residency_cgroup_total.pqt',
                                        postprocess=postprocess_cgroup_task_residency)
        log.info('Loaded tasks_residency_cgroup_total into analysis')

        self.ana.analysis['cgroup_residency_total_melt'] = pd.melt(
            self.ana.analysis['tasks_residency_cgroup_total'], id_vars=['iteration', 'tag', 'cgroup'],
            value_vars=self.ana.CPUS
        ).rename(columns={'variable': 'cpu'})
        log.info('Loaded cgroup_residency_total_melt into analysis')

        self.ana.analysis['cgroup_residency_total_cluster_melt'] = pd.melt(
            self.ana.analysis['tasks_residency_cgroup_total'], id_vars=['iteration', 'tag', 'cgroup'],
            value_vars=self.ana.CLUSTERS_TOTAL
        ).rename(columns={'variable': 'cluster'})
        log.info('Loaded cgroup_residency_total_cluster_melt into analysis')

    @requires_analysis(['cgroup_residency_total_cluster_melt'])
    def cgroup_cpu_residency_cluster_bar(self, height=1100, width=None, percentage=True,
                                         title='Gmean cluster CPU residency per-cgroup', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['cgroup_cpu_residency_cluster'] = self.gmean_bars(
            self.ana.analysis['cgroup_residency_total_cluster_melt'], x='cluster', y='value', facet_col='cgroup',
            facet_col_wrap=1, title=title, width=width, height=height,
            include_columns=['cgroup', 'cluster'], table_sort=['cgroup', 'cluster'],
            order_cluster=True, include_total=True, percentage=percentage
        )

    @requires_analysis(['cgroup_residency_total_melt'])
    def cgroup_cpu_residency_cpu_bar(self, height=1100, width=None, percentage=True,
                                     title='Gmean cgroup CPU residency', include_label=True):
        if include_label:
            title = f"{self.ana.label} - {title}"

        self.ana.summary['cgroup_cpu_residency_cpu'] = self.gmean_bars(
            self.ana.analysis['cgroup_residency_total_melt'], x='cpu', y='value', facet_col='cgroup',
            facet_col_wrap=1, title='', width=width, height=height, include_columns=['cgroup', 'cpu'],
            percentage=percentage
        )

    # -------- Task placement (activations) --------

    def _load_task_activations_stats(self):
        # select all tasks specified in the config by default

        self.ana.load_combined_analysis('task_activations_stats_cluster.pqt')
        log.info('Loaded task_activations_stats_cluster into analysis')
        self.ana.analysis['task_activations_stats_cluster_melt'] = pd.melt(
            self.ana.analysis['task_activations_stats_cluster'],
            id_vars=['kernel', 'tag', 'iteration', 'cluster', 'comm'], value_vars=['count', 'duration']
        )
        log.info('Loaded task_activations_stats_cluster_melt into analysis')

    @requires_analysis(['task_activations_stats_cluster'])
    def task_activations_stats_count_line(self, tasks=None, height=500, width=None, include_label=True,
                                          title='Activation counts of {} per cluster across iterations'):
        if include_label:
            title = f"{self.ana.label} - {title}"

        if not tasks:
            tasks = self.ana.config['processor']['important_tasks'][self.ana.workload_label].get()
        data = self.ana.analysis['task_activations_stats_cluster'].query("comm in @tasks")

        for task, task_df in data.groupby('comm'):
            self.lines_px(
                task_df, x='iteration', y='count', color='tag', facet_col='cluster',
                facet_col_wrap=3, height=height, width=width, scale_y=True, title=title.format(task)
            )

    @requires_analysis(['task_activations_stats_cluster'])
    def task_activations_stats_duration_line(self, tasks=None, height=500, width=None, include_label=True,
                                             title='Activation durations of {} per cluster across iterations'):
        if include_label:
            title = f"{self.ana.label} - {title}"

        if not tasks:
            tasks = self.ana.config['processor']['important_tasks'][self.ana.workload_label].get()
        data = self.ana.analysis['task_activations_stats_cluster'].query("comm in @tasks")

        for task, task_df in data.groupby('comm'):
            self.lines_px(
                task_df, x='iteration', y='count', color='tag', facet_col='cluster',
                facet_col_wrap=3, height=height, width=width, scale_y=True, title=title.format(task)
            )

    @requires_analysis(['task_activations_stats_cluster_melt'])
    def task_activations_stats_count_bar(self, tasks=None, height=1000, width=None, include_label=True,
                                         title='Gmean task activation counts'):
        if include_label:
            title = f"{self.ana.label} - {title}"

        if not tasks:
            tasks = self.ana.config['processor']['important_tasks'][self.ana.workload_label].get()
        data = self.ana.analysis['task_activations_stats_cluster_melt'].query(
            "comm in @tasks and variable == 'count'"
        ).reset_index(drop=True)

        self.ana.summary['activations_stats_count'] = self.gmean_bars(
            data, x='cluster', facet_col='comm', facet_col_wrap=3, title=title, percentage=False,
            height=height, width=width, include_columns=['cluster'], order_cluster=True,
        )

    @requires_analysis(['task_activations_stats_cluster_melt'])
    def task_activations_stats_duration_bar(self, tasks=None, height=1000, width=None, include_label=True,
                                            title='Gmean task activation duration'):
        if include_label:
            title = f"{self.ana.label} - {title}"

        if not tasks:
            tasks = self.ana.config['processor']['important_tasks'][self.ana.workload_label].get()
        data = self.ana.analysis['task_activations_stats_cluster_melt'].query(
            "comm in @tasks and variable == 'duration'"
        ).reset_index(drop=True)

        self.ana.summary['activations_stats_duration'] = self.gmean_bars(
            data, x='cluster', facet_col='comm', facet_col_wrap=3, title=title,
            height=height, width=width, include_columns=['cluster'], order_cluster=True, percentage=False
        )

    def task_activations_detailed(self, tag_a: str, tag_b: str, iteration: int, comm: str, columns: int = 1,
                                  include_label: bool = True):
        """
        Plot detailed task activations in kernelshark-style from `tag_a` and `tag_b` in one layout.

        :param tag_a: The first workload tag to plot data from
        :param tag_b: The second workload tag to plot data from
        :param iteration: Iteration to plot data from
        :param comm: Comm of the task to plot data from
        :param columns: Number of columns to display the two plots in
        """
        title = f'{comm} activations in iteration {iteration}'
        if include_label:
            title = f"{self.ana.label} - {title}"

        plot_a = self.ana.traces[tag_a][iteration].ana.tasks.plot_tasks_activation(
            self.ana.traces[tag_a][iteration].get_task_ids(comm)
        ).opts(title=f"{title} of {tag_a}")

        plot_b = self.ana.traces[tag_b][iteration].ana.tasks.plot_tasks_activation(
            self.ana.traces[tag_b][iteration].get_task_ids(comm)
        ).opts(title=f"{title} of {tag_b}")

        return (plot_a + plot_b).cols(columns).opts(shared_axes=False)

    # -------- uclamp updates --------

    def _load_uclamp_updates(self):
        def postprocess_uclamp_updates(df):
            return df.query("task != '<unknown>'")

        self.ana.load_combined_analysis('uclamp_updates.pqt',
                                        postprocess=postprocess_uclamp_updates, allow_missing=True)
        log.info('Loaded uclamp_updates into analysis')

    @requires_analysis(['uclamp_updates'])
    def uclamp_per_task_line(self, height=600, width=1600, include_label=True,
                             title='Per-task uclamp over time'):
        """
        Plot the uclamp updates for each task across iterations.

        .. note:: This plot uses HoloViews. The resulting figure will be saved to `WorkloadNotebookAnalysis.hv_figures`.
        """
        if include_label:
            title = f"{self.ana.label} - {title}"

        ds = hv.Dataset(
            self.ana.analysis['uclamp_updates'],
            ['time_it', hv.Dimension('tag', values=self.ana.tags), 'iteration', 'uclamp_id', 'task'],
            ['value']
        )
        layout = ds.to(hv.Curve, 'time_it', 'value').overlay('tag').opts(shared_axes=False, title=title)

        layout.opts(
            opts.Curve(height=height, width=width, interpolation='steps-post', framewise=True)
        )
        self.ana.hv_figures[self.ana._title_to_filename(title, '__line')] = layout
        return layout

    # -------- helper functions --------

    def gmean_bars(self, df: pd.DataFrame, x: str = 'stat', y: str = 'value', color: str = 'tag',
                   facet_col: str = 'metric', facet_col_wrap: int = 3, title: str = 'Gmean values',
                   width: Optional[int] = None, height: int = 600, gmean_round: int = 1,
                   include_columns: List[str] = [], table_sort: Optional[List[str]] = None,
                   order_cluster: bool = False, sort_ascending: bool = False, include_total: bool = False,
                   debug: bool = False, percentage: bool = True) -> pd.DataFrame:
        """
        Plot gmean values of some metric with statistical analysis attached.
        It's mainly intended as a way of comparing multiple iterations across workloads and so
        it expects a melt-like (`pd.melt`) dataframe to plot.

        The function heavily relies on multiple assumptions about the underlying dataframe so it might break if those
        are not met.

        .. note:: This function has side-effects. The figure of the resulting plot will be saved to `px_figures`.

        :param df: Dataframe with the data to plot
        :param x: Column denoting the x axis
        :param y: Column denoting the y axis
        :param color: Column denoting how to split the data into bars
        :param facet_col: Column denoting how to split the plot into facets
        :param facet_col_wrap: Number of facet column in one row of the plot
        :param title: Displayed title of the plot
        :param width: Plot width
        :param height: Plot height
        :param gmean_round: Number of decimal places to round the gmeans to
        :param include_columns: Other columns to include in the resulting data table
        :param table_sort: List of columns to sort the data table by
        :param order_cluster: Order the plot by CPU clusters
        :param sort_ascending: Order the plot by ascending values
        :param include_total: Include the 'total' column alongside the clusters
        :param debug: Insert a pdb breakpoint before computing the dataframes
        :param percentage: Include percentage differences and check pvalues

        :return: Dataframe of the ASCII table that will be printed above the resulting plot.

        The resulting dataframe can be included in the summary dict for later use as shown below.
        ```python
        gb5.summary['scores'] = gb5.plot.gmean_bars(
            gb5.results, x='stat', y='value', facet_col='metric', facet_col_wrap=3,
            title='gmean benchmark score', width=1600, height=600
        )
        ```

        """

        shown_clusters = self.ana.CLUSTERS if not include_total else self.ana.CLUSTERS_TOTAL
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

        if percentage and len(self.ana.tags) < 2:
            log.error("Can't compute percentage differences from less than 2 runs")
            percentage = False

        # prepare percentage differences & pvalues
        if percentage:
            # compute percentage differences
            stats_perc = Stats(df, ref_group={'tag': self.ana.tags[0]}, value_col=y,
                               agg_cols=['iteration'], stats={'gmean': sp.stats.gmean}).df
            # re-add stub tag
            stats_perc_vals_temp = stats_perc.query(f"tag == '{self.ana.tags[1]}'")
            stats_perc_vals_temp['tag'] = self.ana.tags[0]
            stats_perc_vals_temp['value'] = 0
            # re-combine a df with percentage differences
            stats_perc_vals = pd.concat([stats_perc_vals_temp, stats_perc])
            stats_perc_vals['order_kernel'] = stats_perc_vals['tag'].map(lambda x: self.ana.tags.index(x))

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
        gmeans['order_kernel'] = gmeans['tag'].map(lambda x: self.ana.tags.index(x))

        if order_cluster:
            gmeans['order_cluster'] = gmeans['cluster'].map(lambda x: shown_clusters.index(x))

        gmeans_mean = gmeans.query("stat == 'gmean'").sort_values(by=sort_list).reset_index(drop=True)

        # prepare the data table
        data_table = gmeans_mean[[
            col for col in gmeans_mean.columns
            if col in (['tag', 'value', 'test_name', 'variable', 'metric', 'chan_name', 'comm'] + include_columns)
        ]]
        if percentage:
            data_table['perc_diff'] = stats_perc_vals['value'].map(lambda x: str(round(x, 2)) + '%')
        data_table['value'] = data_table['value'].apply(lambda x: trim_number(x))
        if table_sort is not None:
            data_table = data_table.sort_values(by=table_sort)
        print_table(data_table)

        # prepare the plot labels
        plot_text = format_percentage(
            gmeans_mean['value'], stats_perc_vals['value'], stats_perc_pvals['value']
        ) if percentage else gmeans_mean['value']

        # plot bars
        fig = px.bar(gmeans_mean, x=x, y=y, color=color, facet_col=facet_col, facet_col_wrap=facet_col_wrap,
                     barmode='group', title=title, width=width, height=height,
                     text=plot_text)
        fig.update_traces(textposition='outside')
        fig.update_yaxes(matches=None)
        if sort_ascending:
            fig.update_xaxes(categoryorder='total ascending')
        self.ana.px_figures[self.ana._title_to_filename(title, "__bar")] = fig
        fig.show(renderer='iframe')

        return data_table

    def lines_px(self, df: pd.DataFrame, x: str = 'iteration', y: str = 'value',
                 color: str = 'tag', facet_col: Optional[str] = None, facet_col_wrap: int = 2,
                 height: int = 600, width: Optional[int] = None, title: Optional[str] = None,
                 scale_y: bool = False, renderer: str = 'iframe'):
        """
        Plot lines of some metric, e.g across iterations.

        .. note:: This function has side-effects. The figure of the resulting plot will be saved to `px_figures`.

        :param df: Dataframe with the data to plot
        :param x: Column denoting the x axis
        :param y: Column denoting the y axis
        :param color: Column denoting how to split the data into lines
        :param facet_col: Column denoting how to split the plot into facets
        :param facet_col_wrap: Number of facet column in one row of the plot
        :param title: Displayed title of the plot
        :param width: Plot width
        :param height: Plot height
        :param scale_y: Whether the y axis should maintain the same scale across facets
        :param renderer: Plotly express renderer to be used
        """
        fig = px.line(df, x=x, y=y, color=color, facet_col=facet_col, facet_col_wrap=facet_col_wrap,
                      height=height, width=width, title=title)
        if not scale_y:
            fig.update_yaxes(matches=None)
        self.ana.px_figures[self.ana._title_to_filename(title, "__line")] = fig
        fig.show(renderer=renderer)


def setup_notebook():
    import plotly.io as pio
    from holoviews import opts
    from bokeh.themes import built_in_themes

    hv.extension('bokeh')
    hv.renderer('bokeh').theme = built_in_themes['dark_minimal']
    hv.renderer('bokeh').webgl = True
    pio.templates.default = "plotly"
    pio.templates.default = "plotly_dark"

    color_cycle = hv.Cycle([
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
    ])

    opts.defaults(
        opts.Curve(tools=['hover'], show_grid=True, color=color_cycle, muted_alpha=0),
        opts.Table(bgcolor='black')
    )


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


def print_table(df):
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False, floatfmt=".3f"))


def trim_wa_path(path):
    return "_".join(path.split("_")[1:-2])

