import os
import pandas as pd
import scipy as sp
import logging as log
import confuse
from tabulate import tabulate
from functools import lru_cache, cached_property

from IPython.display import display
import plotly.express as px

from lisa.wa import WAOutput
from lisa.stats import Stats
from lisa.platforms.platinfo import PlatformInfo
from lisa.utils import LazyMapping

from wp.helpers import wa_output_to_mock_traces
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
    CPUS = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0']
    CLUSTERS = ['little', 'mid', 'big']
    CLUSTERS_TOTAL = ['little', 'mid', 'big', 'total']

    def __init__(self, benchmark_path, benchmark_dirs):
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

    @cached_property
    def plat_info(self):
        self.plat_info = None
        plat_info_path = os.path.expanduser(self.config['target']['plat_info'].get(str))
        if plat_info_path is not None:
            return PlatformInfo.from_yaml_map(plat_info_path)
        return None

    @cached_property
    def traces(self):
        # TODO: fallback version for no-parser
        return LazyMapping({
            trim_wa_path(os.path.basename(wa_output.path)): lru_cache()(
                lambda k: wa_output_to_mock_traces(wa_output, self.plat_info)
            ) for wa_output in self.wa_outputs
        })

    def show(self):
        display(self.results)
        print('benchmark_dirs:', self.benchmark_dirs)
        print('wa_paths:', self.wa_paths)
        print('kernels:', self.kernels)

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
