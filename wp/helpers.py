import os
import yaml
import logging as log
import pandas as pd
from lisa.trace import Trace, MockTraceParser
from lisa.datautils import series_mean


def load_yaml(path):
    with open(path, "r") as ymlfile:
        return yaml.load(ymlfile, Loader=yaml.CLoader)


def df_add_cluster(df, cpu_col='cpu'):
    df['cluster'] = df[cpu_col].copy().apply(lambda c: 'little' if c < 4 else 'big' if c > 5 else 'mid')
    return df


def df_sort_by_clusters(df, value_cols):
    df['order'] = df['cluster'].replace('little', 0).replace('mid', 1).replace('big', 2)
    cols = ['wa_path', 'kernel', 'iteration', 'cluster'] + value_cols
    df = df.sort_values(by=['iteration', 'order'])[cols]
    return df


def df_iterations_mean(df, other_cols=None):
    cols = ['wa_path', 'kernel', 'iteration']
    if other_cols is not None:
        cols += other_cols
    return df.groupby(cols).agg(lambda x: series_mean(x)).reset_index()


def traces_analysis(traces, trace_to_df):
    def analyse_iteration(trace, iteration):
        log.debug(f'Processing iteration {iteration} from {trace.trace_path} with {trace_to_df.__name__}')
        df = trace_to_df(trace)
        df['iteration'] = iteration
        return df
    return pd.concat([analyse_iteration(trace, iteration) for iteration, trace in traces.items()])


def df_add_wa_output_tags(df, out):
    kernel = out._jobs[os.path.basename(out.path)][0].target_info.kernel_version.release
    wa_path = os.path.basename(out.path)
    df['kernel'] = kernel
    df['wa_path'] = wa_path
    return df


def wa_output_to_mock_traces(wa_output, plat_info=None):
    job_events_paths = {
        job.iteration: os.path.join(job.basepath, 'trace-events', '*.pq')
        for job in wa_output.jobs
    }

    def list_files(path):
        return [file.strip() for file in os.popen(f"ls {path}").readlines()]

    def trace_from_pqs(pq_files):
        trace = pqs_to_trace({
            os.path.basename(file).split('.')[0]: pd.read_parquet(file)
            for file in pq_files
        })
        if plat_info is not None:
            trace.plat_info = plat_info
        trace.trace_path = os.path.commonpath(pq_files)
        return trace

    traces = {
        iteration: trace_from_pqs(list_files(job_events))
        for (iteration, job_events) in job_events_paths.items()
    }

    return traces


def convert_event_pqt(pqt):
    pqt = pqt.rename(columns={
        'common_cpu': '__cpu', 'common_pid': '__pid',
        'common_comm': '__comm', 'common_ts': 'Time'
    })
    pqt['Time'] = pqt['Time'] / 1000000000
    return pqt.set_index('Time')


def pqs_to_trace(pqs):
    events = {name: convert_event_pqt(pq) for (name, pq) in pqs.items()}
    return Trace(parser=MockTraceParser(events))


def flatten(t):
    return [item for sublist in t for item in sublist]


def trim_task_comm(task):
    return task[1:-1].split(':')[1]
