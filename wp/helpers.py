import os
import yaml
import logging as log
import polars as pl

from lisa.trace import Trace, MockTraceParser
from lisa.datautils import series_mean, df_update_duplicates


def load_yaml(path):
    with open(path, "r") as ymlfile:
        return yaml.load(ymlfile, Loader=yaml.CLoader)


def cpu_cluster(cpu):
    return 'little' if cpu < 4 else 'big' if cpu > 5 else 'mid'


def df_add_cluster(df, cpu_col='cpu'):
    return df.with_columns(pl.col(cpu_col).apply(cpu_cluster).alias('cluster'))


def df_sort_by_clusters(df, value_cols):
    df = df.with_columns(pl.col('cluster').map_dict({'little': 0, 'mid': 1, 'big': 2}).alias('order'))
    cols = ['wa_path', 'kernel', 'iteration', 'cluster'] + value_cols
    df = df.sort(['iteration', 'order'])[cols]
    return df


def df_iterations_mean(df, other_cols=None):
    cols = ['wa_path', 'kernel', 'iteration']
    if other_cols is not None:
        cols += other_cols
    return df.groupby(cols, maintain_order=True).agg(
        pl.col('*').apply(lambda x: series_mean(x.to_pandas()))
    )


def df_add_wa_output_tags(df, wa_output):
    log.debug(f'Adding WA output tags to {wa_output.path}')
    try:
        kernel = list(wa_output._jobs.values())[0][0].target_info.kernel_version.release
    except Exception:
        kernel = '<unknown>'
    wa_path = os.path.basename(wa_output.path)
    return df.with_columns(pl.lit(kernel).alias('kernel'), pl.lit(wa_path).alias('wa_path'))


def wa_output_to_mock_traces(wa_output, plat_info=None):
    log.debug('Using trace-parquet + MockTraceParser to create traces')

    job_events_paths = {
        job.iteration: os.path.join(job.basepath, 'trace-events', '*.pq')
        for job in wa_output.jobs
    }

    def list_files(path):
        return [file.strip() for file in os.popen(f"ls {path}").readlines()]

    def pqt_readable(file):
        try:
            pl.read_parquet(file)
            return True
        except Exception as e:
            log.error(f"{e} (reading {file})")
            return False

    def trace_from_pqs(pq_files):
        trace = pqs_to_trace({
            os.path.basename(file).split('.')[0]: pl.read_parquet(file)
            for file in pq_files if pqt_readable(file)
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


def wa_output_to_traces(wa_output, plat_info=None):
    log.debug('Using Lisa default trace parsing to create traces')

    def apply_plat_info(trace):
        if plat_info is not None:
            trace.plat_info = plat_info
            return trace

    return {
        int(label.split('-')[1]): apply_plat_info(trace)
        for label, trace in
        wa_output['trace'].traces.items()
    }


def convert_event_pqt(pqt):
    pqt = pqt.rename({
        'common_cpu': '__cpu', 'common_pid': '__pid',
        'common_comm': '__comm', 'common_ts': 'Time'
    }).with_columns(pl.col('Time') / 1000000000)
    return df_update_duplicates(pqt.to_pandas().set_index('Time')).sort_index()


def pqs_to_trace(pqs):
    def trim_event_name(name):
        return name[6:] if name.startswith('lisa__') else name

    events = {trim_event_name(name): convert_event_pqt(pq) for (name, pq) in pqs.items()}
    return Trace(parser=MockTraceParser(events))


def flatten(t):
    return [item for sublist in t for item in sublist]


def trim_task_comm(task):
    return task[1:-1].split(':')[1]


def try_get_task_ids(trace, task):
    try:
        return trace.get_task_ids(task)
    except KeyError:
        log.warning(f'Task "{task}" not found in {trace.trace_path}')
        return None
