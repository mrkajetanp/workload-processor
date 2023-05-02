# workload-processor

This project is an automated workload processor designed to be a simplified way of applying analysis provided by [Lisa](https://github.com/ARM-software/lisa) to workloads created by [workload-automation](https://github.com/ARM-software/workload-automation)
It came around as a way of automating my own workflows and so to a large extent it comes as-is and it very likely contains many assumptions about how certain things are done that might not be the case for others.

## Installing

This project is built on top of [Lisa](https://github.com/ARM-software/lisa) and Lisa needs to be installed for it to work.

1. Clone and install Lisa (follow instructions from the project)
2. Clone this project (`git clone https://github.com/mrkajetanp/workload-processor`)
3. Source the Lisa environment (`cd ~/lisa && source init_env`)
4. Install this project into the Lisa environment (`python -m pip install -e ~/workload-processor`)

## Usage

workload-processor is split into 4 parts - the runner, the processor, the device controller and the notebook analysis.
All of the parts can easily function separately but they are desgined to make using them all together as easy as possible and to some extent are interdependent.

### Configuration

Different parts of the tool use the below configuration options specified in config.yaml in the project root directory.

* `plat_info` - Platform info used for some analysis metrics. Either use the one provided in `assets/` for Pixel 6 or generate one by modifying the provided script.
* `adb_host` & `adb_port` - Defaults should work, ADB access settings used by the `run` and `device` subcommands
* `lisa_module_path` - Path to the Lisa module file on the target device, used by `run` before starting a workload

```
target:
  plat_info: ~/power/pixel6/workload-processor/assets/p6-platform-info.yml
  adb_host: 127.0.0.1
  adb_port: 5037
device:
  lisa_module_path: /data/local/sched_tp.ko
```

### Entry points

The main entry point to the project is through the command line (installed as `workload-processor` into the Lisa PATH using the instructions above).
Alternatively, all of the internals are accessible through the `wp` python package inside their respective modules.
The main module intended to be accessed by end users is `wp.notebook` for the notebook analysis. More on that later.

### The runner

The runner (accessible through `workload-processor run`) is simply a wrapper around the [workload-automation](https://github.com/ARM-software/workload-automation) project bundled with Lisa.
Using it is completely optional, invoking WA directly will work just as well apart from requiring some extra steps.
The runner is simply intended to make sure adb is running as root & the Lisa module is inserted prior to starting WA.
Without specifying an output directory explicitly (`-d/--dir`) the runner can work by just specifying a tag for the workload.
It will then fill out the workload name, number of iterations and date automatically according to the format mentioned below.
In order for the notebook analysis to work as intended the benchmark names need to be in the `<workload_name>_<tag>_<iterations>_<day><month>` format, e.g. `speedometer_baseline_10_0812`.

```
workload-processor run geekbench baseline
workload-processor run geekbench -d geekbench_baseline_3_0205
```

The first argument taken by the runner - `workload` - can either be one of the supported workload names or a WA agenda file directly.
If a workload name is passed, one of the provided agendas in `agendas/` will be used.
They can either be modified directly or copied and then given to the runner by providing a path to the file instead.

```
workload-processor run agenda.yaml baseline
```

#### Relevant help section

```
usage: WA Workload Processor run [-h] [-d DIR] [-f] workload [tag]

positional arguments:
  workload           Workload name or agenda file path
  tag                Tag for the run

optional arguments:
  -h, --help         show this help message and exit
  -d DIR, --dir DIR  Output directory
  -f, --force        Overwrite output directory if it exists.
```

### The processor

The processor is the main part of this project. It can be accessed using `workload-processor process`.
It functions by applying some sort of analysis (metrics found in wp/processor.py) on top of each trace in the run, then aggregating them into one tagged dataframe and saving it as pqt to `analysis/` inside the run directory.
These generated pqts can then be either read manually, by the provided notebooks or by custom-made notebooks.
If no metrics are provided, the default is to apply all of them in turn which might take multiple hours.

Example of extracting power usage (requires the `pixel6_emeter` trace event to be present).
```
workload-processor process speedometer_baseline_10_0812 -m power
```

This will result in `pixel6_emeter.pqt` & `pixel6_emeter_mean.pqt` being created in `speedometer_baseline_10_0812/analysis`.
Multiple space-separated metrics can be provided to the `-m` argument, they will be processed in order.

#### Trace parsing

By default, the tool is designed to use the experimental Rust trace parser `trace-parquet` as long as it can be found in the PATH.
Before processing the workload, if any of the iterations do not contain a `trace-events` directory one will be created and `trace-parquet` will be called on its trace to generate `.pq` files for each event.
This pre-parsing behaviour can be forced with `-i/--init`. Using the parser results in considerably faster workload analysis times for most of the metrics.

If `trace-parquet` is not found or `--no-parser` was passed the tool will default to the normal Lisa way of creating traces.
While much slower it might be useful for some cases where `trace-parquet` might not work.

#### Relevant help section

```
usage: WA Workload Processor process [-h] [-i | --no-parser]
                                     [-m {power,idle,idle_miss,freq,overutil,pelt,uclamp,adpf,thermal,wakeup-latency,tasks-residency,tasks-activations,cgroup-attach,wakeup-latency-cgroup,tasks-residency-cgroup,energy-estimate} [{power,idle,idle_miss,freq,overutil,pelt,uclamp,adpf,thermal,wakeup-latency,tasks-residency,tasks-activations,cgroup-attach,wakeup-latency-cgroup,tasks-residency-cgroup,energy-estimate} ...]
                                     | --no-metrics]
                                     wa_path

positional arguments:
  wa_path

optional arguments:
  -h, --help            show this help message and exit
  -i, --init            Parse traces to init the workload
  --no-parser           Do not use trace-parquet on traces
  -m {power,idle,idle_miss,freq,overutil,pelt,uclamp,adpf,thermal,wakeup-latency,tasks-residency,tasks-activations,cgroup-attach,wakeup-latency-cgroup,tasks-residency-cgroup,energy-estimate} [{power,idle,idle_miss,freq,overutil,pelt,uclamp,adpf,thermal,wakeup-latency,tasks-residency,tasks-activations,cgroup-attach,wakeup-latency-cgroup,tasks-residency-cgroup,energy-estimate} ...], --metrics {power,idle,idle_miss,freq,overutil,pelt,uclamp,adpf,thermal,wakeup-latency,tasks-residency,tasks-activations,cgroup-attach,wakeup-latency-cgroup,tasks-residency-cgroup,energy-estimate} [{power,idle,idle_miss,freq,overutil,pelt,uclamp,adpf,thermal,wakeup-latency,tasks-residency,tasks-activations,cgroup-attach,wakeup-latency-cgroup,tasks-residency-cgroup,energy-estimate} ...]
                        Metrics to process, defaults to all.
  --no-metrics          Do not process metrics
```

### Device controller

The device controller can be accessed through `workload-processor device`.
It's nothing more than a convenience tool for running `adb` commands to get information or change relevant kernel settings in sysfs.
The main command is `status` which will just print available information about the status of the device.
The commands will be run in the provided order and so can be chained (e.g. `workload-processor device sugov-rate-limit status`).
To check and modify which adb commands will be run just edit `wp/device.py`.

#### Relevant help section

```

usage: WA Workload Processor device [-h]
                                    {status,disable-cpusets,disable-cpushares,menu,teo,latency-sensitive,powersave,performance,schedutil,sugov-rate-limit}
                                    [{status,disable-cpusets,disable-cpushares,menu,teo,latency-sensitive,powersave,performance,schedutil,sugov-rate-limit} ...]

positional arguments:
  {status,disable-cpusets,disable-cpushares,menu,teo,latency-sensitive,powersave,performance,schedutil,sugov-rate-limit}
                        Device commands to run

optional arguments:
  -h, --help            show this help message and exit
```

### Notebook analysis

The notebook analysis part is made up of a python module with extracted common helper code (`wp/notebook.py`) along with the notebooks provided under `ipynb/` which make use of it.
Usage examples can be found by simply looking at the provided notebooks.
The main idea is to contain analysis tied to different runs of a specific workload, e.g. Geekbench, into one python object of WorkloadNotebookAnalysis.

#### Creating the analysis object

WorkloadNotebookAnalysis takes a directory with benchmark runs and a list of the run directories inside it as arguments.
The notebooks should be able to automatically adjust to changing the number of runs as long as the number is larger than 1. Providing only 1 might break some statistical analysis code.

```
gb5 = WorkloadNotebookAnalysis('/home/kajpuc01/power/pixel6/geekbench/', [
    'geekbench_baseline_3_3101',
    'geekbench_ufc_feec_all_cpus_3_3001',
])
```

Various metrics related to the workload can then be accessed through said object.

```
gb5.results # the result dataframe from WA
gb5.results_perf # the resulting perf data if perf was enabled for the run
gb5.analysis # a dict for holding various analysis metrics, more on that below
gb5.summary # a dict for holding summary data used by the TLDR/summary cells in the notebooks
gb5.traces # a dict of <workload_tag>:[traces of iterations], generated with gb5.load_traces()
```

#### Plotting statistical comparison bar plots

The `plot_gmean_bars` helper method can be used to plot a given dataframe as bars and automatically attach statistical analysis to it.
It's mainly intended as a way of comparing gmean values of multiple iterations across workloads and so it expects a melt-like (`pd.melt`) dataframe to plot.
Its signature can be found in `wp/notebook.py` and the function heavily relies on multiple assumptions about the underlying dataframe so it might break.
It returns a dataframe of the ASCII table that will be printed above the resulting plot. That dataframe can be included in the summary dict for later use as shown.

```
gb5.summary['scores'] = gb5.plot_gmean_bars(gb5.results, x='stat', y='value', facet_col='metric', facet_col_wrap=3, title='gmean benchmark score', width=1600, height=600)
```

#### Loading metrics generated by the processor

The analysis pqts generated by `workload-processor process` in `analysis/` can be loaded using `load_combined_analysis` as shown below.
The function will take a filename, then go across every directory in `gb5.benchmark_dirs`, collect the file from its `analysis/` directory and concat them into one.
Unless `trim_path=False` is passed it will also automatically trim the `wa_path` column to only contain the tag instead of the full directory name.
Optionally the function also takes `preprocess` and `postprocess` function arguments.
The former will be applied onto each workload analysis dataframe before they're all concatenated into one.
The latter will be applied onto the resulting concatenated dataframe.
The function will automatically add the final dataframe to `gb5.analysis` using the part before `.` in `name` as the key.
E.g. in the below example the resulting dataframe can be found in `gb5.analysis['overutilized']`.


```
def postprocess_overutil(df):
    df['time'] = round(df['time'], 2)
    df['total_time'] = round(df['total_time'], 2)
    return df

gb5.load_combined_analysis('overutilized.pqt', postprocess=postprocess_overutil)
```
