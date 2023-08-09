# workload-processor

Automated workload processor designed to be a simplified way of applying analysis provided by [Lisa](https://github.com/ARM-software/lisa) to workloads created by [workload-automation](https://github.com/ARM-software/workload-automation)
It came around as a way of automating my own workflows and so to a large extent it comes as-is and it very likely contains many assumptions about how certain things are done that might not be the case for others.

## Installing

workload-processor is built on top of [Lisa](https://github.com/ARM-software/lisa) and Lisa needs to be installed for it to work.

1. Clone and install Lisa (follow instructions from the project)
2. Clone this project (`git clone https://github.com/mrkajetanp/workload-processor`)
3. Source the Lisa environment (`cd ~/lisa && source init_env`)
4. Install this project into the Lisa environment (`python -m pip install -e ~/workload-processor`)

## Usage

workload-processor is split into 4 parts - the runner, the processor, the device controller and the notebook analysis.
All of the parts can easily function separately but they are desgined to make using them all together as easy as possible and to some extent are interdependent.

## Configuration

Different parts of the tool use the configuration options below.
By default, the values in wp/config_default.yaml will be used. They can be overridden as-needed in ~/.config/workload-processor/config.yaml.

* `plat_info` - Platform info used for some analysis metrics. Either use the one provided in `assets/` for Pixel 6 or generate one by modifying the provided script.
* `target_conf` - Path to the device target config. Used to build the Lisa module with lisa-load-kmod.
* `adb_host` & `adb_port` - Defaults should work, ADB access settings used by the `run` and `device` subcommands

```
target:
  plat_info: ~/power/pixel6/workload-processor/assets/p6-platform-info.yml
  target_conf: ~/tools/lisa/target_conf_p6.yml
host:
  adb_host: 127.0.0.1
  adb_port: 5037
```

Additionally, the following things can be configured in the same way:
* important tasks for each workload to be selected in analysis
* perf counter ids to be selected and renamed
* cgroups to be considered in cgroup-related functiosn
* clusters denoting names and types of cpus
* thermal zones present on the target device

Consult wp/config_default.yaml for the complete set of overrideable options.

## Entry points

The main entry point to the project is through the command line (installed as `workload-processor` into the Lisa PATH using the instructions above).
Alternatively, all of the internals are accessible through the `wp` python package inside their respective modules.
The main module intended to be accessed by end users is `wp.notebook` for the notebook analysis.

## The runner

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

### Relevant help section

```
usage: WA Workload Processor run [-h] [-d DIR] [-n] [-f] [-a] workload [tag]

positional arguments:
  workload            Workload name or agenda file path
  tag                 Tag for the run

optional arguments:
  -h, --help          show this help message and exit
  -d DIR, --dir DIR   Output directory
  -n, --no-module     Don't try to load the Lisa kernel module
  -f, --force         Overwrite output directory if it exists
  -a, --auto-process  Auto process after the run completes
```

### workload-automation plugins

Some useful non-upstreamed workload-automation plugins can be found under the plugins/ directory.
In order to make them available to WA they just need to be put under `~/.workload_automation/plugins/`.

## The processor

The processor is the main part of this project. It can be accessed using `workload-processor process`.
It functions by applying some sort of analysis (metrics found in `wp/processor.py`) on top of each trace in the run, then aggregating them into one tagged dataframe and saving it as pqt to `analysis/` inside the run directory.
These generated pqts can then be either read manually, by the provided notebooks or by custom-made notebooks.
If no metrics are provided, the default is to apply all of them in turn which might take multiple hours.

Example of extracting power usage (requires the `pixel6_emeter` trace event to be present).
```
workload-processor process speedometer_baseline_10_0812 -m power
```

This will result in `pixel6_emeter.pqt` & `pixel6_emeter_mean.pqt` being created in `speedometer_baseline_10_0812/analysis`.
Multiple space-separated metrics can be provided to the `-m` argument, they will be processed in order.

### Trace parsing

By default, the tool is designed to use the experimental Rust trace parser `trace-parquet` as long as it can be found in the PATH.
Before processing the workload, if any of the iterations do not contain a `trace-events` directory one will be created and `trace-parquet` will be called on its trace to generate `.pq` files for each event.
This pre-parsing behaviour can be forced with `-i/--init`. Using the parser results in considerably faster workload analysis times for most of the metrics.

If `trace-parquet` is not found or `--no-parser` was passed the tool will default to the normal Lisa way of creating traces.
While much slower it might be useful for some cases where `trace-parquet` might not work.

### Relevant help section

```
usage: WA Workload Processor process [-h] [-i | --no-parser] [-s]
                                     [-m {...} [{...} ...]
                                     | --no-metrics]
                                     wa_path

positional arguments:
  wa_path

optional arguments:
  -h, --help            show this help message and exit
  -m {...} [{...} ...], --metrics {...} [{...} ...] Metrics to process, defaults to all.
  --no-metrics          Do not process metrics

Trace parsing:
  Options for parsing traces

  -i, --init            Parse traces to init the workload
  --no-parser           Do not use trace-parquet on traces
  -s, --skip-validation
                        Skip trace validation (only when using trace-parquet)
```

## Device controller

The device controller can be accessed through `workload-processor device`.
It's nothing more than a convenience tool for running `adb` commands to get information or change relevant kernel settings in sysfs.
The main command is `status` which will just print available information about the status of the device.
The commands will be run in the provided order and so can be chained (e.g. `workload-processor device sugov-rate-limit status`).
To check which adb commands will be run just consult `wp.device`.

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

## Notebook analysis

The notebook analysis part is made up of a python module with extracted common helper code (`wp.notebook`) along with the notebooks provided under `ipynb/` which make use of it.
Usage examples can be found by simply looking at the provided notebooks.
The main idea is to contain analysis tied to different runs of a specific workload, e.g. Geekbench, into one python object of `wp.notebook.WorkloadNotebookAnalysis`.

### Creating the analysis object

`wp.notebook.WorkloadNotebookAnalysis` takes a directory with benchmark runs and a list of the run directories inside it as arguments.
The notebooks should be able to automatically adjust to changing the number of runs.

```python
gb5 = WorkloadNotebookAnalysis('/home/user/tmp/geekbench/', [
    'geekbench_baseline_3_3101',
    'geekbench_ufc_feec_all_cpus_3_3001',
], label='Geekbench 5')
```

Various information related to the workload can then be accessed through said object. Consult the class documentation section for details.

### Plotting

Every `wp.notebook.WorkloadNotebookAnalysis` object will automatically be created with an associated object of `wp.notebook.WorkloadNotebookPlotter` accessible through its `plot` property.
The `plot` proxy can be used to accessed all the pre-defined plotting methods, for the complete list of available plots consult `wp.notebook.WorkloadNotebookPlotter`.

#### Manual plotting

The `wp.notebook.WorkloadNotebookPlotter.gmean_bars` helper method can be used to plot a given dataframe as bars and automatically attach statistical analysis to it.
There is a corresponding helper method for line plots - `wp.notebook.WorkloadNotebookPlotter.lines_px`.

Otherwise, as long as the data is loaded into the analysis it can be plotted using any other library that supports Pandas dataframes.

### Loading metrics generated by the processor

When using the pre-defined plotting functions the relevant metrics will automatically be loaded the first time the plot is generated and the re-used. No further steps should be necessary.
The metrics are loaded using the `wp.notebook.WorkloadNotebookPlotter.requires_analysis` decorator.

To find out which metrics correspond to which private loader functions consult `wp.notebook.WorkloadNotebookPlotter.analysis_to_loader`. The loader functions can be called manually if needed but it should not be necessary.

#### Manually loading the metrics

The analysis pqts generated by `workload-processor process` in `analysis/` can be manually loaded using `wp.notebook.WorkloadNotebookAnalysis.load_combined_analysis` as shown below.
The function will take a filename, then go across every directory in `gb5.benchmark_dirs`, collect the file from its `analysis/` directory and concat them into one.
In the below example the resulting dataframe can be found in `gb5.analysis['overutilized']`.


```python
def postprocess_overutil(df):
    df['time'] = round(df['time'], 2)
    df['total_time'] = round(df['total_time'], 2)
    return df

gb5.load_combined_analysis('overutilized.pqt', postprocess=postprocess_overutil)
```
