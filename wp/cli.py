import os
import argparse
import logging as log
import confuse

from lisa.utils import setup_logging

from wp.constants import FULL_METRICS, APP_NAME, DEVICE_COMMANDS
from wp.processor import WorkloadProcessor
from wp.runner import WorkloadRunner
from wp.device import WorkloadDevice


def process(args):
    # Load the config file
    config = confuse.Configuration(APP_NAME, __name__)
    plat_info_path = os.path.expanduser(config['target']['plat_info'].get(str))
    processor = WorkloadProcessor(args.wa_path, init=args.init, plat_info_path=plat_info_path,
                                  no_parser=args.no_parser, validate=not args.skip_validation)

    metrics = args.metrics if args.metrics else FULL_METRICS
    if not args.no_metrics:
        processor.run_metrics(metrics)
    else:
        log.warning('No metrics requested, exiting..')


def run(args):
    runner = WorkloadRunner(args.dir, force=args.force, module=not args.no_module)
    output = runner.run(args.workload, args.tag)

    if args.auto_process and output is not None:
        config = confuse.Configuration(APP_NAME, __name__)
        plat_info_path = os.path.expanduser(config['target']['plat_info'].get(str))
        processor = WorkloadProcessor(output, init=True, plat_info_path=plat_info_path, validate=True)
        processor.run_metrics(FULL_METRICS)


def device(args):
    device = WorkloadDevice()

    for command in args.commands:
        device.dispatch(command)


def main():
    parser = argparse.ArgumentParser(prog='WA Workload Processor')
    subparsers = parser.add_subparsers(required=True, dest='subparser_name', title='subcommands')
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose output, including lisa debug logging")

    parser_process = subparsers.add_parser('process', help='Process a workload run and parse traces')
    parser_process.add_argument('wa_path')
    process_group_init = parser_process.add_argument_group('Trace parsing', 'Options for parsing traces')
    process_group_init_parse = process_group_init.add_mutually_exclusive_group()
    process_group_init_parse.add_argument('-i', '--init', action='store_true', help='Parse traces to init the workload')
    process_group_init_parse.add_argument('--no-parser', action='store_true', help='Do not use trace-parquet on traces')
    process_group_init.add_argument('-s', '--skip-validation', action='store_true',
                                    help='Skip trace validation (only when using trace-parquet)')
    process_group_metric = parser_process.add_mutually_exclusive_group()
    process_group_metric.add_argument('-m', '--metrics', nargs='+', choices=FULL_METRICS,
                                      help='Metrics to process, defaults to all.')
    process_group_metric.add_argument('--no-metrics', action='store_true', help="Do not process metrics")
    parser_process.set_defaults(func=process)

    parser_run = subparsers.add_parser('run', help='Run a workload')
    parser_run.add_argument('workload', help='Workload name or agenda file path')
    group_run = parser_run.add_mutually_exclusive_group()
    group_run.add_argument('tag', nargs='?', help='Tag for the run')
    group_run.add_argument('-d', '--dir', help='Output directory')
    parser_run.add_argument('-n', '--no-module', action='store_true', help="Don't try to load the Lisa kernel module")
    parser_run.add_argument('-f', '--force', action='store_true', help='Overwrite output directory if it exists')
    parser_run.add_argument('-a', '--auto-process', action='store_true', help='Auto process after the run completes')
    parser_run.set_defaults(func=run)

    parser_device = subparsers.add_parser('device', help='Control the device')
    parser_device.add_argument('commands', choices=DEVICE_COMMANDS, nargs='+', help='Device commands to run')
    parser_device.set_defaults(func=device)

    args = parser.parse_args()

    if args.verbose:
        setup_logging(level=log.DEBUG)
    else:
        setup_logging(level=log.INFO)

    args.func(args)
