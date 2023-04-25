import os
import argparse
import logging as log

from lisa.utils import setup_logging
from lisa.wa import WAOutput

from wp.helpers import load_yaml
from wp.constants import FULL_METRICS, CONFIG_PATH, DEVICE_COMMANDS
from wp.processor import WorkloadProcessor
from wp.runner import WorkloadRunner
from wp.device import WorkloadDevice


def process(args):
    # Load the config file
    config = load_yaml(CONFIG_PATH)
    plat_info_path = os.path.expanduser(config['target']['plat_info'])
    processor = WorkloadProcessor(WAOutput(args.wa_path), init=args.init, plat_info_path=plat_info_path)

    if not args.no_metrics:
        processor.run_metrics(metrics=args.metrics)
    else:
        log.info('No metrics requested, exiting..')


def run(args):
    runner = WorkloadRunner(args.dir, force=args.force)
    runner.run(args.workload, args.tag)


def device(args):
    device = WorkloadDevice()

    cmd_to_device_function = {
        'status': device.status,
        'disable-cpusets': device.disable_cpusets,
        'disable-cpushares': device.disable_cpushares,
        'menu': device.menu,
        'teo': device.teo,
        'latency-sensitive': device.latency_sensitive,
        'powersave': device.powersave,
        'performance': device.performance,
        'schedutil': device.schedutil,
        'sugov-rate-limit': device.sugov_rate_limit,
    }

    for command in args.commands:
        cmd_to_device_function[command]()


def main():
    parser = argparse.ArgumentParser(prog='WA Workload Processor')
    subparsers = parser.add_subparsers(required=True, dest='subparser_name', title='subcommands')
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose output, including lisa debug logging")

    parser_process = subparsers.add_parser('process', help='Process a workload run and parse traces')
    parser_process.add_argument('wa_path')
    parser_process.add_argument('-i', '--init', action='store_true', help='Parse traces to initialise the workload')
    group_process = parser_process.add_mutually_exclusive_group()
    group_process.add_argument('-m', '--metrics', nargs='+', choices=FULL_METRICS,
                               help='Metrics to process, defaults to all.')
    group_process.add_argument('--no-metrics', action='store_true', help="Do not process metrics")
    parser_process.set_defaults(func=process)

    parser_run = subparsers.add_parser('run', help='Run a workload')
    parser_run.add_argument('workload', help='Workload name or agenda file path')
    parser_run.add_argument('tag', help='Tag for the run')
    parser_run.add_argument('-d', '--dir', help='Output directory')
    parser_run.add_argument('-f', '--force', action='store_true', help='Overwrite output directory if it exists.')
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
