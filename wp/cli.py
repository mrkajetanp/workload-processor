import os
import argparse
import logging as log
from lisa.utils import setup_logging
from lisa.wa import WAOutput

from wp.helpers import parse_config
from wp.constants import FULL_METRICS
from wp.processor import WorkloadProcessor


def process(args):
    # Load the config file
    config = parse_config()
    plat_info_path = os.path.expanduser(config['target']['plat_info'])
    processor = WorkloadProcessor(WAOutput(args.wa_path), init=args.init, plat_info_path=plat_info_path)

    if not args.no_metrics:
        processor.run_metrics(metrics=args.metrics)
    else:
        log.info('No metrics requested, exiting..')


def run():
    parser = argparse.ArgumentParser(prog='WA Workload Processor')
    subparsers = parser.add_subparsers(required=True, dest='subparser_name', title='subcommands')
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose output, including lisa debug logging")

    parser_process = subparsers.add_parser('process', help='Process workload and parse traces')
    parser_process.add_argument('wa_path')
    parser_process.add_argument('-i', '--init', action='store_true', help='Parse traces to initialise the workload')
    group_process = parser_process.add_mutually_exclusive_group()
    group_process.add_argument('-m', '--metrics', nargs='+', choices=FULL_METRICS,
                               help='Metrics to process, defaults to all.')
    group_process.add_argument('--no-metrics', action='store_true', help="Do not process metrics")
    parser_process.set_defaults(func=process)

    args = parser.parse_args()

    if args.verbose:
        setup_logging(level=log.DEBUG)
    else:
        setup_logging(level=log.INFO)

    args.func(args)
