import os
import argparse
import yaml
import logging as log
from pathlib import Path
from lisa.utils import setup_logging
from lisa.wa import WAOutput

from wp.constants import FULL_METRICS
from wp.processor import WorkloadProcessor


def run():
    parser = argparse.ArgumentParser(prog='WA Workload Processor')
    parser.add_argument('wa_path')
    parser.add_argument('-i', '--init', action='store_true', help='Parse traces to initialise the workload')
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose output, including lisa debug logging")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-m', '--metrics', nargs='+', choices=FULL_METRICS, help='Metrics to process, defaults to all.')
    group.add_argument('--no-metrics', action='store_true', help="Do not process metrics")

    args = parser.parse_args()

    if args.verbose:
        setup_logging(level=log.DEBUG)
    else:
        setup_logging(level=log.INFO)

    # Load the config file
    config_path = Path(__file__).resolve().parent.parent.joinpath('config.yaml')
    with open(config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.CLoader)
    print(config)

    plat_info_path = os.path.expanduser(config['target']['plat_info'])
    processor = WorkloadProcessor(WAOutput(args.wa_path), init=args.init, plat_info_path=plat_info_path)

    if not args.no_metrics:
        processor.run_metrics(metrics=args.metrics)
    else:
        log.info('No metrics requested, exiting..')
