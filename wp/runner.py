import sys
import os
import subprocess
import logging as log
import confuse

from datetime import date
from ppadb.client import Client as AdbClient

from devlib.exception import TargetStableError
from wp.helpers import load_yaml
from wp.constants import AGENDAS_PATH, SUPPORTED_WORKLOADS, APP_NAME


class WorkloadRunner:
    def __init__(self, output_dir, config=None):
        self.config = confuse.Configuration(APP_NAME, __name__) if config is None else config
        self.output_dir = output_dir
        self.force = self.config['force'].get(False)
        self.adb_client = AdbClient(host=self.config['host']['adb_host'].get(str),
                                    port=int(self.config['host']['adb_port'].get(int)))

        try:
            self.device = self.adb_client.devices()[0]
        except IndexError:
            raise TargetStableError('No target devices found')

        log.debug('Restarting adb as root')
        try:
            print(self.device.root())
        except RuntimeError:
            log.debug('adb already running as root')

        if self.config['no_module'].get(False):
            return

        # check if the lisa module is loaded
        module_present = bool(self.device.shell('lsmod | grep lisa'))
        if module_present:
            log.info('Lisa module found on the target device')
            return

        # insert the lisa module
        target_conf_path = os.path.expanduser(self.config['target']['target_conf'].get(str))
        log.debug(f'Calling lisa-load-kmod with {target_conf_path}')
        log_level = 'debug' if log.getLogger().isEnabledFor(log.DEBUG) else 'info'
        cmd = ['lisa-load-kmod', '--log-level', log_level, '--conf', target_conf_path]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        for c in iter(lambda: process.stdout.read(1), b""):
            sys.stdout.buffer.write(c)

    def run(self, workload, tag):
        if workload.endswith('yaml') or workload.endswith('yml'):
            agenda_path = workload
        elif workload in SUPPORTED_WORKLOADS:
            agenda_path = os.path.join(AGENDAS_PATH, f"agenda_{workload}.yaml")
        else:
            log.error(f"workload or agenda {workload} not supported")
            return None

        agenda_parsed = load_yaml(agenda_path)
        workload_name = agenda_parsed['workloads'][0]['name']
        iterations = agenda_parsed['config']['iterations']
        day_month = date.today().strftime("%d%m")

        if not self.output_dir:
            self.output_dir = os.path.join(os.getcwd(), f"{workload_name}_{tag}_{iterations}_{day_month}")

        cmd = ["wa", "run", str(agenda_path), "-d", self.output_dir]
        if self.force:
            cmd.append("-f")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        for c in iter(lambda: process.stdout.read(1), b""):
            sys.stdout.buffer.write(c)

        return self.output_dir
