import sys
import os
import subprocess
import logging as log

from datetime import date
from ppadb.client import Client as AdbClient

from wp.helpers import load_yaml
from wp.constants import CONFIG_PATH, AGENDAS_PATH, SUPPORTED_WORKLOADS


class WorkloadRunner:
    def __init__(self, output_dir, force=False):
        self.config = load_yaml(CONFIG_PATH)
        self.output_dir = output_dir
        self.force = force
        self.adb_client = AdbClient(host=self.config['target']['adb_host'], port=int(self.config['target']['adb_port']))
        self.device = self.adb_client.devices()[0]

        log.debug('Restarting adb as root')
        try:
            print(self.device.root())
        except RuntimeError:
            log.debug('adb already running as root')

        # insert the lisa module
        module_path = self.config['device']['lisa_module_path']
        log.debug('Inserting the lisa module')
        print(self.device.shell(f"insmod {module_path}"))

    def run(self, workload, tag):
        if workload.endswith('yaml') or workload.endswith('yml'):
            agenda_path = workload
        elif workload in SUPPORTED_WORKLOADS:
            agenda_path = os.path.join(AGENDAS_PATH, f"agenda_{workload}.yaml")
        else:
            log.error(f"workload or agenda {workload} not supported")
            return

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
