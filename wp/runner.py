"""
Workload runner module. Effectively a proxy for `workload-automation` that makes sure the created workloads are
compatible with the rest of workload-processor.

Intended to be accessed through the CLI as follows:
```
usage: workload-processor run [-h] [-d DIR] [-n] [-f] [-a] workload [tag]

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

The module can still normally be used through Python.
"""

import os
import sys
import subprocess
import logging as log
import confuse

from typing import Optional

from pathlib import Path
from datetime import date

from wp.helpers import load_yaml
from wp.constants import AGENDAS_PATH, SUPPORTED_WORKLOADS, APP_NAME
from wp.device import WorkloadDevice


# TODO: use devlib target to make this platform-agnostic
class WorkloadRunner:
    """The runner class that performs setup and dispatches the run to `workload-automation`."""

    def __init__(self, output_dir: Optional[str], config: Optional[confuse.Configuration] = None):
        """
        Initialise the runner

        :param output_dir: Output directory for the run to be stored (defaults to putting the run under CWD)
        :param config: `Confuse` config object
        :raises TargetStableError: When no target devices are found
        """
        self.config = confuse.Configuration(APP_NAME, __name__) if config is None else config
        """`Confuse` configuration handle"""
        self.output_dir = Path(output_dir) if output_dir else None
        """Output directory for the run to be stored"""
        self.force = self.config['force'].get(False)
        """Overwrite the run output directory if it already exists"""
        self.device: WorkloadDevice = WorkloadDevice()
        """`device.WorkloadDevice` handle for device control"""

        if not self.config['no_module'].get(False):
            self.device.load_module()

    def run(self, workload: str, tag: str) -> Path:
        """
        Run workload `workload` with tag `tag`.

        :param workload: Either a name of a predefined workload (consult `wp.constants.SUPPORTED_WORKLOADS`)
        or a path to a `workload-automation` agenda.
        :param tag: Tag for the run, e.g. `baseline`.
        :return: Returns the path to where the run was saved on completion.
        """
        if workload.endswith('yaml') or workload.endswith('yml'):
            agenda_path = workload
        elif workload in SUPPORTED_WORKLOADS:
            agenda_path = Path(AGENDAS_PATH) / f"agenda_{workload}.yaml"
        else:
            log.error(f"workload or agenda {workload} not supported")
            return None

        agenda_parsed = load_yaml(str(agenda_path))
        workload_name = agenda_parsed['workloads'][0]['name']
        iterations = agenda_parsed['config']['iterations']
        day_month = date.today().strftime("%d%m")

        if not self.output_dir:
            self.output_dir = Path(os.getcwd()) / f"{workload_name}_{tag}_{iterations}_{day_month}"

        cmd = ["wa", "run", str(agenda_path), "-d", str(self.output_dir)]
        if self.force:
            cmd.append("-f")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        for c in iter(lambda: process.stdout.read(1), b""):
            sys.stdout.buffer.write(c)

        return self.output_dir
