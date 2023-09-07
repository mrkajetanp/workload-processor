"""
Device controller module. Primarily intended to be accessed through the CLI as follows:
```
usage: workload-processor device [-h]
                             {status,disable-cpusets,disable-cpushares,menu,teo,latency-sensitive,powersave,performance,schedutil,sugov-rate-limit,load-module}
                             [{status,disable-cpusets,disable-cpushares,menu,teo,latency-sensitive,powersave,performance,schedutil,sugov-rate-limit,load-module} ...]

positional arguments:
  {status,disable-cpusets,disable-cpushares,menu,teo,latency-sensitive,powersave,performance,schedutil,sugov-rate-limit,load-module}
                        Device commands to run

optional arguments:
  -h, --help            show this help message and exit
```
Providers helpers for controlling the target device over ADB.
"""

import os
import sys
import logging as log
import confuse
import subprocess

from typing import List

from lisa.target import Target, TargetConf
from pathlib import Path
from devlib.exception import TargetStableCalledProcessError

from wp.constants import APP_NAME
from wp.exception import WPConfigError


class WorkloadDevice:
    def __init__(self):
        self.config = confuse.Configuration(APP_NAME, __name__)
        """Handle for the `Confuse` configuration object."""
        self.device = None
        """Handle for the Target device"""
        try:
            self.device = Target.from_conf(
                TargetConf.from_yaml_map(
                    Path(self.config['target']['target_conf'].get()).expanduser()
                )
            )
        except FileNotFoundError:
            raise WPConfigError('target_conf was not properly set in the config')

        self.device.execute("setenforce 0", as_root=True)

    def dispatch(self, command: str):
        """
        Call the device controller function corresponding to the command string.

        :param command: Command string
        :raises RuntimeError: In case of ADB errors
        """
        def command_to_function(cmd):
            return getattr(self, cmd.replace('-', '_'))

        try:
            command_to_function(command)()
        except RuntimeError as e:
            if 'closed' in str(e):
                log.error('ADB Connection closed')
            else:
                raise e

    def status(self):
        log.info('Showing current device status')
        kernel_version = self.device.execute("uname -a").strip().split()[2]
        log.info(f'Kernel version {kernel_version}')

        selinux = self.device.execute("getenforce").strip()
        log.info(f'SELinux status: {selinux}')

        module_present = bool(self.device.execute('lsmod | grep lisa'))
        log.info(f"Lisa module {'' if module_present else 'not '}loaded")

        tz_cmd = "cat /sys/class/thermal/thermal_zone{}/temp"
        big_temp = self.device.execute(tz_cmd.format(0)).strip()
        mid_temp = self.device.execute(tz_cmd.format(1)).strip()
        ltl_temp = self.device.execute(tz_cmd.format(2)).strip()
        log.info(f"Temperature: BIG {big_temp} MID {mid_temp} LTL {ltl_temp}")

        cpufreq_cmd = "cat /sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor"
        ltl_cpufreq = self.device.execute(cpufreq_cmd.format(0)).strip()
        mid_cpufreq = self.device.execute(cpufreq_cmd.format(4)).strip()
        big_cpufreq = self.device.execute(cpufreq_cmd.format(6)).strip()
        log.info(f"BIG {big_cpufreq} MID {mid_cpufreq} LTL {ltl_cpufreq}")

        ls_system = self.device.execute("cat /dev/cpuctl/system/cpu.uclamp.latency_sensitive").strip()
        ls_fg = self.device.execute("cat /dev/cpuctl/foreground/cpu.uclamp.latency_sensitive").strip()
        ls_bg = self.device.execute("cat /dev/cpuctl/background/cpu.uclamp.latency_sensitive").strip()
        ls_sysbg = self.device.execute("cat /dev/cpuctl/system-background/cpu.uclamp.latency_sensitive").strip()
        log.info(f"latency_sensitive system: {ls_system} fg: {ls_fg} bg: {ls_bg} sys-bg: {ls_sysbg}")

        idle_governor = self.device.execute("cat /sys/devices/system/cpu/cpuidle/current_governor_ro").strip()
        log.info(f"cpuidle governor: {idle_governor}")

        # TODO: use cgroups in config
        cpuset_bg = self.device.execute("cat /dev/cpuset/background/cpus").strip()
        cpuset_fg = self.device.execute("cat /dev/cpuset/foreground/cpus").strip()
        cpuset_sbg = self.device.execute("cat /dev/cpuset/system-background/cpus").strip()
        log.info(f"cpusets: background: {cpuset_bg}, foreground: {cpuset_fg}, system-bg: {cpuset_sbg}")

        cpushares_bg = self.device.execute("cat /dev/cpuctl/background/cpu.shares").strip()
        cpushares_fg = self.device.execute("cat /dev/cpuctl/foreground/cpu.shares").strip()
        cpushares_sbg = self.device.execute("cat /dev/cpuctl/system-background/cpu.shares").strip()
        cpushares_sys = self.device.execute("cat /dev/cpuctl/system/cpu.shares").strip()
        log.info(f"cpushares: bg: {cpushares_bg} fg: {cpushares_fg} sys: {cpushares_sys} sys-bg: {cpushares_sbg}")

        if 'schedutil' in [ltl_cpufreq, mid_cpufreq, big_cpufreq]:
            su_rate_cmd = "cat /sys/devices/system/cpu/cpufreq/policy{}/schedutil/rate_limit_us"
            pol_0_rl = self.device.execute(su_rate_cmd.format(0)).strip()
            pol_4_rl = self.device.execute(su_rate_cmd.format(4)).strip()
            pol_6_rl = self.device.execute(su_rate_cmd.format(6)).strip()
            log.info(f"policy rate limits: 0: {pol_0_rl}, 4: {pol_4_rl}, 6: {pol_6_rl}")

    def load_module(self):
        # Try to load the module using modprobe (for in-tree builds
        log.debug('Loading the Lisa module')
        try:
            self.device.execute("rmmod lisa", as_root=True)
        except TargetStableCalledProcessError:
            pass

        modules_base_path = self.config['target']['modules_path'].get()
        modules_dir_count = int(self.device.execute(f"ls {modules_base_path} | wc -l").strip())
        modules_dir_version_cmd = f"ls {modules_base_path} | head -1" if modules_dir_count == 1 else 'uname -r'
        modules_dir_version = self.device.execute(modules_dir_version_cmd).strip()
        modules_path = os.path.join(modules_base_path, modules_dir_version)
        self.device.execute(f"modprobe -d {modules_path} lisa", as_root=True)

        # check if the lisa module is loaded
        module_present = bool(self.device.execute('lsmod | grep lisa'))
        if module_present:
            log.info('Lisa module loaded successfully from the target device')
            return

        # Try to load the module using lisa-load-kmod
        target_conf_path = Path(self.config['target']['target_conf'].get(str)).expanduser()
        log.debug(f'Calling lisa-load-kmod with {target_conf_path}')
        log_level = 'debug' if log.getLogger().isEnabledFor(log.DEBUG) else 'info'
        cmd = ['lisa-load-kmod', '--log-level', log_level, '--conf', str(target_conf_path)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        for c in iter(lambda: process.stdout.read(1), b""):
            sys.stdout.buffer.write(c)

    def disable_cpusets(self):
        groups: List[str] = self.config['target']['cgroups'].get()
        log.info(f'Disabling cpusets for groups {", ".join(groups)}')
        for group in groups:
            self.device.execute(f"echo '0-7' > /dev/cpuset/{group}/cpus")

    def disable_cpushares(self):
        groups: List[str] = self.config['target']['cgroups'].get()
        log.info(f'Setting cpushares to 20480 for groups {", ".join(groups)}')
        for group in groups:
            self.device.execute(f"echo 20480 > /dev/cpuctl/{group}/cpu.shares")

    def menu(self):
        log.info('Setting the current cpuidle governor to menu')
        self.device.execute("echo 'menu' > /sys/devices/system/cpu/cpuidle/current_governor")

    def teo(self):
        log.info('Setting the current cpuidle governor to teo')
        self.device.execute("echo 'teo' > /sys/devices/system/cpu/cpuidle/current_governor")

    def latency_sensitive(self):
        log.info('Setting the system cgroup to latency sensitive')
        self.device.execute("echo 1 > /dev/cpuctl/system/cpu.uclamp.latency_sensitive")

    def powersave(self):
        log.info('Setting the cpufreq governor to powersave')
        for cpu in range(8):
            self.device.execute(f"echo 'powersave' > /sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")

    def performance(self):
        log.info('Setting the cpufreq governor to performance')
        for cpu in range(8):
            self.device.execute(f"echo 'performance' > /sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")

    def schedutil(self):
        log.info('Setting the cpufreq governor to schedutil')
        for cpu in range(8):
            self.device.execute(f"echo 'schedutil' > /sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")

    def sugov_rate_limit(self):
        log.info('Setting the sugov rate limit to 500')
        for policy in [0, 4, 6]:
            self.device.execute(f"echo '500' > /sys/devices/system/cpu/cpufreq/policy{policy}/schedutil/rate_limit_us")
