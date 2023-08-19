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
import time
import logging as log
import confuse
import subprocess

from ppadb.client import Client as AdbClient
from devlib.exception import TargetStableError
from pathlib import Path

from wp.constants import APP_NAME


class WorkloadDevice:
    def __init__(self):
        self.config = confuse.Configuration(APP_NAME, __name__)
        """Handle for the `Confuse` configuration object."""
        self.adb_client = AdbClient(host=self.config['host']['adb_host'].get(str),
                                    port=int(self.config['host']['adb_port'].get(int)))
        """Handle for the ADB client"""
        self.device = None
        """Handle for the ADB device"""
        try:
            self.device = self.adb_client.devices()[0]
        except IndexError:
            raise TargetStableError('No target devices found')

        log.debug('Restarting adb as root')
        try:
            self.device.root()
            log.info('ADB restarted as root')
        except RuntimeError as e:
            log.error(e)

        # Give ADB on device a moment to initialise
        time.sleep(3)

        self.device.shell("setenforce 0")

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
        kernel_version = self.device.shell("uname -a").strip().split()[2]
        log.info(f'Kernel version {kernel_version}')

        selinux = self.device.shell("getenforce").strip()
        log.info(f'SELinux status: {selinux}')

        module_present = bool(self.device.shell('lsmod | grep lisa'))
        log.info(f"Lisa module {'' if module_present else 'not '}loaded")

        big_temp = self.device.shell("cat /sys/class/thermal/thermal_zone0/temp").strip()
        mid_temp = self.device.shell("cat /sys/class/thermal/thermal_zone1/temp").strip()
        ltl_temp = self.device.shell("cat /sys/class/thermal/thermal_zone2/temp").strip()
        log.info(f"Temperature: BIG {big_temp} MID {mid_temp} LTL {ltl_temp}")

        ltl_cpufreq = self.device.shell("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").strip()
        mid_cpufreq = self.device.shell("cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor").strip()
        big_cpufreq = self.device.shell("cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor").strip()
        log.info(f"cpufreq governor: BIG {big_cpufreq} MID {mid_cpufreq} LTL {ltl_cpufreq}")

        ls_system = self.device.shell("cat /dev/cpuctl/system/cpu.uclamp.latency_sensitive").strip()
        ls_fg = self.device.shell("cat /dev/cpuctl/foreground/cpu.uclamp.latency_sensitive").strip()
        ls_bg = self.device.shell("cat /dev/cpuctl/background/cpu.uclamp.latency_sensitive").strip()
        ls_sysbg = self.device.shell("cat /dev/cpuctl/system-background/cpu.uclamp.latency_sensitive").strip()
        log.info(f"latency_sensitive system: {ls_system} fg: {ls_fg} bg: {ls_bg} sys-bg: {ls_sysbg}")

        idle_governor = self.device.shell("cat /sys/devices/system/cpu/cpuidle/current_governor_ro").strip()
        log.info(f"cpuidle governor: {idle_governor}")

        # TODO: use cgroups in config
        cpuset_bg = self.device.shell("cat /dev/cpuset/background/cpus").strip()
        cpuset_fg = self.device.shell("cat /dev/cpuset/foreground/cpus").strip()
        cpuset_sbg = self.device.shell("cat /dev/cpuset/system-background/cpus").strip()
        log.info(f"cpusets: background: {cpuset_bg}, foreground: {cpuset_fg}, system-bg: {cpuset_sbg}")

        cpushares_bg = self.device.shell("cat /dev/cpuctl/background/cpu.shares").strip()
        cpushares_fg = self.device.shell("cat /dev/cpuctl/foreground/cpu.shares").strip()
        cpushares_sbg = self.device.shell("cat /dev/cpuctl/system-background/cpu.shares").strip()
        cpushares_sys = self.device.shell("cat /dev/cpuctl/system/cpu.shares").strip()
        log.info(f"cpushares: bg: {cpushares_bg} fg: {cpushares_fg} sys: {cpushares_sys} sys-bg: {cpushares_sbg}")

        if 'schedutil' in [ltl_cpufreq, mid_cpufreq, big_cpufreq]:
            pol_0_rl = self.device.shell("cat /sys/devices/system/cpu/cpufreq/policy0/schedutil/rate_limit_us").strip()
            pol_4_rl = self.device.shell("cat /sys/devices/system/cpu/cpufreq/policy4/schedutil/rate_limit_us").strip()
            pol_6_rl = self.device.shell("cat /sys/devices/system/cpu/cpufreq/policy6/schedutil/rate_limit_us").strip()
            log.info(f"policy rate limits: 0: {pol_0_rl}, 4: {pol_4_rl}, 6: {pol_6_rl}")

    def load_module(self):
        # Try to load the module using modprobe (for in-tree builds
        log.debug('Loading the Lisa module')
        self.device.shell("rmmod lisa")
        modules_base_path = self.config['target']['modules_path'].get()
        modules_dir_count = int(self.device.shell(f"ls {modules_base_path} | wc -l").strip())
        modules_dir_version_cmd = f"ls {modules_base_path} | head -1" if modules_dir_count == 1 else 'uname -r'
        modules_dir_version = self.device.shell(modules_dir_version_cmd).strip()
        modules_path = os.path.join(modules_base_path, modules_dir_version)
        self.device.shell(f"modprobe -d {modules_path} lisa")

        # check if the lisa module is loaded
        module_present = bool(self.device.shell('lsmod | grep lisa'))
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
        log.info('Disabling cpusets for groups background, foreground, system-background and restricted')
        self.device.shell("echo '0-7' > /dev/cpuset/background/cpus")
        self.device.shell("echo '0-7' > /dev/cpuset/foreground/cpus")
        self.device.shell("echo '0-7' > /dev/cpuset/system-background/cpus")
        self.device.shell("echo '0-7' > /dev/cpuset/restricted/cpus")

    def disable_cpushares(self):
        log.info('Setting cpushares to 20480 for groups background and system-background')
        self.device.shell("echo 20480 > /dev/cpuctl/background/cpu.shares")
        self.device.shell("echo 20480 > /dev/cpuctl/system-background/cpu.shares")

    def menu(self):
        log.info('Setting the current cpuidle governor to menu')
        self.device.shell("echo 'menu' > /sys/devices/system/cpu/cpuidle/current_governor")

    def teo(self):
        log.info('Setting the current cpuidle governor to teo')
        self.device.shell("echo 'teo' > /sys/devices/system/cpu/cpuidle/current_governor")

    def latency_sensitive(self):
        log.info('Setting the system cgroup to latency sensitive')
        self.device.shell("echo 1 > /dev/cpuctl/system/cpu.uclamp.latency_sensitive")

    def powersave(self):
        log.info('Setting the cpufreq governor to powersave')
        for cpu in range(8):
            self.device.shell(f"echo 'powersave' > /sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")

    def performance(self):
        log.info('Setting the cpufreq governor to performance')
        for cpu in range(8):
            self.device.shell(f"echo 'performance' > /sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")

    def schedutil(self):
        log.info('Setting the cpufreq governor to schedutil')
        for cpu in range(8):
            self.device.shell(f"echo 'schedutil' > /sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")

    def sugov_rate_limit(self):
        log.info('Setting the sugov rate limit to 500')
        for policy in [0, 4, 6]:
            self.device.shell(f"echo '500' > /sys/devices/system/cpu/cpufreq/policy{policy}/schedutil/rate_limit_us")
