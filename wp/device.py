import logging as log

from ppadb.client import Client as AdbClient
from wp.constants import CONFIG_PATH
from wp.helpers import load_yaml


class WorkloadDevice:
    def __init__(self):
        self.config = load_yaml(CONFIG_PATH)
        self.adb_client = AdbClient(host=self.config['target']['adb_host'], port=int(self.config['target']['adb_port']))
        self.device = self.adb_client.devices()[0]

        log.debug('Restarting adb as root')
        try:
            log.info(self.device.root())
        except RuntimeError as e:
            log.debug(e)

    def status(self):
        log.info('Showing current device status')
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
            policy_0_rate_limit = self.device.shell("cat /sys/devices/system/cpu/cpufreq/policy0/schedutil/rate_limit_us").strip()
            policy_4_rate_limit = self.device.shell("cat /sys/devices/system/cpu/cpufreq/policy4/schedutil/rate_limit_us").strip()
            policy_6_rate_limit = self.device.shell("cat /sys/devices/system/cpu/cpufreq/policy6/schedutil/rate_limit_us").strip()
            log.info(f"policy rate limits: 0: {policy_0_rate_limit}, 4: {policy_4_rate_limit}, 6: {policy_6_rate_limit}")

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

