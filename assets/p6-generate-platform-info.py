import os
from lisa.target import Target, TargetConf
from lisa.trace import Trace
from lisa.energy_model import LinuxEnergyModel
from lisa._kmod import LISAFtraceDynamicKmod

target = Target.from_one_conf('/home/kajpuc01/tools/lisa/target_conf_p6.yml')
kmod = target.get_kmod(LISAFtraceDynamicKmod)

with kmod.run(kmod_params={'features': ['__em_sysfs']}):
    target = Target.from_one_conf('/home/kajpuc01/tools/lisa/target_conf_p6.yml')
    plat_info = target.plat_info
    plat_info.eval_deferred(error='log')
    print(plat_info)
    path = 'p6-platform-info.yml'
    plat_info.to_yaml_map(path)
