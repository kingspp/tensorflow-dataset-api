# -*- coding: utf-8 -*-
"""
| **@created on:** 08/06/18,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** --
|
..todo::
"""

import matplotlib.pyplot as plt
import json


class Plotter(object):
    def __init__(self, plot_title: str, data_dict: dict):
        self.clear_plot()
        self.plot_title = plot_title
        self.data_dict = {k: json.load(open(v)) for k, v in data_dict.items()}

    def plot_cpu_values(self):
        plt.title('CPU Performance')
        plt.ylabel('CPU Utilization %')
        plt.xlabel('Time in secs')
        for k, v in self.data_dict.items():
            cpu_usage = v['monitor_statistics']['cpu_monitor']['cpu_usage_per_second (%/s)']
            plt.plot(range(len(cpu_usage)), cpu_usage, linewidth=2, markersize=3, label=k)
        # plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))
        return plt

    def plot_memory_values(self):
        plt.title('RAM Performance')
        plt.ylabel('RAM Utilization MB')
        plt.xlabel('Time in secs')
        for k, v in self.data_dict.items():
            ram_usage = v['monitor_statistics']['memory_monitor']['memory_usage_per_second (MBps)']
            plt.plot(range(len(ram_usage)), ram_usage, linewidth=2, markersize=3, label=k)
        # plt.legend()
        return plt

    def plot_gpu_values(self):
        plt.title('GPU Performance')
        plt.ylabel('GPU Utilization (MB)')
        plt.xlabel('Time in secs')
        for k, v in self.data_dict.items():
            gpu_usage = v['monitor_statistics']['gpu_monitor']['gpu_memory_usage_per_interval (in MiB)'][
                'GPU-46e22c0b-2df8-c403-0245-d2c523489343']
            plt.plot(range(len(gpu_usage)), gpu_usage, linewidth=2, markersize=3, label=k)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return plt

    def plot_time(self):
        plt.title('Time Analysis')
        plt.ylabel('Time Consumption')
        plt.xlabel('Episodes')
        count = 0
        for k, v in self.data_dict.items():
            plt.scatter(count, v['total_elapsed_time (secs)'], label=k)
            count += 1
        # plt.legend()
        return plt

    def plot_all(self):
        plt.figure(figsize=(15,10))
        plt.subplot(221)
        self.plot_cpu_values()
        plt.subplot(222)
        self.plot_gpu_values()
        plt.subplot(223)
        self.plot_memory_values()
        plt.subplot(224)
        self.plot_time()
        return plt

    def clear_plot(self):
        plt.clf()
        plt.cla()
        plt.close()


GLOBAL_PATH = '/Users/prathyushsp/Development/stats/'
only_cpu_data_dict = {
    'EP1 Basic Placeholder': '/Users/prathyushsp/Development/stats/benchmark_20180608_134134_basic_placeholders_cpu.json',
    'EP3 Feedable Dataset': '/Users/prathyushsp/Development/stats/benchmark_ep3_cpu.json',
    'EP5 Feedable Iterator': '/Users/prathyushsp/Development/stats/benchmark_20180608_144702_ep5_cpu.json',
    'EP11 ReInitializable Iterator': '/Users/prathyushsp/Development/stats/benchmark_20180608_151011_ep11_cpu.json'
}

cpu_gpu_data_dict = {
    'EP1 Basic Placeholder': '/Users/prathyushsp/Development/stats/benchmark_20180608_155112_ep1_gpu.json',
    'EP3 Feedable Dataset': '/Users/prathyushsp/Development/stats/benchmark_20180608_143449_ep3_gpu.json',
    'EP5 Feedable Iterator': '/Users/prathyushsp/Development/stats/benchmark_20180608_145638_ep5_gpu.json',
    'EP11 ReInitializable Iterator': '/Users/prathyushsp/Development/stats/benchmark_20180608_151831_ep11_gpu.json'
}

# Plotter('', only_cpu_data_dict).plot_cpu_values().show()
Plotter('', only_cpu_data_dict).plot_all().savefig('plot.png')

# d = json.load(open('/private/tmp/stats/benchmark_training.json'))
# cpu_usage = d['monitor_statistics']['gpu_monitor']['cpu_usage_per_second (%/s)']
#
# plt.plot(range(len(cpu_usage)), cpu_usage, 'go-', linewidth=2, markersize=2)
# plt.title('CPU Performance')
# plt.ylabel('CPU Utilization %')
# plt.xlabel('Time in secs')
# plt.show()
# plot(x, y, color='green', marker='o', linestyle='dashed',
#         linewidth=2, markersize=12)
