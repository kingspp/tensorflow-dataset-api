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
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return plt

    def plot_time(self):
        plt.title('Time Analysis')
        plt.ylabel('Time Consumption')
        plt.xlabel('Episodes')
        # count = 0
        # et = [v['total_elapsed_time (secs)'] for v in self.data_dict.values()]
        # plt.plot(range(len(et)), et, label='Total Time')
        # for k, v in self.data_dict.items():
        #     plt.scatter(count, v['total_elapsed_time (secs)'], label=k)
        #     count += 1

        count = 0
        et = [v['internal_time'] for v in self.data_dict.values()]
        plt.plot(range(len(et)), et, label='Internal Time')
        for k, v in self.data_dict.items():
            plt.scatter(count, v['internal_time'], label=k)
            count += 1
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return plt

    def plot_all(self):
        plt.figure(figsize=(25, 20))
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


json_root = "/Users/prathyushsp/Development/logs_stats/stats/"
only_cpu_data_dict = {
    'EP1 Basic Placeholder': json_root + 'benchmark_EP1_Basic_Placeholder_cpu_20180612_165810.json',
    'EP3 Feedable Dataset': json_root + 'benchmark_EP3_Feedable_Dataset_cpu_20180612_170759.json',
    'EP5 Feedable Iterator': json_root + 'benchmark_EP5_Feedable_Iterator_cpu_20180612_171953.json',
    'EP11 ReInitializable Iterator': json_root + 'benchmark_EP11_Reinitializable_iterator_switch_cpu_20180612_172811.json',
    'EP14 Feedable Iterator MD': json_root + 'benchmark_EP14_Feedable_Iterator_Multiple_Dataset_cpu_20180612_173607.json',
    # 'EP15 Feedable Iterator MD II': json_root + 'benchmark_EP15_Feedable_Iterator_Multiple_Dataset_Initializable_Iterator_cpu_20180611_154107.json',
    'EP17 Replace Placeholder': json_root + 'benchmark_EP17_Replaceable_Placeholder_cpu_20180612_174438.json',
    'EP19 Feedable Iterator Fetch OP': json_root + 'benchmark_EP19_Feedable_Iterator_Fetch_OP_cpu_20180612_175358.json'

}

cpu_gpu_data_dict = {
    'EP1 Basic Placeholder': json_root + 'benchmark_EP1_Basic_Placeholder_gpu_20180612_170313.json',
    'EP3 Feedable Dataset': json_root + 'benchmark_EP3_Feedable_Dataset_gpu_20180612_171349.json',
    'EP5 Feedable Iterator': json_root + 'benchmark_EP5_Feedable_Iterator_gpu_20180612_172416.json',
    'EP11 ReInitializable Iterator': json_root + 'benchmark_EP11_Reinitializable_iterator_switch_gpu_20180612_173228.json',
    'EP14 Feedable Iterator MD': json_root + 'benchmark_EP14_Feedable_Iterator_Multiple_Dataset_gpu_20180612_174028.json',
    # 'EP15 Feedable Iterator MD II': json_root + 'benchmark_EP15_Feedable_Iterator_Multiple_Dataset_Initializable_Iterator_gpu_20180611_154536.json',
    'EP17 Replace Placeholder': json_root + 'benchmark_EP17_Replaceable_Placeholder_gpu_20180612_174907.json',
    'EP19 Feedable Iterator Fetch OP': json_root + 'benchmark_EP19_Feedable_Iterator_Fetch_OP_gpu_20180612_175837.json'
}

# Plotter('', only_cpu_data_dict).plot_cpu_values().show()
Plotter('', only_cpu_data_dict).plot_all().savefig('plot-cpu.png')
Plotter('', cpu_gpu_data_dict).plot_all().savefig('plot-gpu.png')

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
