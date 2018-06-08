# -*- coding: utf-8 -*-
"""
| **@created on:** 16/12/16,
| **@author:** Prathyush SP,
| **@version:** v0.0.1
|
| **Description:**
| DL Timer Class
|
| Sphinx Documentation Status:**
|
..todo::


Benchmark Statistics:
11.	Physical Memory Consumption - RAM	List[Float]	Max Physical Memory (RAM) consumed by the run in Mb/ GB - DONE
12.	Physical Memory Consumption - GPU	List[Float]	Max Physical Memory (GPU) consumed by the run in Mb/ GB
13.	Physical Processing Power Consumption	List[Float]	 Max Physical Power (CPU) consumed by the run in % - DONE
"""

import json
from collections import OrderedDict
import datetime
from multiprocessing import Process
from multiprocessing.managers import BaseManager
import time
import os
import logging
from benchmark.utils import DLTimer, generate_timestamp, TimeUtils
from typeguard import typechecked
import glob
from functools import wraps

logger = logging.getLogger(__name__)


# noinspection PyMissingOrEmptyDocstring
class BenchmarkStats(object):
    """
    | **@author:** Prathyush SP
    |
    | Benchmarking Statistics
    """

    def __init__(self, process_name: str):  # pragma: no cover
        self.process_name = process_name
        self.total_elapsed_time = None
        self.monitor_statistics = OrderedDict()

    def get_monitor_statistics(self):
        return self.monitor_statistics

    def set_monitor_statistics(self, status: OrderedDict):
        self.monitor_statistics = status

    def get_total_elapsed_time(self):
        return self.total_elapsed_time

    def set_total_elapsed_time(self, t):
        self.total_elapsed_time = t

    def info(self):  # pragma: no cover
        return OrderedDict([
            ('total_elapsed_time (secs)', self.total_elapsed_time),
            ('monitor_statistics', self.monitor_statistics)
        ])


class BenchmarkUtil(object):
    """
    | **@author:** Prathyush SP
    |
    | Benchmark Util - 
    | Performs Training and Inference Benchmarks
    """

    @typechecked
    def __init__(self, model_name: str, stats_save_path: str,
                 monitors: list = None, benchmark_interval: int = 1):
        self.model_name = model_name
        self.deployed_monitors = monitors
        self.monitors = None
        self.timestamp = generate_timestamp()
        self.benchmark_interval = benchmark_interval
        self.pid = None
        self.stats_save_path = stats_save_path
        os.system('mkdir -p ' + self.stats_save_path + '../graphs/')

    # @staticmethod
    # def _extract_data_from_timeline(trace_file_list):
    #     fp_dur, bp_dur, tot_time = 0, 0, 0
    #     for f in trace_file_list:
    #         timeline = json.load(open(f), object_pairs_hook=OrderedDict)['traceEvents']
    #         for e, d in enumerate(timeline):
    #             if 'args' in d and 'Apply' in d['args']['name']:
    #                 break_even, break_even_data = e, d
    #                 break
    #             fp_dur += d['dur'] if 'dur' in d else 0
    #             tot_time += d['dur'] if 'dur' in d else 0
    #         for e, d in enumerate(timeline[break_even:]):
    #             bp_dur += d['dur'] if 'dur' in d else 0
    #             tot_time += d['dur'] if 'dur' in d else 0
    #     fp_dur = (fp_dur / len(trace_file_list)) / 10 ** 6
    #     bp_dur = (bp_dur / len(trace_file_list)) / 10 ** 6
    #     return {
    #         'sfp_time': fp_dur,
    #         'sbp_time': bp_dur,
    #         'single_batch_time': (fp_dur + bp_dur) / 10 ** 6,
    #         'total_time_elapsed': tot_time / 10 ** 6
    #     }

    @typechecked
    def _attach_monitors(self, pid: int):
        """
        | **@author:** Prathyush SP
        |
        | Attach Various Monitors and waits
        :param pid: Process Id
        """

        if self.deployed_monitors:
            # Initialize Monitors
            self.monitors = [
                monitor(pid=pid, interval=self.benchmark_interval) if isinstance(monitor, type) else monitor
                for monitor in self.deployed_monitors]

            # Start Monitors
            for monitor in self.monitors:
                monitor.start()

            # Wait for Monitors
            for monitor in self.monitors:
                monitor.join()

    def _collect_monitor_stats(self):
        """
        | **@author:** Prathyush SP
        |
        | Collect Monitor Statistics
        """
        if self.monitors:
            return OrderedDict([(monitor.monitor_type, monitor.monitor_stats()) for monitor in self.monitors])
        return None

    def monitor(self, f):
        """
        | **@author:** Prathyush SP
        |
        | Value Exception Decorator.
        :param f: Function
        :return: Function Return Parameter
        """

        @wraps(f)
        def wrapped(*args, **kwargs):
            start = time.time()
            print('Running Benchmark - Training . . .')
            BaseManager.register('BenchmarkStats', BenchmarkStats)
            manager = BaseManager()
            manager.start()
            b_stats = manager.BenchmarkStats(self.model_name)
            try:
                p = Process(target=f, args=())
                p.start()
                self.pid = p.pid
                self._attach_monitors(pid=p.pid)
                p.join()
                b_stats.set_monitor_statistics(self._collect_monitor_stats())
                b_stats.set_total_elapsed_time(time.time() - start)
                json.dump(b_stats.info(), open(self.stats_save_path + '/benchmark_training.json', 'w'), indent=2)
                print('Benchmark Util - Training completed successfully. Results stored at: {}'.format(
                    self.stats_save_path + '/benchmark_training.json'))
            except ValueError as ve:
                logger.error('Value Error - {}'.format(ve))
                raise Exception('Value Error', ve)

        return wrapped

    def clean_up(self):
        """
        | **@author:** Prathyush SP
        |
        | Cleanup operations after benchmarking
        """
        pass  # pragma: no cover
